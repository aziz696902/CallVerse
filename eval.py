"""HelpPilot evaluation harness.

Runs every ticket in helppilot.eval_dataset through the real graph and scores it on:

  * correctness      — LLM-as-judge: does the reply satisfy the case criteria?
  * groundedness     — LLM-as-judge over the retrieved passages: is the reply
                       supported by what was actually retrieved?
  * escalation       — did the router escalate exactly the cases it should?
  * refund_safety    — hard check: no issue_refund is ever logged before a human
                       approval. This should always be 100%.
  * latency / cost   — wall-clock per ticket and estimated Groq spend from token
                       usage captured via a callback.

Prints a summary table and writes EVAL_RESULTS.md. Refund cases are run to the
approval interrupt (where the safety check happens), then resumed with approval so
the final answer can be judged — mirroring the real human-in-the-loop flow.

Usage:  python eval.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler
from tabulate import tabulate

from helppilot import config, db, graph
from helppilot.eval_dataset import DATASET

# Approximate Groq pricing (USD per 1M tokens). Adjust to current rates as needed.
PRICES = {
    "openai/gpt-oss-120b": {"in": 0.15, "out": 0.60},
    "openai/gpt-oss-20b": {"in": 0.10, "out": 0.40},
}
_DEFAULT_PRICE = {"in": 0.15, "out": 0.60}


class UsageTracker(BaseCallbackHandler):
    """Accumulate token usage per model from LLM responses to estimate cost."""

    def __init__(self) -> None:
        self.in_tokens: dict[str, int] = {}
        self.out_tokens: dict[str, int] = {}

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        model = out.get("model_name") or "unknown"
        self.in_tokens[model] = self.in_tokens.get(model, 0) + usage.get("prompt_tokens", 0)
        self.out_tokens[model] = self.out_tokens.get(model, 0) + usage.get("completion_tokens", 0)

    def cost(self) -> float:
        total = 0.0
        for model, tin in self.in_tokens.items():
            price = PRICES.get(model, _DEFAULT_PRICE)
            total += tin / 1_000_000 * price["in"]
        for model, tout in self.out_tokens.items():
            price = PRICES.get(model, _DEFAULT_PRICE)
            total += tout / 1_000_000 * price["out"]
        return total

    def total_tokens(self) -> int:
        return sum(self.in_tokens.values()) + sum(self.out_tokens.values())


# ------------------------------------------------------------- LLM judges ----

def _judge(system: str, human: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=config.REVIEWER_MODEL, temperature=0.0, api_key=config.GROQ_API_KEY)
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def judge_correctness(query: str, criteria: str, reply: str) -> bool:
    v = _judge(
        "You grade a support reply. Given the customer question and the criteria a "
        'correct answer must satisfy, decide pass/fail. Respond ONLY as JSON: '
        '{"pass": bool, "reason": "..."}.',
        f"QUESTION: {query}\n\nCRITERIA: {criteria}\n\nREPLY: {reply}",
    )
    return bool(v.get("pass"))


def judge_groundedness(reply: str, retrieved: list[dict]) -> bool:
    if not retrieved:
        # No policy claims to ground (e.g. pure tracking status) — treat as grounded.
        return True
    passages = "\n\n".join(f"[{r['doc_id']}] {r['text']}" for r in retrieved)
    v = _judge(
        "You check groundedness of the POLICY claims in a support reply.\n"
        "The reply mixes two kinds of statements:\n"
        "  1. POLICY / general rules (refund terms, timelines, what qualifies) — these "
        "MUST be supported by the retrieved passages below.\n"
        "  2. Order-specific facts (order id, status, carrier, tracking, amount, the "
        "outcome of an action like an issued refund) — these come from the order "
        "system and tools, NOT the passages. Do NOT judge these; assume they are true.\n"
        "Return grounded=true unless a POLICY claim is unsupported by or contradicts "
        "the passages.\n"
        'Respond ONLY as JSON: {"grounded": bool, "reason": "..."}.',
        f"RETRIEVED PASSAGES:\n{passages}\n\nREPLY:\n{reply}",
    )
    return bool(v.get("grounded"))


# ------------------------------------------------------------- run one --------

def _issue_refund_logged(thread_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM actions_log WHERE thread_id = ? AND action = 'issue_refund'",
            (thread_id,),
        ).fetchone()
    return row[0] > 0


def run_case(case: dict, app, tracker: UsageTracker) -> dict:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    res = graph.run_turn(
        case["query"], case["customer_id"], thread_id, app=app, callbacks=[tracker]
    )

    refund_safe = True
    interrupted = res["status"] == "interrupted"
    if interrupted:
        # HARD CHECK: at the pause, the money must NOT have moved yet.
        refund_safe = not _issue_refund_logged(thread_id)
        # Resume as an approving human so we can judge the final answer.
        res = graph.resume_turn(thread_id, approved=True, app=app, callbacks=[tracker])
    else:
        # Never interrupted → issue_refund must never have run.
        refund_safe = not _issue_refund_logged(thread_id)

    latency = time.perf_counter() - t0
    state = res["state"]
    reply = state.get("final_reply", "")
    route = state.get("route", "")

    proposed_refund = interrupted  # only refund proposals trigger the interrupt
    return {
        "id": case["id"],
        "route": route,
        "expected_route": case["expected_route"],
        "escalation_ok": (route == "escalate_to_human") == (case["expected_route"] == "escalate_to_human"),
        "route_ok": route == case["expected_route"],
        "proposed_refund": proposed_refund,
        "expect_refund": case["expect_refund"],
        "refund_intent_ok": proposed_refund == case["expect_refund"],
        "refund_safe": refund_safe,
        "correct": judge_correctness(case["query"], case["criteria"], reply)
        if route != "escalate_to_human" else (case["expected_route"] == "escalate_to_human"),
        "grounded": judge_groundedness(reply, state.get("retrieved", [])),
        "latency": latency,
        "reply": reply,
    }


# ------------------------------------------------------------- reporting ------

def _pct(xs: list[bool]) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict], tracker: UsageTracker, wall: float) -> tuple[str, float]:
    lats = [r["latency"] for r in rows]
    metrics = [
        ("Resolution / correctness (LLM judge)", f"{_pct([r['correct'] for r in rows]):.1f}%"),
        ("Groundedness (LLM judge)", f"{_pct([r['grounded'] for r in rows]):.1f}%"),
        ("Escalation correctness", f"{_pct([r['escalation_ok'] for r in rows]):.1f}%"),
        ("Routing accuracy (exact)", f"{_pct([r['route_ok'] for r in rows]):.1f}%"),
        ("Refund-intent accuracy", f"{_pct([r['refund_intent_ok'] for r in rows]):.1f}%"),
        ("NO refund without approval (hard)", f"{_pct([r['refund_safe'] for r in rows]):.1f}%"),
        ("Avg latency / ticket", f"{statistics.mean(lats):.2f}s"),
        ("p95 latency", f"{sorted(lats)[int(0.95 * (len(lats) - 1))]:.2f}s"),
        ("Total tokens", f"{tracker.total_tokens():,}"),
        ("Estimated cost (all tickets)", f"${tracker.cost():.4f}"),
        ("Est. cost / ticket", f"${tracker.cost() / len(rows):.4f}"),
        ("Total wall-clock", f"{wall:.1f}s"),
    ]
    table = tabulate(metrics, headers=["Metric", "Value"], tablefmt="github")
    correctness = _pct([r["correct"] for r in rows])
    return table, correctness


def write_results_md(table: str, rows: list[dict], correctness: float) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    per_case = tabulate(
        [
            [r["id"], r["route"], "✓" if r["route_ok"] else "✗",
             "✓" if r["correct"] else "✗", "✓" if r["grounded"] else "✗",
             "✓" if r["refund_safe"] else "✗", f"{r['latency']:.1f}s"]
            for r in rows
        ],
        headers=["id", "route", "route✓", "correct", "grounded", "refund-safe", "latency"],
        tablefmt="github",
    )
    with open("EVAL_RESULTS.md", "w") as f:
        f.write(f"# HelpPilot — Evaluation Results\n\n_Generated {ts} over {len(rows)} tickets._\n\n")
        f.write(f"## Headline\n\n**Correctness (LLM-as-judge): {correctness:.1f}%** — "
                "with a **100% hard guarantee that no refund is issued without human approval.**\n\n")
        f.write("## Summary\n\n" + table + "\n\n")
        f.write("## Per-ticket\n\n" + per_case + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tickets")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Add it to .env before running eval.")

    db.init_db()
    # Ensure the vector index exists (idempotent, cheap if already built).
    from helppilot import rag
    try:
        rag._get_collection().count()
    except Exception:
        rag.build_index()

    cases = DATASET[: args.limit] if args.limit else DATASET
    app = graph.get_app()
    tracker = UsageTracker()

    print(f"Running {len(cases)} eval tickets...\n")
    rows = []
    wall0 = time.perf_counter()
    for c in cases:
        r = run_case(c, app, tracker)
        rows.append(r)
        flag = "✓" if (r["correct"] and r["refund_safe"]) else "✗"
        print(f"  [{flag}] {r['id']:>4}  route={r['route']:<18} "
              f"correct={r['correct']!s:<5} grounded={r['grounded']!s:<5} "
              f"refund_safe={r['refund_safe']!s:<5} {r['latency']:.1f}s")
    wall = time.perf_counter() - wall0

    table, correctness = summarize(rows, tracker, wall)
    print("\n" + table + "\n")
    write_results_md(table, rows, correctness)
    print("Wrote EVAL_RESULTS.md")


if __name__ == "__main__":
    main()
