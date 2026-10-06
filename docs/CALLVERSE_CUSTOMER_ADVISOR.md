# CallVerse Customer Advisor Integration

## Scope and architecture

Phase 6 integrates the CallVerse intent classifier with HelpPilot; it does not
replace or duplicate HelpPilot. `CallVerseCustomerAdvisor` implements the
existing CallVerse `Advisor` boundary and keeps LangGraph-specific state behind
the adapter.

The runtime flow is:

```text
customer message
  -> CallVerse classification and deterministic order-ID extraction
  -> confidence/support policy
     -> accepted: explicit structured intent context
     -> uncertain/unsupported/unavailable: unchanged message to HelpPilot triage
  -> existing HelpPilot graph
  -> existing business tools and/or RAG
  -> answer, approval interrupt, or escalation
  -> CallVerse AdvisorResult + compact InteractionMetadata
```

The six trained classifier intents are `tracking`, `refund`, `cancel_order`,
`address_change`, `payment_issue`, and `complaint`. A prediction is accepted
only when it is one of those labels, meets the metadata threshold (currently
0.50), and is not marked `needs_review`.

Damaged-item language and general greetings are explicit unsupported/ambiguous
guards. Low confidence, an absent model, and inference failure also select the
HelpPilot fallback. The fallback is observable through `routing_path` and
`fallback_reason`; it is never a silent guess.

## HelpPilot components reused unchanged

The integration reuses these inherited components:

- `AgentState` and the compiled LangGraph workflow;
- the LLM triage node and its `answer_directly`, `use_tools`, and
  `escalate_to_human` routes;
- the solver node and its existing ReAct tool selection;
- the approval interrupt/resume node;
- the grounding/PII reviewer and retry path;
- the deterministic response/escalation node;
- the public `run_turn`, `resume_turn`, and `peek_state` entry points;
- SQLite customer, order, ticket, action, approval, and response storage;
- `get_order`, `get_tracking`, `check_refund_policy`,
  `create_refund_draft`, `log_to_crm`, `issue_refund`, and `send_reply`;
- the existing Chroma retrieval, sentence-transformer embeddings, and
  cross-encoder reranker;
- the existing Streamlit application, which continues to call the inherited
  graph and was not redesigned in this phase.

No HelpPilot graph node was rewritten. The only inherited-source additions are
two small synthetic demonstration policies and one delayed seed order.

## Routing and factual safety

For an accepted prediction, the original message is prefixed with a compact
`CALLVERSE ROUTING CONTEXT` containing the accepted intent, extracted order ID,
and provisional urgency. It instructs HelpPilot to verify customer facts with
tools. HelpPilot still performs its internal safety triage for compatibility;
the observable trace records both the classifier prediction and HelpPilot route
so they cannot silently disappear into one another.

If the message or request contains an order ID, the orchestrator first invokes
HelpPilot's existing `get_order` tool. A missing order, lookup failure, or
customer/order mismatch stops processing and safely escalates; no status is
invented. The graph may then use `get_tracking` or another business tool for
customer-specific facts. RAG is reserved for policy, procedure, FAQ, and
escalation guidance rather than being treated as an order database.

Refunds remain protected. The solver can create a refund draft but cannot call
`issue_refund`. The inherited graph calls that sensitive function only after a
human approval interrupt is resumed. The adapter maps an interruption to an
escalated `AdvisorResult`.

## Knowledge base and demonstration data

The inherited documents already cover shipping/tracking, delays, refunds,
cancellation, missing packages, returns, payments, warranties, and escalation.
Phase 6 adds only:

- `callverse-address-change-demo`;
- `callverse-damaged-item-demo`.

Both documents identify themselves as synthetic CallVerse demonstration
policies, not real courier-company rules. The seed data remains small: five
customers and six orders, including in-transit, delivered, processing, lost,
and delayed examples. A nonexistent ID is intentionally represented by absence,
not a fake record.

## Observable result contract

`AdvisorResult` remains the stable external result. `handle_with_trace` adds a
separate compact `InteractionMetadata` object with only observable execution
facts:

- routing path, classifier intent/confidence, acceptance, and fallback reason;
- extracted order ID and provisional urgency;
- HelpPilot status and route;
- tool names and RAG citation IDs;
- final resolved/escalated flags.

It contains no chain-of-thought, prompts from internal graph nodes, or private
model reasoning. `handle_support_request` continues to return only
`AdvisorResult`, satisfying the original protocol.

## Offline verification and demo

Integration tests inject deterministic classifier outcomes and a fake
HelpPilot runner, exercising classification policy, routing, adapter mapping,
order preflight, result mapping, and metadata without Groq. Only the external
advisor/LLM boundary is replaced; the CallVerse orchestration itself is real.

The selected real classifier can be exercised with the deterministic demo:

```powershell
python -m callverse.customer_advisor_demo "Track my order ORD-5003" --customer-id CUST-1003
```

The command clearly labels its output `DETERMINISTIC OFFLINE DEMO`; it does not
claim to generate a live HelpPilot answer. On 2026-10-06, `GROQ_API_KEY` was not
available in the local configuration, so live LLM execution was not verified.

## Model artifact and provenance

The selected `tfidf_logistic.joblib` is 133,576 bytes. The source Bitext data is
licensed under CDLA-Sharing-1.0. Section 1.11 defines computational outputs as
“Results,” and section 3.5 states that the license places no obligations or
restrictions on using or publishing Results when they contain no more than a
de minimis portion of the data. Based on those clauses, the small trained model
is treated as a derived Result and is tracked for private-repository
reproducibility. This is an implementation/licensing interpretation, not legal
advice. See the [official CDLA-Sharing-1.0 text](https://cdla.dev/sharing-1-0/).

Only the selected model and existing metadata are trackable. Raw Bitext data,
transformer weights, caches, intermediate artifacts, and checkpoints remain
ignored.

## Adaptation from the presentation

The presentation proposed bge-m3 + Qdrant + Llama 3.1 8B. The implemented
system currently reuses HelpPilot's existing retrieval/vector stack and LLM
provider because replacing a functioning RAG stack purely to match the proposal
would add complexity without improving the research objective. In particular,
Phase 6 retains Chroma, HelpPilot's sentence-transformer embedding/reranking
pipeline, and its configured Groq models. No Qdrant stack or second agent graph
was introduced.
