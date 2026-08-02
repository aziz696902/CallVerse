# HelpPilot — Evaluation Results

_Generated 2026-08-02 15:04 UTC over 30 tickets._

## Headline

**Correctness (LLM-as-judge): 86.7%** — with a **100% hard guarantee that no refund is issued without human approval.**

## Summary

| Metric                               | Value   |
|--------------------------------------|---------|
| Resolution / correctness (LLM judge) | 86.7%   |
| Groundedness (LLM judge)             | 100.0%  |
| Escalation correctness               | 100.0%  |
| Routing accuracy (exact)             | 96.7%   |
| Refund-intent accuracy               | 100.0%  |
| NO refund without approval (hard)    | 100.0%  |
| Avg latency / ticket                 | 16.29s  |
| p95 latency                          | 34.08s  |
| Total tokens                         | 112,701 |
| Estimated cost (all tickets)         | $0.0241 |
| Est. cost / ticket                   | $0.0008 |
| Total wall-clock                     | 608.4s  |

## Per-ticket

| id   | route             | route✓   | correct   | grounded   | refund-safe   | latency   |
|------|-------------------|----------|-----------|------------|---------------|-----------|
| t01  | use_tools         | ✓        | ✓         | ✓          | ✓             | 14.7s     |
| t02  | use_tools         | ✓        | ✓         | ✓          | ✓             | 8.6s      |
| t03  | use_tools         | ✓        | ✗         | ✓          | ✓             | 62.4s     |
| t04  | use_tools         | ✓        | ✓         | ✓          | ✓             | 29.7s     |
| t05  | use_tools         | ✓        | ✓         | ✓          | ✓             | 13.6s     |
| t06  | use_tools         | ✓        | ✓         | ✓          | ✓             | 9.0s      |
| t07  | use_tools         | ✓        | ✓         | ✓          | ✓             | 11.6s     |
| t08  | use_tools         | ✓        | ✗         | ✓          | ✓             | 8.6s      |
| t09  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 32.7s     |
| t10  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 15.4s     |
| t11  | answer_directly   | ✓        | ✗         | ✓          | ✓             | 6.5s      |
| t12  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 12.0s     |
| t13  | use_tools         | ✗        | ✓         | ✓          | ✓             | 14.0s     |
| t14  | answer_directly   | ✓        | ✗         | ✓          | ✓             | 6.6s      |
| t15  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 34.1s     |
| t16  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 18.2s     |
| t17  | escalate_to_human | ✓        | ✓         | ✓          | ✓             | 0.6s      |
| t18  | escalate_to_human | ✓        | ✓         | ✓          | ✓             | 0.8s      |
| t19  | escalate_to_human | ✓        | ✓         | ✓          | ✓             | 0.4s      |
| t20  | escalate_to_human | ✓        | ✓         | ✓          | ✓             | 0.6s      |
| t21  | use_tools         | ✓        | ✓         | ✓          | ✓             | 28.4s     |
| t22  | use_tools         | ✓        | ✓         | ✓          | ✓             | 26.1s     |
| t23  | use_tools         | ✓        | ✓         | ✓          | ✓             | 8.5s      |
| t24  | use_tools         | ✓        | ✓         | ✓          | ✓             | 15.8s     |
| t25  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 72.1s     |
| t26  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 4.7s      |
| t27  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 3.5s      |
| t28  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 2.0s      |
| t29  | answer_directly   | ✓        | ✓         | ✓          | ✓             | 2.7s      |
| t30  | use_tools         | ✓        | ✓         | ✓          | ✓             | 24.8s     |
