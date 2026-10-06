# CallVerse Delivery-Support Intent Classifier V1

## Purpose and boundary

The classifier converts English customer text into a `RequestIntent`, confidence,
review flag, deterministic order ID, and provisional rule-based urgency. It is an
independent CallVerse boundary and is not wired into HelpPilot's live LLM triage yet.
The frozen Digital Twin continues generating known intent labels and does not classify
its own synthetic labels.

## Dataset and cleaning

The official Bitext Customer Support LLM Chatbot Training Dataset revision
`430d1a89bd93bd1fa23c16f29dd53e73f0087443` contains 26,872 English rows, five columns,
11 observed categories, and 27 observed intents. It is hybrid synthetic data under
CDLA-Sharing-1.0. Full provenance and hash are in `DATA_SOURCES.md`.

No values were missing and no complete rows were duplicated. There were 2,237 repeated
instruction/intent pairs whose responses differed. Mapping selected 7,957 rows. A
case/punctuation-insensitive fingerprint removed 1,269 duplicate selected examples,
leaving 6,688. Model text preserves casing, punctuation, misspellings, and informal
language; only Unicode and repeated whitespace are normalized.

The fixed seed is `20261006`. Stratified splits are 4,681 train, 1,003 validation, and
1,004 untouched test rows. Final test counts are: address change 144, cancellation 63,
complaint 150, payment issue 149, refund 384, and tracking 114. Fingerprinted duplicates
cannot cross splits.

## Source-label audit and mapping

The tracked `dataset_audit.json` contains counts and three short examples for every
source intent. The complete mapping decision is:

| Source intent(s) | Count before selected deduplication | CallVerse intent | Decision |
|---|---:|---|---|
| `track_order` | 995 | tracking | DIRECT |
| `cancel_order` | 998 | cancel_order | DIRECT |
| `change_shipping_address` | 973 | address_change | DIRECT |
| `payment_issue` | 999 | payment_issue | DIRECT |
| `complaint` | 1,000 | complaint | DIRECT |
| `check_refund_policy`, `get_refund`, `track_refund` | 2,992 | refund | COMBINED |
| Remaining 19 source intents | 18,915 | — | UNSUPPORTED / EXCLUDED |

Excluded labels are `create_account`, `delete_account`, `edit_account`,
`recover_password`, `registration_problems`, `switch_account`,
`check_cancellation_fee`, `contact_customer_service`, `contact_human_agent`,
`delivery_options`, `delivery_period`, `review`, `check_invoice`, `get_invoice`,
`change_order`, `place_order`, `check_payment_methods`, `set_up_shipping_address`, and
`newsletter_subscription`. They are not forced into weak delivery-support mappings.

Bitext does not support `damaged_item`. Its contact and miscellaneous labels are not a
sound training definition for `general`, so that class is also unsupported. At
inference time, unfamiliar requests can be sent for review via confidence handling.

## Models and untouched-test results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority (`refund`) | 0.3825 | 0.0922 | 0.2116 |
| Character TF-IDF + balanced logistic regression | **1.0000** | **1.0000** | **1.0000** |
| Tiny BERT (`prajjwal1/bert-tiny`, four epochs) | 0.9920 | 0.9841 | 0.9919 |

The transformer was selected only if validation macro-F1 exceeded TF-IDF by more than
0.005. It did not, so the 133,576-byte TF-IDF model is the final V1 artifact and the
temporary transformer checkpoint was removed. Perfect TF-IDF performance reflects a
highly regular synthetic dataset and must not be interpreted as real-world perfection.
The confusion-matrix PNG and complete per-class metrics are tracked.

## Error analysis

The selected TF-IDF model made no errors on this held-out synthetic test split. The
transformer made eight, retained in `transformer_errors.csv` for useful comparison:

- Seven cancellation messages were classified as tracking. Most contained strong
  spelling corruption (`camcel`, `cfanceling`, `canmcel`) or implied cancellation with
  vague wording such as “I no longer want purchase”.
- One misspelled payment message (“pamyent issues”) was classified as address change.

The experiment therefore exposes spelling corruption and vague action wording as
likely failure modes. The source mostly contains single-intent template expansions, so
it provides little evidence about real multi-intent requests, damaged items, or vague
general messages.

## Confidence, extraction, and urgency

The confidence threshold is 0.50, selected on validation data. Validation coverage was
99.80% with 100% accepted accuracy/macro-F1. Test coverage was 100%, so the test review
rate was 0%. This should not be treated as a real traffic review-rate forecast.

`RequestClassifier` retains the predicted intent and marks `needs_review=True` below
the threshold. For example, the unsupported local message “The item arrived broken”
was assigned low confidence 0.415 and sent for review.

Order IDs are extracted deterministically from forms including `ORD-4471`, `order
4471`, and `#4471`; no LLM is used. Bitext has no urgency labels. Urgency is therefore
explicitly `rule_based_provisional_v1`, using transparent deadline, urgent, lost-package,
repeated-complaint, and safety expressions. It is not a trained urgency model.

## Integration and limitations

Future integration is `customer text → RequestClassifier → structured result →
Customer Advisor`. HelpPilot's existing LLM triage remains unchanged to avoid silently
competing classifiers. The classifier is English-only; no French or Arabic performance
has been measured. Before live use it needs genuinely independent, human-authored,
delivery-support evaluation data, especially for unsupported and multi-intent cases.
