# Local Data Layout

Raw source data is intentionally untracked. Download it from the official sources in
[`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md) and place files as follows:

```text
data/raw/technion_anonymous_bank/*1999.txt
data/raw/olist/olist_orders_dataset.csv
data/raw/olist/olist_order_reviews_dataset.csv
data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

Build all compact artifacts from the repository root:

```bash
python -m callverse.calibration.build
```

The command validates schemas and cleaning rules, keeps the two datasets separate,
and writes aggregate JSON/CSV summaries under `data/processed/calibration/`. Raw IDs,
review text, and credentials are not written to processed artifacts. Download archives
may be removed after their required files and hashes have been verified because the
official source and reproducible extraction layout are documented.

Phase 5 prepares the ignored Bitext dataset and deterministic 70/15/15 splits under
`data/processed/classification/`. Only the small `evaluation/` audit, metrics,
confusion matrix, and compact error samples are tracked. Rebuild them with:

```bash
python -m callverse.classification.train
```
