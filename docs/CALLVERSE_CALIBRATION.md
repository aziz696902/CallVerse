# CallVerse Calibration V1

## Design and provenance

CallVerse combines mechanisms, not records. Technion is a 1999 bank call center and is
used only for transferable contact-center mechanics. Olist is an e-commerce marketplace
and is used only for delivery/order/review behavior. They describe unrelated people and
organizations, so no row-wise merge is performed. CallVerse creates a new synthetic
delivery-support environment from these separately calibrated mechanisms. Full source
and use constraints are recorded in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Cleaning and data quality

Technion's 12 documented tab-separated monthly files contained 444,448 rows across 363
observed dates. The loader validates the documented timestamps/durations and outcome.
It retained 435,785 calls and excluded 8,663: 3,581 phantom calls, 2,360 impossible
timestamp orderings, 1,108 service durations on abandoned calls, 694 invalid service
durations, 557 missing service timestamps, 350 negative durations, 12 malformed values,
and one queue duration without a queue. No duplicate call-grain keys were found. The
source also contained 180,837 calls that did not enter a queue and 87,150 records with
no service timestamp; those are reported as relevant structural missingness and are
only used where their fields support the statistic.

Olist orders contained 99,441 rows. Analysis retained 96,470 delivered orders with the
required timestamps and excluded 2,963 non-delivered orders plus 8 delivered orders
missing actual delivery time. No duplicate order IDs or impossible delivery-before-
purchase timestamps were detected. Reviews contained 99,224 rows and became 98,673
order-grain reviews; 551 additional reviews were deterministically superseded by the
latest answer timestamp, then creation timestamp and review ID. This prevents a
one-to-many join from multiplying orders. Of eligible orders, 95,824 had a selected
review.

## Measured support-center mechanics

Daily usable volume averaged 1,200.51 calls (median 1,415; p25 529.5; p75 1,678.5), but
absolute bank volume is not transferred to CallVerse. The normalized temporal shape is
transferable: the largest half-hour multiplier was 10:00 at 2.241× the all-day slot
mean, followed by 10:30 at 2.141× and 09:30 at 2.106×. Overnight demand was lowest at
03:30 (0.007×). All 48 overall slots and seven weekday-specific profiles are preserved.

Among 349,292 valid served calls, service time averaged 3.182 minutes; median 1.917,
standard deviation 4.618, p25 1.000, p75 3.717, p90 7.000, p95 10.200, and p99 20.067
minutes. The maximum was 604.85 minutes, so compact empirical quantiles are used rather
than forcing a named distribution.

There were 255,867 valid queued calls. Observed waits averaged 1.645 minutes (median
1.033, p75 2.250, p90 3.883, p95 5.150, p99 8.250). Of those calls, 62,805 abandoned,
an observed 24.546% rate.

Patience is estimated with Kaplan–Meier logic: abandonment is the event; a served
call's observed wait is right-censored because true patience is only known to exceed
that wait. The estimate uses 62,805 events and 193,062 censored observations. Estimated
survival was 82.58% past one minute, 72.18% past two, 54.31% past five, and 38.66% past
ten. Compact inverse-survival quantiles are stored for simulation sampling; the sparse
upper tail is capped at the largest observed queue wait and should not be interpreted
as a universal behavioral law.

## Measured delivery and review behavior

Of 96,470 eligible delivered orders, 7,826 were late: 8.112%. Signed actual-minus-
estimated delay had median -11.948 days (negative means early). For late orders, delay
averaged 9.552 days; median 5.806, p75 11.821, p90 21.536, p95 29.640, and p99 62.787.

Descriptive bins—not learned scientific thresholds—are: on-time/early (`<=0` days),
slightly late (`>0` to 2), moderately late (`>2` to 7), and severely late (`>7`).
Counts were 88,644, 2,117, 2,364, and 3,345 respectively.

Reviewed on-time/early orders (88,163) averaged 4.294/5, versus 2.565/5 for 7,661 late
orders. Using a documented low-review threshold of score `<=2`, rates were 9.222% and
54.066%. The association strengthens across descriptive bands: average scores were
3.916 slightly late, 2.517 moderately late, and 1.727 severely late. This is an
association; the observational data do not establish causation.

## Prototype versus calibrated support policy

The Phase 3 prototype remains the default. The optional calibrated policy replaces the
flat arrival shape with empirical 30-minute multipliers and replaces generic triangular
service/patience shapes with compact empirical quantiles. It intentionally keeps the
prototype absolute base rate (0.75/minute), the SLA, and relative intent/persona
modifiers. Those relative modifiers remain provisional because Technion does not tell
us how long a delivery complaint takes or how patient a CallVerse persona is.

Every preset starts at minute 480 (08:00) as an explicit scenario assumption. Scenario
`demand_multiplier` still scales the base intensity. Olist's 8.112% global lateness is
evidence for calibration and scenario sanity checking, but does not overwrite stress-
test scenarios.

The controlled seed-42 comparison is stored in
`data/processed/calibration/prototype_vs_calibrated.csv`. Slot weights are normalized
over each scenario horizon, so the empirical shape changes when contacts arrive but
does not alter expected total demand. Observed generated counts remain stochastic.
The empirical service baseline materially reduces AHT. Held-out evidence and the
normalization formula are documented in `CALLVERSE_CALIBRATION_VALIDATION.md`.

## Unsupported / Scenario-Calibrated Relationships

Olist has no variable stating that a customer contacted support because delivery was
late. Therefore CallVerse cannot estimate `P(support contact | delivery lateness)` from
Olist. Delivery disruption → greater support demand/tracking contacts remains an
explicit scenario/sensitivity assumption. Olist likewise does not support inference of
complaint-contact probability, refund-contact probability, or support-channel choice.

Other limitations include the age and banking domain of Technion, Brazilian marketplace
and historical-period specificity of Olist, unobserved confounding in reviews, sparse
survival tails, and the deliberate retention of provisional intent/persona modifiers.
