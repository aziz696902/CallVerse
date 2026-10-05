# CallVerse Calibration Validation

## Chronological holdout

The 363 usable Technion dates were sorted chronologically once. The first 290 dates
(80%), 1999-01-01 through 1999-10-19, form the calibration period; the final 73 dates,
1999-10-20 through 1999-12-31, are held out. After the same cleaning rules, these
periods contain 340,851 and 94,934 calls. The holdout was not used to tune repeatedly.
The permanent full-data profile is rebuilt only after this component validation.

## Arrival process

A single homogeneous all-day Poisson rate is clearly inappropriate. When every
date/slot count is pooled, variance-to-mean dispersion is 29.88 in calibration and
28.30 in validation because demand changes strongly by time of day. The normalized
48-slot shapes nevertheless transfer reasonably: calibration-versus-validation MAE is
0.0735 multiplier units (raw mean-count MAE 3.187 calls/slot).

Within fixed half-hour slots, median dispersion remains 13.63 in calibration and 8.39
in validation, well above the Poisson ideal of 1. The piecewise Poisson process is
therefore a useful first-order approximation to timing, not a claim of perfect Poisson
randomness. It captures the stable daily shape but under-represents day-to-day
overdispersion; scenarios retain responsibility for stress and total demand.

## Horizon normalization

Before validation, an 08:00 scenario inherited above-average bank weights and therefore
generated roughly 1.8–1.9 times its intended workload. The engine now divides each raw
slot weight by its time-weighted mean over the exact scenario horizon:

`normalized_weight(t) = raw_weight(t) / ((1/T) × integral[0,T] raw_weight(u) du)`

Consequently the normalized weight integrates to `T`; base rate × demand multiplier
controls expected total requests, while Technion controls when they arrive. Counts
remain stochastic and need not match the prototype exactly.

## Service-time representation

The validation set has 77,868 served calls: median 2.017 minutes, p90 7.167, and p95
10.467. Models fitted on 271,424 calibration calls were sampled with a fixed seed:

| Representation | KS | Wasserstein (min) | Median error | p90 error | p95 error |
|---|---:|---:|---:|---:|---:|
| Phase 3 triangular | 0.719 | 3.879 | +4.239 | +1.176 | -1.638 |
| Exponential | 0.091 | 0.533 | +0.162 | +0.026 | -1.126 |
| Log-normal | 0.137 | 0.882 | -0.342 | +1.350 | +3.165 |
| Moment-fit gamma | 0.280 | 0.768 | -0.719 | +1.452 | +1.947 |
| Compact empirical quantiles | **0.066** | **0.210** | **-0.133** | **-0.135** | **-0.249** |

Compact empirical quantiles remain the final representation: they reproduce the held-
out center and upper quantiles best without imposing a questionable named family.
Additional extreme-tail knots prevent interpolation from p99 directly to an outlier.
Intent-specific relative modifiers remain provisional delivery-domain assumptions.

## Patience and abandonment

Abandonment remains the observed patience event; served waiting remains right-censored.
Calibration contains 50,216 events and 151,585 censored calls; validation contains
12,589 events and 41,477 censored calls. Survival probabilities are:

| Minutes | Calibration | Validation | Absolute difference |
|---:|---:|---:|---:|
| 1 | 82.15% | 84.16% | 2.01 pp |
| 2 | 71.63% | 74.20% | 2.57 pp |
| 5 | 53.28% | 57.95% | 4.67 pp |
| 10 | 37.69% | 42.00% | 4.31 pp |

Descriptive queued-call abandonment was 24.88% in calibration and 23.28% in validation.
The direction and scale remain plausible, with modest temporal drift.

## Scope and limitations

Validation is component-level. Reliable historical staffing schedules/capacity are not
available, so reproducing whole-center queue KPIs would create false precision. Olist
remains descriptive rather than predictive; rerunning the full pipeline reproduced
96,470 eligible orders, a late rate of 8.112%, and the order-grain review statistics.
Neither dataset proves that lateness causes support contact or poor reviews.
