# Fixed-Income Risk Engine

[![CI](https://github.com/crollila/fixed-income-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/crollila/fixed-income-risk-engine/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

This is a Treasury and investment-grade corporate bond pricing and portfolio-risk engine, written from scratch in Python. It
builds a yield curve from real U.S. Treasury data, prices every bond from its cash flows, measures interest-rate and
credit risk in dollars, runs stress tests, finds the main ways the curve actually moves (PCA), and then rebalances the
portfolio toward a target risk profile without breaking its investment limits.

> **Simulated portfolio, not real money.** The ~$100MM book, its benchmark, the corporate issuers (fictional names),
> their ratings, spreads and trading costs are all **manufactured for demonstration**. It is not client or real AUM.
> The only market data is the U.S. Treasury par curve (Federal Reserve H.15 via FRED), which is cached in the repo so
> everything runs offline.

Every number below was produced by `python -m firisk.run_all`. The README itself is generated from a template, and a
test checks that it matches the program output exactly.

---

## Headline results (valuation date 1 October 2026)

| | Simulated portfolio | What it means |
|---|---:|---|
| Market value | **$100,000,846** | simulated book: 7 Treasuries + 25 corporate bonds from 13 issuers in 8 sectors |
| Effective duration | **7.23 yrs** | if all yields rise 1%, the book loses roughly 7.23% (benchmark: 5.18) |
| DV01 | **$72,277** per bp | dollars lost if every Treasury par yield rises 0.01% |
| Largest key-rate bucket | **10Y: $17,614** per bp | 24.4% of all rate risk; the 20Y+30Y points add another 44.6% |
| CS01 | **$49,092** per bp | dollars lost if every corporate credit spread widens 0.01% |
| Largest issuer / sector | Atlas Bancorp 7.50% / Financials 19.50% | both above policy limits (6% / 18%) |
| Worst stress | **-$9.17MM** (-9.17%) | Stagflation: rates +100 + IG widening |
| Duration-only error at +100bp | -$425,447 (6.25%) | adding convexity cuts it to $25,330 (0.37%) |
| PCA: level / slope / curvature | 85.1% / 11.0% / 2.1% | share of daily curve-move variance, 2016-01-05 to 2026-10-01 |
| Constrained rebalance | duration 7.23 → **5.30** | key-rate gap to benchmark cut 88%, all 3 limit breaches fixed, 19.99% turnover, $14,688 cost |

---

## How the engine fits together

```mermaid
flowchart LR
    A[FRED Treasury par yields<br/>cached CSV] --> B[Bootstrap<br/>zero curve]
    B --> C[Price every bond<br/>from cash flows]
    U[Simulated bonds<br/>+ z-spreads] --> C
    C --> D[Sensitivities<br/>DV01, key-rate DV01,<br/>CS01, convexity]
    D --> E[Portfolio aggregation<br/>vs benchmark]
    E --> F[Stress tests<br/>full repricing vs approximations]
    A --> G[PCA of daily<br/>curve changes]
    G --> H[Factor risk / VaR]
    D --> H
    E --> I[LP optimizer<br/>with policy constraints]
    I --> E
```

None of the core math wraps a pricing library. Day counts, schedules, yields, bootstrapping, curve interpolation and every
sensitivity are implemented in `src/firisk/`. QuantLib is only used as an *optional* independent cross-check in the
test suite (`tests/test_quantlib_crosscheck.py`).

---

## Fixed income in ten minutes: a guided tour through the outputs

### 1. A bond is a list of promised cash flows

A bond pays a fixed **coupon** every six months and gives back its **face value** ($100 per $100 of bonds) at maturity.
Its fair price today is the sum of each payment's **present value**, meaning what that future dollar is worth now.

![cash flows](figures/fig01_cashflows.png)

Two prices get quoted. **Accrued interest** is the part of the next coupon the seller has already earned since the last
payment date. The buyer pays the **dirty price** (clean + accrued), while screens show the **clean price**, which doesn't
jump on coupon dates. Treasuries count days *Actual/Actual* and U.S. corporates use *30/360*. Both are implemented, with
the month-end and February edge cases.

| Bond | Clean | Accrued | Dirty | YTM | Macaulay | Modified | Effective | Convexity | DV01 /100 | Spread dur |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UST 5.250 08/15/36 | 100.0646 | 0.6705 | 100.7351 | 5.241% | 7.779 | 7.580 | 7.635 | 71.3 | 0.0769 | 0.000 |
| UST STRIP 0 11/15/36 | 58.8680 | 0.0000 | 58.8680 | 5.304% | 10.122 | 9.861 | 9.972 | 103.9 | 0.0587 | 0.000 |
| ATLAS 5.875 03/15/36 | 96.1691 | 0.2611 | 96.4302 | 6.421% | 7.320 | 7.093 | 7.180 | 63.8 | 0.0692 | 7.308 |

*Per $100 face. The Treasury STRIP is a zero-coupon bond: no coupons and no accrued interest, and its duration is close to its maturity.*

### 2. The yield curve: par, spot, forward and discount factors

Treasury publishes **par yields**: the coupon a new bond at each maturity would need to price at exactly $100. Pricing
needs **spot (zero-coupon) rates** instead, meaning the rate for a single payment at time *t*. The engine **bootstraps**
them. It solves the 1M, 3M and 6M bills first, then each coupon tenor in turn, so that every input instrument reprices
to par (largest error below 1e-12 per $100). **Forward rates** are the rates the curve implies *between* two future
dates. A **discount factor** DF(t) is the value today of $1 paid at *t*.

![curves](figures/fig03_curves.png)

On 1 October 2026 the curve slopes upward (2Y 4.78%, 10Y 5.24%, 30Y 5.61%) with the familiar 20-year hump.
The step-shaped forward curve comes from the interpolation choice (log-linear discount factors give piecewise-flat
forwards). It is easy to see that the 10–20Y forward has to sit high, because the 20Y par yield is above both its neighbours.

### 3. Yield, duration and convexity: how price reacts to rates

**Yield to maturity** is the single discount rate that reproduces a bond's price. When yields go up, prices go down.
**Duration** is the slope of that relationship: a duration of 14 means about a 14% price drop for a 1% rise in yield.
The relationship is curved, though. **Convexity** measures the bend, and for an ordinary bond it always works in the
holder's favour.

![price-yield](figures/fig02_price_yield.png)

The engine reports *Macaulay* duration (the weighted-average time to cash flows), *modified* duration (sensitivity to
the bond's own yield) and *effective* duration (full repricing after shifting the whole Treasury curve and
re-bootstrapping). Effective is slightly larger than modified in the table above because a 1bp move in *par* yields
moves long *zero* rates by a little more than 1bp.

### 4. Risk in dollars: DV01 and key-rate DV01

Traders think in dollars per basis point (1bp = 0.01%). **DV01** is the loss from a 1bp rise in rates. **Key-rate DV01**
breaks that into the points of the curve where the risk lives. Each of the 11 Treasury par tenors is bumped on its own
and the whole book is repriced.

![key-rate DV01](figures/fig05_key_rate_dv01.png)

| Key rate | Portfolio | Benchmark | Active | Rebalanced |
|---|---:|---:|---:|---:|
| 1M | $5 | $3 | $2 | $5 |
| 3M | $22 | $21 | $2 | $21 |
| 6M | -$20 | $369 | -$389 | $369 |
| 1Y | $16 | -$6 | $21 | $21 |
| 2Y | $929 | $1,798 | -$870 | $1,798 |
| 3Y | $1,937 | $5,236 | -$3,299 | $4,051 |
| 5Y | $10,538 | $10,199 | $340 | $10,632 |
| 7Y | $8,994 | $8,396 | $598 | $8,987 |
| 10Y | $17,614 | $10,684 | $6,930 | $12,042 |
| 20Y | $16,813 | $7,090 | $9,723 | $7,089 |
| 30Y | $15,430 | $7,985 | $7,445 | $7,985 |
| **Sum** | **$72,277** | **$51,775** | **$20,502** | **$52,999** |

The simulated book runs **+2.05 years longer** than its benchmark, and almost all of the extra risk sits at
10Y–30Y. The key-rate buckets add back to the parallel DV01: $72,277 against $72,277, a -0.00003% difference.
Portfolio DV01 from repricing the whole book ($72,276.8857) equals the sum of the individual bond DV01s ($72,276.8857).

### 5. Credit risk: spreads, CS01 and concentration

Corporate bonds yield more than Treasuries because the issuer could default. The engine prices each corporate at a
**z-spread** over the Treasury zero curve. **CS01** is the loss if that spread widens by 1bp, and **spread duration** is
the same measure as a percentage. Treasuries are the risk-free reference and carry no credit risk by construction.

| Rating | Weight | CS01 |
|---|---:|---:|
| AA | 12.0% | $10,138 |
| A | 30.5% | $22,528 |
| BBB | 24.5% | $16,426 |
| **Corporates total** | **67.0%** | **$49,092** |

![composition](figures/fig10_composition.png)

The starting book breaks three simulated policy limits: Atlas Bancorp at 7.50% (limit 6%),
Financials at 19.50% (limit 18%) and Treasuries at 33.0% (minimum 35%). The
**maturity ladder** below shows where the money sits by years to maturity:

![ladder](figures/fig11_ladder.png)

### 6. Stress tests: what happens on a bad day, and how good are the shortcuts?

Every scenario fully reprices every bond on a shocked curve. The result is then compared with the two shortcuts risk
desks use: **first-order** (key-rate DV01 × shock + CS01 × spread shock) and **second-order** (first-order plus convexity).

![stress](figures/fig06_stress_pnl.png)

| Scenario | Full repricing | % NAV | Rates | Spread | 1st-order | 1st err | 2nd-order | 2nd err | After rebalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel +25bp | -$1.78MM | -1.78% | -$1.78MM | +$0.00MM | -$1.81MM | -$27,761 | -$1.78MM | $413 | -$1.31MM |
| Parallel +50bp | -$3.50MM | -3.50% | -$3.50MM | +$0.00MM | -$3.61MM | -$109,441 | -$3.50MM | $3,253 | -$2.58MM |
| Parallel +100bp | -$6.80MM | -6.80% | -$6.80MM | +$0.00MM | -$7.23MM | -$425,447 | -$6.78MM | $25,330 | -$5.05MM |
| Parallel -25bp | +$1.84MM | 1.84% | +$1.84MM | +$0.00MM | +$1.81MM | -$28,596 | +$1.84MM | -$422 | +$1.34MM |
| Parallel -50bp | +$3.73MM | 3.73% | +$3.73MM | +$0.00MM | +$3.61MM | -$116,125 | +$3.73MM | -$3,431 | +$2.72MM |
| Parallel -100bp | +$7.71MM | 7.71% | +$7.71MM | +$0.00MM | +$7.23MM | -$479,033 | +$7.68MM | -$28,256 | +$5.58MM |
| Bear steepener (2s -25 / 10s+ +25) | -$1.16MM | -1.16% | -$1.16MM | +$0.00MM | -$1.18MM | -$18,003 | -$1.15MM | $5,729 | -$0.53MM |
| Bull flattener (2s +25 / 10s+ -25) | +$1.20MM | 1.20% | +$1.20MM | +$0.00MM | +$1.18MM | -$18,557 | +$1.20MM | $5,175 | +$0.55MM |
| IG spreads widen (AA+30/A+50/BBB+80) | -$2.65MM | -2.65% | +$0.00MM | -$2.65MM | -$2.74MM | -$94,451 | -$2.64MM | $5,467 | -$2.35MM |
| Risk-off: rates -50 + IG widening | +$0.92MM | 0.92% | +$3.73MM | -$2.65MM | +$0.87MM | -$51,797 | +$0.92MM | -$3,388 | +$0.23MM |
| Stagflation: rates +100 + IG widening | -$9.17MM | -9.17% | -$6.80MM | -$2.65MM | -$9.97MM | -$800,030 | -$9.09MM | $79,073 | -$7.16MM |

*Steepener: 2Y and shorter −25bp, 10Y and longer +25bp, linear in between. Credit widening is tiered by rating
(AA +30, A +50, BBB +80bp).*

Duration alone misses by up to 8.72% of the true P&L. Adding convexity brings the worst miss down to
0.86%. The chart below shows the textbook behaviour: the duration-only error grows with the **square** of the
shock and the convexity-corrected error with its **cube**. Fitted log-log slopes over 2–50bp shocks are
1.99 and 3.00 (theory: 2 and 3).

![convergence](figures/fig08_approx_convergence.png)

For the combined scenarios, P&L is split into a rates part, a spread part and their interaction. In the stagflation case,
-$6.80MM comes from rates, -$2.65MM from spreads and +$0.280MM from the interaction:

![attribution](figures/fig07_attribution.png)

### 7. How the Treasury curve really moves: PCA

Rather than assuming shapes like "parallel" or "steepener", the engine learns them from 2,687 daily changes in
Treasury par yields (2016-01-05 to 2026-10-01, 1Y–30Y). It removes the average change, computes the covariance
matrix and takes its eigenvectors (principal components):

![yield history](figures/fig04_yield_history.png)

![PCA loadings](figures/fig09_pca_loadings.png)

The labels are assigned by counting sign changes across tenors, not by hand:

* **PC1 = level (85.1%)**: every tenor moves the same way (loadings 0.23 to 0.40).
* **PC2 = slope (11.0%)**: the short end and long end move in opposite directions (-0.44 at 1Y, +0.49 at 30Y).
* **PC3 = curvature (2.1%)**: the belly moves against both wings.

Together they explain **98.2%** of daily curve variance. Key-rate DV01 is measured on the same par tenors, so the
portfolio's factor exposure is just a dot product:

| Factor | Empirical shape | Variance explained | Daily sigma (bp along PC) | Portfolio 1-sigma daily P&L | Share of portfolio rate variance | After rebalance 1-sigma P&L |
|---|---:|---:|---:|---:|---:|---:|
| PC1 | level | 85.1% | 13.57 | -$354,899 | 94.5% | -$265,385 |
| PC2 | slope | 11.0% | 4.87 | -$85,350 | 5.5% | -$35,401 |
| PC3 | curvature | 2.1% | 2.13 | $3,368 | 0.0% | -$6,708 |

Rate-risk daily volatility is $365,057, which gives a parametric 99% one-day VaR of $849,250. A first-order
historical simulation, replaying every past day's curve move against today's key-rate DV01s, gives $960,859. The
worst replayed day was -$1,973,182, on 2020-03-17.

### 8. Rebalancing under constraints

A linear program (HiGHS via `scipy.optimize.linprog`) chooses buys and sells that move the book's 11 key-rate durations
and its total duration toward the benchmark's. Every trade has to respect the policy:

* max 6% per corporate issuer and 18% per corporate sector;
* at least 35% in Treasuries;
* one-way turnover of at most 20%;
* max 10% in any one bond, long only;
* the trades must pay for themselves *after* simulated bid-ask costs.

| Constraint | Limit | Before | After |
|---|---:|---:|---:|
| Max issuer weight | <= 6.0% | 7.50% BREACH | 5.99% PASS |
| Max sector weight | <= 18.0% | 19.50% BREACH | 17.05% PASS |
| Min Treasury weight | >= 35.0% | 33.00% BREACH | 40.54% PASS |
| One-way turnover | <= 20.0% | - | 19.99% PASS |
| Max single position | <= 10.0% | 9.00% PASS | 9.00% PASS |
| Long only | >= 0.0% | 0.00% PASS | 0.00% PASS |

| Bond | Side | Face | Market value | Cost | Weight before | Weight after |
|---|---:|---:|---:|---:|---:|---:|
| UST 5.625 08/15/56 | SELL | -3,966,000 | -$4,000,230 | $1,395 | 4.00% | 0.00% |
| UST 4.750 02/15/45 | SELL | -4,413,000 | -$3,999,582 | $935 | 4.00% | 0.00% |
| VRTX 5.400 02/01/46 | SELL | -3,925,000 | -$3,499,560 | $4,491 | 3.50% | 0.00% |
| UST 5.250 08/15/36 | SELL | -2,443,000 | -$2,460,958 | $366 | 6.00% | 3.54% |
| UST STRIP 0 11/15/36 | SELL | -3,397,000 | -$1,999,745 | $302 | 2.00% | 0.00% |
| RIVR 6.000 06/01/55 | SELL | -1,739,000 | -$1,582,215 | $2,716 | 3.50% | 1.92% |
| ATLAS 5.875 03/15/36 | SELL | -1,565,000 | -$1,509,133 | $1,866 | 3.00% | 1.49% |
| KEYST 5.600 05/15/47 | SELL | -1,030,000 | -$933,880 | $1,228 | 3.00% | 2.07% |
| MERID 6.125 10/15/33 | SELL | -11,000 | -$10,927 | $17 | 4.00% | 3.99% |
| HALC 5.950 04/15/54 | SELL | -10,000 | -$9,222 | $16 | 3.50% | 3.49% |
| UST 4.625 09/30/28 | BUY | +4,220,000 | $4,207,676 | $294 | 4.00% | 8.21% |
| UST 4.875 09/15/29 | BUY | +7,825,000 | $7,833,946 | $623 | 0.00% | 7.84% |
| UST 4.250 03/31/27 | BUY | +7,948,000 | $7,948,627 | $437 | 0.00% | 7.95% |

| | Before | After | Benchmark |
|---|---:|---:|---:|
| Effective duration | 7.23 | **5.30** | 5.18 |
| DV01 | $72,277 | **$52,999** | $51,775 |
| CS01 | $49,092 | **$40,876** | $26,043 |
| Key-rate duration gap (sum of absolute differences, yrs) | 2.96 | **0.36** | 0 |
| Yield | 6.06% | 5.80% | 5.50% |
| Parallel +100bp P&L | -$6.80MM | **-$5.05MM** | |
| Stagflation P&L | -$9.17MM | **-$7.16MM** | |

The turnover limit binds (19.99% against 20%), so the book lands close to the target duration but not
exactly on it. Long Treasuries are the cheapest risk to sell (simulated costs of 0.5–3.5bp against 8–22bp for corporates), so
the optimizer cuts duration mostly through Treasuries and keeps most of the credit. DV01 falls 27% and CS01
falls 17%. The cost is -25bp of yield.

---

## What is validated

`pytest` runs 56 test functions. Most of them check the engine against *independent* closed-form math rather
than against its own output:

| Property | Check |
|---|---|
| Zero-coupon identity | price = 100 / (1 + y/2)^(2T) and = 100·DF(T) under a curve |
| Par bond at par | coupon = yield on a coupon date → price 100; bootstrapped par instruments reprice to 100 |
| Closed forms | annuity price formula, closed-form par-bond Macaulay / modified duration, textbook 6%/8%/10y price |
| Monotonicity | price strictly decreases as yield rises (grid of yields, every bond) |
| DV01 | analytic derivative vs central finite difference |
| Convergence | duration+convexity error shrinks ~cubically as the shock shrinks, duration-only ~quadratically |
| Dates & accrual | end-of-month schedules (Feb 28/29 ↔ Aug 31), settle on coupon date (accrued 0, coupon excluded), 30/360 day-31 and February rules, final-period simple yield |
| Key-rate reconciliation | key-rate DV01s sum to parallel DV01 per bond and per portfolio |
| Aggregation | portfolio DV01/CS01/KR by full repricing = sum of positions |
| Optimizer | every limit holds after rounding and costs; infeasible limits raise; costs reduce turnover |
| PCA | recovers known eigenvectors from a synthetic covariance; sign conventions; explained variance sums to 1 |
| Reproducibility | README / ANALYSIS / RESUME_BULLETS re-render exactly from the saved results |
| QuantLib (optional) | clean price, accrued, yield and duration match QuantLib for a Treasury and a 30/360 corporate |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m firisk.run_all      # results/, figures/, README.md, ANALYSIS.md, RESUME_BULLETS.md
pytest                        # full test suite
```

It runs fully offline from `data/cache/fred_treasury_cmt.csv` (2,688 days, 2016-01-04 to 2026-10-01). To
refresh the cache from FRED (this changes the results), run `python -m firisk.data --refresh` and update `VALUATION_DATE`
in `src/firisk/config.py`. If the cache is missing, a clearly labeled synthetic curve history is generated instead.
Seeds are fixed (`SEED = 7` for simulated spreads).

## Repository layout

```
src/firisk/
  daycount.py   ACT/ACT (ICMA), 30/360 US, ACT/365F
  schedule.py   coupon schedules with end-of-month rule
  bond.py       cash flows, accrued, clean/dirty, YTM, Macaulay/modified duration, convexity
  curve.py      zero curve: discount factors, spot, forward rates
  bootstrap.py  par-curve bootstrapping
  market.py     market state + vectorized curve pricing, z-spreads
  risk.py       effective duration/convexity, DV01, key-rate DV01, CS01, spread duration
  universe.py   SIMULATED bonds, portfolio, benchmark, policy, spreads, costs
  portfolio.py  aggregation, concentration, credit exposure, ladder
  scenarios.py  stress scenarios, full repricing vs 1st/2nd-order, attribution
  pca.py        curve PCA, factor risk, VaR
  optimizer.py  constrained LP rebalance + independent constraint checker
  report.py     results files + template rendering;  plots.py figures;  run_all.py entry point
  compare.py    tolerance-based results comparison used by CI
templates/      README / ANALYSIS / RESUME_BULLETS templates (every number is a template token)
results/        CSV/JSON outputs (summary.json, holdings, stress, PCA, trades, constraints)
figures/        PNG charts
tests/          pytest suite
```

## Conventions and simplifications

* Settlement = valuation date (T+0). Coupon dates are unadjusted for business days.
* The Treasury curve is bootstrapped from H.15 constant-maturity par yields on a stylized semiannual grid: bills ≤ 6M
  are simple-interest zeros, and coupon tenors are par bonds with exact 0.5-year periods. Interpolation is log-linear in
  discount factors.
* Corporate prices are generated from synthetic z-spreads (rating base + term premium + sector + seeded issuer noise),
  so spreads and prices are internally consistent but not market observations.
* Stress P&L is instantaneous (no carry or roll-down). Spread shocks are parallel within each rating tier.
* PCA uses 1Y–30Y. With the bills included, the third factor turns into a 1M-bill factor (loading -0.87 at 1M;
  explained variance 72.7% / 10.6% / 9.0%), driven by debt-ceiling and policy episodes. See
  [ANALYSIS.md](ANALYSIS.md).
* Historical VaR is first order (key-rate DV01 × historical changes) and covers rates only.

## Data

U.S. Treasury constant-maturity yields: Board of Governors of the Federal Reserve System, H.15 Selected Interest Rates,
retrieved from FRED (Federal Reserve Bank of St. Louis): series DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7,
DGS10, DGS20, DGS30. Retrieval metadata is in `data/cache/fred_treasury_cmt.meta.json`.

## Glossary

| Term | Plain-English meaning |
|---|---|
| bp | basis point, 0.01% |
| Clean / dirty price | price without / with the accrued coupon |
| YTM | the single discount rate that reproduces the price |
| Spot (zero) rate | rate for one payment at a single future date |
| Forward rate | rate locked in today for borrowing between two future dates |
| Duration | % price change for a 1% (100bp) parallel yield move |
| Convexity | how much duration itself changes as yields move (the curvature) |
| DV01 / PV01 | $ change for a 1bp rate move |
| Key-rate DV01 | DV01 for a bump at one maturity point only |
| Z-spread | constant extra discount rate over the Treasury zero curve that matches the bond's price |
| CS01 / spread duration | $ / % change for a 1bp move in the credit spread |
| PCA | statistical method that finds the few independent patterns explaining most curve moves |
| Turnover | fraction of the book traded (one-way: buys only) |

## License

MIT. See [LICENSE](LICENSE).
