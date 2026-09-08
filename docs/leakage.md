# Leakage to look for in this corpus

A checklist for anyone training or evaluating a forecasting model on the `terrastat` series layer.
Each entry says what the leak is, why it is a leak, what evidence of it is already visible in the
data we have collected, and how it could be detected.

The corpus is 956 million series from three sources that overlap heavily, republish each other,
and publish the same underlying quantity in many presentations. A random train/test split over
series is therefore almost certainly contaminated. Most of what follows is about that.

Throughout: **`X` is the target, `Z` is anything the model can see.**

---

## A. The same quantity in a different dress

The corpus stores one measurement many times over. These are exact deterministic relations, so a
model that sees one side has the other for free.

### A1. Seasonal adjustment pairs

FRED alone:

| `seasonal_adjustment` | series |
|---|---|
| Not Seasonally Adjusted | 750,215 |
| Seasonally Adjusted | 74,429 |
| Seasonally Adjusted Annual Rate | 20,080 |
| Smoothed Seasonally Adjusted | 788 |

Many SA series have an NSA twin under a near-identical id. Eurostat and OECD carry the same thing
as a dimension (`s_adj` with `SCA` / `NSA` / `CA`), so the twins sit inside one dataset and differ
in exactly one dimension code.

*Detect:* group by every dimension except `s_adj`; any group of size > 1 is a twin set.

### A2. Unit transformations of one series

FRED's `units` field is explicit about it — 13,848 series are already
`Percent Change from Year Ago`, and there are `Index`, `Percent Change`, `Change`, and
`Chained 2017 Dollars` variants of the same underlying quantity. FRED's API offers these as
transformations precisely because they are deterministic functions of the level.

*Detect:* the `units` string; then confirm numerically on the overlapping window.

### A3. Currency and denomination variants

Eurostat's `unit` dimension routinely carries `MIO_EUR`, `MIO_NAC`, `MIO_PPS`, `PC_GDP`, `I15`
for the same measurement. These differ by an exchange rate, a PPP, or a GDP series — **all three
of which are themselves in the corpus**, so the mapping is recoverable, not just correlated.

### A4. Per-capita and ratio series

`X per capita` is `X` divided by a population series that is also here. Same for `% of GDP`,
`per employee`, `per household`.

### A5. Rebased indices

`2010 = 100` and `2015 = 100` versions of one index are an affine transform of each other. Note
that rebasing uses the *whole sample*, so the base-year normalisation itself is full-sample
information — see also §I.

---

## B. Derived with a lag — the inflation case

This is the one you raised, and it deserves its own entry because the leak runs *backwards in
time* in a way that is easy to miss.

Year-on-year inflation is `π_t = 100·(P_t / P_{t−12} − 1)`. Two distinct problems:

1. **Forward:** if `P` is visible and `π` is the target, the target is computed, not predicted.
2. **Backward:** `π_t` embeds `P_{t−12}`. So a model given the inflation series has exact
   information about the price level a year earlier. If you hold out `P` for 2025 but leave `π`
   for 2026 in the inputs, you have handed over `P` for 2025.

The same shape applies to every growth rate, contribution-to-growth, and difference series. A
YoY series plus one level anchor reconstructs the entire level path.

*Detect:* within a dataset, take candidate (level, change) pairs from `units` and test the
identity numerically over the common window. Cheap, because it is one dataset at a time.

---

## C. Accounting and aggregation identities

### C1. Geographic hierarchies

Eurostat `geo` contains `EU27_2020`, `EA20` and every member state; the aggregate is the sum (or
weighted mean) of the parts. FRED carries national, state, county and metro versions of the same
statistic — the `County Population Estimates` release alone is 65,971 series that sum to state
and national totals also in the corpus.

### C2. Product, sector and function hierarchies

SDMX marks totals explicitly with `_T` / `TOTAL`. The BIMTS trade dataflows are the extreme case:
HS 2-digit = Σ 4-digit = Σ 6-digit, all three published, and the 2-digit version alone is over
20 GB. NACE, ISIC, COICOP and COFOG all behave this way.

*Detect:* `terrastat hierarchy` does this. It finds series whose dimension key differs in exactly
one position where one side is a total, and `--check` verifies the sum numerically. Measured over
the corpus: **71.5% of datasets carry a reserved total code** (a further 6.4% only a free-text
"Total"/"All" label), 52.8% in two or more dimensions, and
53.6% have a prefix-nested dimension such as NUTS regions. This is the largest single source of
duplicated information in the corpus, and it defeats a random split over series.

### C3. National accounts identities

`GDP = C + I + G + (X − M)` with both sides present. Likewise the sector accounts: every
`DSD_NASEC*` dataflow is built on balancing identities.

### C4. Mirror trade statistics

Exports from A to B and imports by B from A are the same flow seen twice. Worse, BIMTS and BATIS
are *balanced* datasets — the two sides have been explicitly reconciled by OECD, so they are near
identical by construction rather than merely correlated.

### C5. Stock–flow relations

Cumulative flows equal the change in the stock. Financial accounts publish both.

---

## D. Temporal aggregation

The same indicator appears at monthly, quarterly and annual frequency, where the lower frequency
is a sum or mean of the higher. Eurostat encodes this in the dataset code itself: of the 6,636
Eurostat datasets on disk, **88 stems (217 datasets) differ only by a trailing frequency letter** —
`apro_ec_poula` / `apro_ec_poulm`, `apri_pi_outa` / `apri_pi_outq`, `bop_c6_a` / `bop_c6_m`, and
so on. Those pairs are the same collection at two frequencies.

This matters especially for the **aligned cohort**: an annual value for 2026 is almost entirely
determined by the monthly values for January–November 2026. A model asked to forecast the annual
figure while holding monthly data for the same year is not forecasting.

*Detect:* dataset codes differing only in a trailing `m`/`q`/`a`; datasets published at more than
one frequency; then confirm by aggregating and comparing.

---

## E. Vintage and revision leakage

**This is the most serious one for the real-time exercise, and it is invisible from inside our
data.**

We store the *latest* revision of every observation, by design. So a value labelled 2024 in our
files may have been restated in 2026, using information from 2025 and 2026. A "real-time" forecast
of 2025 made from "data up to 2024" is therefore using numbers that did not exist in 2024 and that
partly encode what happened next.

The effect is not small: GDP and employment revisions are routinely larger than the forecast
errors being measured, and benchmark revisions restate decades at once.

*Detect:* not possible from this corpus. It needs vintage archives — ALFRED for FRED, Eurostat's
and OECD's revision databases. Either restrict real-time claims to series where revisions are
known to be small, or state plainly that the exercise is pseudo-real-time.

---

## F. Publication lag

The reference period is not the availability date. Aligning series by reference period — which is
what the aligned cohort does — silently assumes every 2025 observation existed at the end of 2025.
Many did not: annual structural statistics are often published 12 to 24 months late, while CPI
appears within weeks.

*Detect:* `last_updated` minus `end_date` gives a per-series estimate of the lag. It is a
series-level field rather than per observation, so it is an approximation, but a usable one for
excluding the worst offenders.

---

## G. Forecasts and projections sitting inside the data

The anomaly table already counts these: **3,443,528 series end after today** (1,902,901 Eurostat
annual, 1,533,783 OECD annual, and smaller counts elsewhere). They are population projections to
2100, OECD Economic Outlook projections, CBO and OMB budget projections.

Two separate harms:

- **In training:** the model learns to reproduce someone else's model output rather than data.
- **In evaluation:** if a projected value is used as ground truth, you are scoring against a
  forecast.

*Detect:* `end_date > today` (already reported); titles containing "projection", "outlook",
"forecast", "scenario"; SDMX observation-status codes for forecast values.

---

## H. Estimated, imputed and interpolated values

Eurostat flags observations, and the flags are stored. In a sample of the annual and monthly
files: `e` estimated (24,192), `d` definition differs (2,619), `i` (835), `b` break in series (42),
alongside `m` for missing (942,976).

The dangerous subclass is **temporal disaggregation**: monthly series interpolated from quarterly
totals by Chow–Lin or Denton. The January value is then a function of the whole Q1 total, which is
not known until Q1 ends — future information inside a monthly observation.

*Detect:* the `flags` column and `n_flagged`. Method-level detection needs the dataset's
methodological metadata (Eurostat ESMS), which we do not currently download.

---

## I. Two-sided filters — a leak *inside* a single series

This one is often overlooked because it needs no second series.

- **Seasonal adjustment** (X-13, TRAMO-SEATS) extends the series with forecasts before filtering,
  and the adjusted value at the end of the sample is revised as new data arrives. The stored SA
  value at time *t* was therefore computed using data from after *t*.
- **Trend-cycle components** (Henderson filters) are explicitly two-sided.
- **HP filters, band-pass filters, centred moving averages** are two-sided by construction.
- FRED labels 788 series "Smoothed Seasonally Adjusted" outright, and OECD's Composite Leading
  Indicators are filtered and routinely revised.

If you train on seasonally adjusted data and evaluate a real-time exercise, the target already
contains future information. Using NSA data avoids it — which is convenient, since NSA is 89% of
FRED's series.

---

## J. The same series arriving from two sources

FRED republishes other institutions' data. From `origin_agencies`:

| originating body | FRED series |
|---|---|
| OECD | 80,274 |
| World Bank | 16,897 |
| IMF | 14,680 |
| Eurostat | 7,940 |
| Bank for International Settlements | 7,234 |

FRED release 205, "Main Economic Indicators", is 80,216 series and is an OECD product. So a
random split over series can put the OECD copy in train and the FRED copy in test.

*Detect:* `origin_agencies` for the metadata route; hashing the value vector for the certain one.

---

## K. Near-duplicates within one source

Overlapping datasets republish the same series, and dimension cross-products generate series that
are identical because the dimensions that distinguish them are empty in practice.

*Detect:* hash `(frequency, start_date, values)` and look for collisions. Feasible on the aligned
cohort; expensive over all 956 million.

---

## L. Indicators built to lead their own target

Composite leading indicators, PMI-style diffusion indices, consumer confidence, yield spreads.
Not leakage in the strict sense, but if the corpus contains a series *constructed* to anticipate
another series in the corpus, a model can learn the construction rather than the economics, and
the result will not transfer.

---

## What this implies for splitting

Categories A, C, D, J and K all defeat a random split over series. The practical mitigations, in
increasing order of safety:

1. **Split by dataset**, never by series. Kills A1–A3, C2, and most of K.
2. **Split by group** (the theme prefix — `nama`, `apro`, `OECD.SDD.NAD`). Also kills C1 and C3.
3. **Split by country or region.** Kills the geographic identities in C1 outright.
4. **Deduplicate by value hash first**, across sources, to catch J.
5. **Exclude future-dated series** from both sides (G).
6. **Prefer NSA over SA** for real-time work (I).

None of these touch E (vintages), which cannot be fixed by splitting and should simply be
documented as a limitation of a latest-revision corpus.

---

## Cost of detection

| tier | what it catches | cost |
|---|---|---|
| metadata only — `units`, `s_adj`, `_T` codes, dataset-code suffixes, `origin_agencies`, `end_date`, flags | A1, A2, C2, D, G, H, J | seconds; scalar columns only |
| value hashing on the aligned cohort | J, K | minutes |
| identity tests within a dataset (sums, YoY, ratios) | A3, A4, B, C1, C3, C5 | hours; needs `values`, but only within a dataset |
| pairwise correlation across the whole corpus | everything else | infeasible at 956M series; restrict to the cohort |
