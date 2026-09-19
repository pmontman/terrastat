# Quarterly forecasting tutorial

[Open the notebook](../notebooks/quarterly_forecasting.ipynb) to work through a complete
forecasting experiment on a CPU. You will inspect the data, train a small model shared across
series, and compare its forecasts with two standard benchmarks.

From the repository root:

```bash
uv pip install -e ".[notebook,forecast]"
uv run --no-sync jupyter lab notebooks/quarterly_forecasting.ipynb
```

Run the cells in order. If you have `data/snapshot/starter/`, the default settings use it
directly. Otherwise, set `INPUT_MODE = "sources"` in the settings cell to obtain the four
Eurostat datasets. To use another existing snapshot, change `SNAPSHOT` to its name or path.
The plots and score tables appear in the notebook; saved runs go under
`reports/quarterly_forecasting/`.

New to the package? The [quick start](../README.md) covers installation and your first data read.
For a paper or replication package, also read [licensing and citation](licensing.md).
The remaining sections explain the experiment in enough detail to adapt or report it.

The default input is the existing `data/snapshot/starter/` directory. It reads the current
float32/list schema without migration or rewriting any snapshot files. Set `SNAPSHOT` to another
existing name/path. Snapshot mode does not fetch data or fall back to source files; a missing
snapshot or absent quarterly subset is an explicit error. `INPUT_MODE="sources"` is the optional
acquisition route. Re-running later uses the same snapshot; see [refresh.md](refresh.md).

## What the experiment measures

One residual MLP learns shared weights across a small subset of Eurostat GDP (`namq_10_gdp`),
unemployment (`une_rt_q`), industrial production (`sts_inpr_q`) and labour costs (`lc_lci_r2_q`).
It uses numerical history and observation masks, without text embeddings or external covariates.
The default caps selection at 64 series per dataset and preserves the snapshot's geographic
selection (`GEOS=None`). Set a list of country codes to restrict it explicitly. Candidate
sampling uses series identifiers; eligibility uses training-period availability only.

| Setting | Default |
|---|---|
| Context / horizon | 24 quarters / 4 quarters |
| Training targets | Through 2022Q4, stride four quarters |
| Validation targets | 2023Q1–Q4; select checkpoint by masked normalized MAE |
| Test targets | 2024Q1–Q4 and 2025Q1–Q4 |
| Model | Width 64, two residual blocks, seasonal skip, four direct outputs |
| Optimization | AdamW, 25 epochs, batch 256, seed 7, two CPU threads |
| Benchmarks | Seasonal naive; additive damped Holt-Winters with period four |

At each test origin, all methods receive the same 24-quarter context. The network remains frozen;
ETS refits using the available context. The 2025 forecast can use observed 2024 history. Neither
the network's training targets nor checkpoint selection use test targets. All methods score the
same observed target entries; absent targets are counted in the coverage report.

This is retrospective forecasting of known series using latest-vintage data. It demonstrates
the shared training pattern of a miniature foundation-style model, not zero-shot transfer or
historically available information. Series variants and aggregates remain dependent. See
[leakage.md](leakage.md) before designing a publication benchmark.

## Missing-data contract

The on-disk series layer stays sparse. An unflagged missing quarter may have no row, but the
remaining observations keep their dates. Flagged null observations remain explicit. Thus
`n_points - n_obs` measures stored nulls, not all calendar gaps. Sparse storage does not recover
an unknown leading/trailing extent or a series never stored at all.

`terrastat.calendar.regularize(dates, values, frequency, max_periods=10_000)` reconstructs only
the interval between the earliest and latest stored date. It returns `dates`, `values` (NaN at
gaps), and `observed`. The source lists and files are not modified.

- Period counts are validated **before allocation**. The notebook uses a stricter 2,000-quarter
  source limit and a fixed 104-quarter experiment grid, with at most 2,048 input candidates.
- Duplicate periods, null dates, the year-9999 sentinel, infinite values and weekly misalignment
  raise descriptive errors. Sorting dates is allowed; duplicate values are never silently averaged.
- M/Q/S/A labels normalize to calendar period starts. D/W/BW are supported; B means weekdays,
  without an exchange holiday calendar. H/I/P/A3/OTHER need an explicit policy and are rejected.
- Missing model features receive zero **after** context-only standardization, with a separate
  mask. This tensor encoding does not change the original series. Targets are never imputed.
- ETS and seasonal naive forward-fill context copies and trim only leading missing values.
  Failed or nonconverged ETS fits fall back to seasonal naive and are reported individually.

`dataset.windows()` now reconstructs calendars and returns `X_mask`, `y_mask` and `target_dates`
as well as the existing arrays. Its default retains missing values. For compatibility, the
`drop_missing=True` argument remains accepted, but now skips entire incomplete windows; it
never removes individual periods. Undefined MASE scales are NaN and no longer cause windows
to disappear. This helper alone is not a cross-series chronological split: use explicit cutoffs.
The older `dataset.mase` helper expects complete arrays; masked scoring in this tutorial uses
`forecasting.score_forecasts`, which also reports observed-target counts.

## Reading snapshots correctly

`dataset.load("starter")` opens an existing local snapshot; it does not download one.
Snapshots contain `shard-*.parquet` with the series and `index.parquet` with lookup metadata.
Only shards belong in a series scan. Mixing the index and shards produces schema mismatches
or missing-column errors. The loader now excludes the index and applies frequency/length filters
before projecting requested output columns.

The repository's named-download registry is currently empty, but local `starter` snapshots are
supported. The default tutorial opens your existing snapshot; the optional source mode uses the
source adapters directly rather than assuming a published download exists.

## Scores and reproducibility

Report mean and median MASE, sMAPE, per-dataset results, and coverage. MASE's denominator is the
mean absolute difference of observed pairs four quarters apart in the original context; it never
uses the target. Zero denominators produce undefined MASE, counted separately. MAE remains useful
within units, but should not be averaged across GDP amounts and unemployment percentages.
sMAPE treats a pair of zeros as zero error; missing targets are excluded.

The verified run on the existing `starter` snapshot on 2026-09-17 selected 148 series from
160 candidates. It scored 134 series in 2024 and 133 in 2025, with 534 and 530 observed targets
respectively out of 592 possible targets per year. Across the two origins, 134 distinct series
were scored and 133 contributed defined MASE values. With checkpoint 21 selected by validation:

| Method | Mean MASE | Median MASE | Mean sMAPE (%) |
|---|---:|---:|---:|
| ETS with seasonal-naive fallback | 0.626 | 0.517 | 31.97 |
| Shared residual MLP | 0.633 | 0.515 | 34.67 |
| Seasonal naive | 0.788 | 0.679 | 33.62 |

Two of 267 ETS fits used fallback. CPU training took about 0.8 seconds and ETS fitting about
21 seconds on the review machine, excluding imports, reading and plotting. These are example
measurements, not timing guarantees or corpus-wide accuracy claims. The network's median MASE
was slightly lower than ETS; ETS had lower mean MASE and sMAPE. The network did not beat seasonal
naive on sMAPE. This snapshot cohort differs from the earlier source-file example, so those runs
must not be compared as a model change. All 19 files across the three existing local snapshots
were checked to remain byte-for-byte unchanged during compatibility validation.

Each run writes a fresh directory under `reports/quarterly_forecasting/` with the cohort, source
checksums, versions, settings, model weights, predictions, target masks, scores and learning curve.
These artifacts are ignored by Git. Committed notebook outputs describe a particular local run;
they can change when source revisions, dependencies or platforms change.

Generate the notebook source with `python notebooks/_build_quarterly_forecasting.py`. This clears
outputs; execute it again before committing a refreshed real-data example. CI executes the same
notebook with `TERRASTAT_TUTORIAL_SMOKE=1`, which writes explicitly labeled synthetic data into
temporary float32 snapshot shards with a separate index, loads them through the same reader,
and trains for two epochs. Smoke mode never replaces a missing real snapshot silently.

References: [time-series cross-validation](https://otexts.com/fpp3/tscv.html),
[forecast accuracy](https://otexts.com/fpp3/accuracy.html),
[statsmodels Holt-Winters](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html).
