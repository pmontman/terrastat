# Working on terrastat

From the repository root, create a Python 3.11+ environment and install the editable package:

```powershell
uv venv --python 3.12
uv pip install -e ".[dev,notebook,forecast]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m terrastat corpus --check
```

On macOS/Linux use `.venv/bin/python`. Package installation is required before testing the
`src/` layout; a `ModuleNotFoundError: terrastat` is not a parser failure. Core development can
install only `.[dev]`; notebook execution additionally needs `.[notebook,forecast]`.

## Keep changes close to the contract

- Source-specific payload handling belongs in `src/terrastat/sources/`. Add small fixture-based
  regressions for real parser failures; ordinary tests should not call live providers.
- Preserve original dates, flags and provenance. Sparse storage is intentional. Modeling code
  must reconstruct calendars or use dates explicitly; never compress away missing periods.
- Apply time cutoffs before fitting transforms or selecting training windows. Test this by
  changing future values and checking that training arrays and normalization stay identical.
- Keep optional training libraries out of core imports. Avoid writing data, model weights or
  derived experiment outputs into tracked source directories.

## Notebooks and generated documentation

Edit `notebooks/_build_quarterly_forecasting.py`, then regenerate the notebook. Execute it against
real data to refresh the committed example outputs:

```powershell
.venv\Scripts\python notebooks/_build_quarterly_forecasting.py
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks/quarterly_forecasting.ipynb
```

For an offline execution check, set `TERRASTAT_TUTORIAL_SMOKE=1` and write the executed notebook
under `reports/` so synthetic outputs do not overwrite the real-data example. The CI workflow
shows the exact command. Inspect plots for clipping and inspect the coverage/fallback tables;
successful execution alone does not establish that the model comparison is meaningful.

`docs/index.html` and `docs/corpus.md` come from renderers and `docs/corpus_tables.json`.
After changing a renderer, use `python -m terrastat corpus --render`, then `--check`.
Do not manually edit the generated figures or present old measurements as a fresh crawl.

For the architecture findings and next work, see [docs/design-review.md](docs/design-review.md).
