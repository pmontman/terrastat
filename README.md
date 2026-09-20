# terrastat

Economic time series from **Eurostat, OECD and FRED**, ready to read in Python, R or any tool
that supports Parquet. terrastat downloads the data, keeps its dates and source metadata, and
helps you assemble a reproducible dataset for research.

**Already have a snapshot?** You can open it directly and run the CPU forecasting tutorial.
**Starting without data?** The example below downloads one Eurostat dataset without an API key.

The software is Apache-2.0. **The data has separate licences and attribution requirements** from
its providers; see [Licences](#licences) before sharing a dataset or using it to train a model.

## 1. Install

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). These commands
create a Python 3.12 environment and install terrastat with notebook support:

```bash
git clone https://github.com/pmontman/terrastat.git
cd terrastat
uv venv --python 3.12
uv pip install -e ".[notebook]"
```

If you already cloned the repository, work from that folder. If its environment is already
set up, keep it and run just the install command. The commands below use `uv run --no-sync`,
so you do not need to activate the environment. They work in PowerShell and macOS/Linux shells.

## 2. Open or get data

### I already have a snapshot

A snapshot is a folder containing Parquet files and a `manifest.json` describing its contents.
Keep the whole folder together. For the examples, place it at `data/snapshot/starter/`.
If it is already there, leave it as it is.

Start Jupyter:

```bash
uv run --no-sync jupyter lab
```

In a Python notebook opened from the repository root:

```python
from terrastat import dataset

quarterly = dataset.load("starter", frequencies=["Q"])
sample = quarterly.head(5).collect()
sample.select("series_uid", "title", "units", "n_obs")
```

`starter` refers to your local snapshot; it is not an automatic download. You can also pass
an existing snapshot directory to `dataset.load`. Reading it does not change its files.

### I have a saved normalized release

You can start from normalized data shared online or from archives downloaded manually.
The [saved-data deployment walkthrough](docs/deployment.md) covers both routes: restore the
observations and reference metadata, rebuild the series view offline, then read it in Python.
No official download URL is configured yet; use the URLs or files supplied with your release.

### I need to download data from the providers

Start with quarterly unemployment from Eurostat:

```bash
uv run --no-sync terrastat fetch eurostat --ids une_rt_q --with-series
uv run --no-sync terrastat peek eurostat une_rt_q
```

The first command downloads the dataset and prepares one row per time series. The second
shows its metadata and a few observations. Data is saved under `data/`; download time depends
on the provider. This example needs no API key.

Open Jupyter with the command above, then read the result:

```python
from terrastat.config import data_dir
from terrastat import dataset

path = data_dir() / "series/eurostat/freq=Q/une_rt_q.parquet"
sample = dataset.load(path).head(5).collect()
sample.select("series_uid", "title", "units", "n_obs")
```

Each row contains one series. Its `dates` and `values` are matching lists, and `n_obs` counts
observed values. To inspect its observations and attribution:

```python
row = sample.row(0, named=True)
print(row["attribution"])
print(row["license_url"])
list(zip(row["dates"], row["values"]))[:8]
```

Keep dates with values: missing periods may be absent from these lists. The modeling tutorial
reconstructs the calendar and retains those gaps as missing values.

## 3. Run a forecasting experiment

The [quarterly forecasting notebook](notebooks/quarterly_forecasting.ipynb) takes you through
data inspection, plots, training a small shared residual network on CPU, and comparison with
ETS and seasonal naive. It includes separate training, validation and test periods.

```bash
uv pip install -e ".[notebook,forecast]"
uv run --no-sync jupyter lab notebooks/quarterly_forecasting.ipynb
```

Run its cells from top to bottom. The default uses the existing `starter` snapshot.
Without that snapshot, set `INPUT_MODE = "sources"` in the settings cell: the notebook will
obtain the four Eurostat datasets it uses. No GPU or FRED key is needed.

The notebook saves model weights, predictions, settings and source checksums under
`reports/quarterly_forecasting/`. See the [experiment notes](docs/forecasting.md) for the
holdouts, missing-data handling and interpretation of the scores.

## Licences

**Code and data must be cited and licensed separately.** The code is covered by
[Apache-2.0](LICENSE); that licence does not grant rights to the downloaded data.

For a paper or replication package, keep the dataset identifiers, retrieval dates, provider
links and attribution text. These travel with each series. Snapshots also include
`ATTRIBUTIONS.md` when produced by the current exporter. Cite the software using
[CITATION.cff](CITATION.cff), and cite the original data providers as well.

Eurostat and OECD have reuse policies with attribution requirements and dataset-specific
exceptions. FRED has additional service terms, including restrictions on machine-learning
use. The [licensing and citation guide](docs/licensing.md) links the official terms and
explains the metadata. Academic use is not a blanket exemption.

When exporting for others, use `--public-only` as an initial licence filter and retain the
attribution files. It **does not certify permission** for every dataset or intended use.
The [sharing recipe](MANUAL.md#6b-a-snapshot-for-training-export) explains the export step.

## What to read next

| I want to… | Start here |
|---|---|
| Find datasets, download more, or resume a job | [User guide](MANUAL.md) |
| Understand Parquet and explore one dataset | [Parquet notebook](notebooks/parquet_basics.ipynb) |
| Check source terms or prepare citations | [Licensing and citation](docs/licensing.md) |
| Understand columns, flags and file layout | [Schema](docs/schema.md) |
| Design a forecasting evaluation | [Experiment notes](docs/forecasting.md) and [leakage guide](docs/leakage.md) |
| Know whether a rerun gets new data | [Refresh behavior](docs/refresh.md) |
| Store data compactly, transfer it, or restore an archive | [Storage and deployment](docs/storage.md) |
| Start from normalized data shared online or downloaded manually | [Saved-data deployment](docs/deployment.md) |
| Look up commands, source behavior or storage details | [Technical reference](docs/reference.md) |
| See the collected corpus statistics | [Dated corpus report](docs/corpus.md) |
| Contribute code or review the design | [Contributing](CONTRIBUTING.md) and [design review](docs/design-review.md) |

You can stop a download with Ctrl+C and rerun the same command to resume eligible unfinished
work. Completed datasets are not automatically refreshed with new observations or revisions.

terrastat is an independent research tool, not an official product of the data providers.
