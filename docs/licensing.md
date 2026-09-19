# Licences and citation for research use

terrastat's code is licensed under [Apache-2.0](../LICENSE). The economic data keeps the terms
of its providers. A software licence, a citation, and permission to redistribute data serve
different purposes; keep all three in mind when preparing a paper or replication package.

## Find the terms for your series

Each series stores `license_id`, `license_url`, `attribution` and `license_notes`, together with
its source and retrieval metadata. To inspect a small selection from your existing snapshot:

```python
from terrastat import dataset

terms = dataset.load("starter", columns=[
    "source", "dataset_id", "license_id", "license_url", "attribution", "license_notes"
]).head(10).collect()
terms
```

The `license_id` is terrastat's classification of source metadata. Follow `license_url` and
check any dataset-specific notes when deciding whether a particular use is permitted.
Academic affiliation does not by itself remove restrictions on data use or sharing.

## Provider policies

These notes were checked against the linked provider pages on 2026-09-17. The provider's
current terms and the individual dataset notices take precedence over this summary.

| Provider | Practical starting point |
|---|---|
| Eurostat | Reuse generally requires source acknowledgement. Dataset-specific exceptions apply, including restrictions on commercial reuse of some country and trade data. Clearly identify modifications. [Official notice](https://ec.europa.eu/eurostat/web/main/help/copyright-notice) |
| OECD | Use the citation supplied with the dataset. Check metadata for third-party ownership and additional restrictions before incorporating data into your work. [Official terms, Data section](https://www.oecd.org/en/about/terms-conditions.html) |
| FRED | Check both the originating provider's rights and FRED's service/API terms. The terms restrict machine-learning use and API-based storage/archiving; a public-domain label does not override those terms. [Official terms](https://fred.stlouisfed.org/legal/) |

The CPU forecasting example uses Eurostat series. FRED-origin labels and
`origin_us_federal` are useful metadata, but the latter is a heuristic about the originating
agency, not a determination of permission. For a restricted use, obtain the relevant permission
or assess data obtained directly from the originating provider under that provider's terms.

## Cite both the software and the data

Use [CITATION.cff](../CITATION.cff) for the software citation; GitHub also exposes it through
**Cite this repository**. Cite the statistical agencies and datasets separately, using their
recorded `attribution` and dataset identifiers, DOI or source URL, and access date.

For reproducibility, retain the snapshot's `manifest.json`, its attribution file, and the
settings used to select and transform the data. In your methods section, report the dataset
version or retrieval date, units and seasonal adjustment, sample period, treatment of missing
values, and evaluation cutoffs. Stored observations use the latest revisions available when
collected; they are not necessarily the values available at each historical forecast date.

## Share a replication package

When creating an export, `--public-only` applies the project's initial licence filter. It keeps
Eurostat and OECD series, plus FRED series with the selected public-domain status and a US federal
origin flag. **It is a selection rule, not a certificate of permission.** It does not resolve
third-party exceptions or restrictions imposed by service terms.

Keep `ATTRIBUTIONS.md`, the dataset card (`README.md`) and `manifest.json` with any snapshot
you are permitted to share. Check the intended use against the relevant terms; if redistribution
is not permitted, a replication package can instead document acquisition steps and series IDs.
Use a new snapshot name for a new release so earlier experiments remain reproducible.

See the [export recipe](../MANUAL.md#6b-a-snapshot-for-training-export) for commands and
[NOTICE](../NOTICE) for the project's attribution and licensing notice.
