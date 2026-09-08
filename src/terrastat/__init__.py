"""terrastat: polite, resumable gathering of public economic time series into a tidy Parquet database.

Three layers, each usable on its own while the next one is still being built:

1. ``data/raw/``       exact payloads as served by the source (compressed), for reproducibility.
2. ``data/datasets/``  one folder per dataset (a FRED release, a Eurostat dataset, an OECD dataflow)
                       holding a compact long table ``observations.parquet`` and a ``dataset.json``
                       with every piece of metadata the source gives us (codelists, licence, links).
3. ``data/series/``    one row per time series with the metadata already expanded into text,
                       partitioned by source and frequency. This is the model-ready view.
"""

__version__ = "0.1.0"
