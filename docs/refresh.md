# What happens when you run terrastat again?

Current behavior, checked against the implementation on 2026-09-17: terrastat resumes ingestion,
but it does not automatically refresh completed datasets after they age. There is no implemented
delta-update or revision-merge protocol.

| Operation | Existing data behavior |
|---|---|
| Quarterly notebook, default snapshot mode | Reads the same local snapshot and retrains; no network or snapshot writes |
| Quarterly notebook, source mode | Reuses existing quarterly files; fetches only if a requested file is absent |
| `terrastat run` / normal `fetch` | Skips datasets marked `done`; resumes eligible unfinished work |
| `terrastat catalog SOURCE --refresh` | Refreshes the catalogue; does not refresh existing observations |
| `terrastat fetch SOURCE --force` | Bypasses completed-state skipping, but source adapters can reuse old raw files |
| `terrastat series SOURCE --force` | Rebuilds derived series from local observations; no download |
| `terrastat export NAME` | Repackages local series; does not obtain new provider data |

Elapsed time alone changes none of these rules. A completed collection run today does not become
an incremental update three months later. Catalogue caches also persist unless refreshed explicitly.
Some auxiliary metadata/tag work may still run; that is not a refresh of completed observations.

## Why `--force` does not guarantee freshness

The pipeline's force flag changes which datasets are selected for processing. It is not passed
as a cache-invalidation instruction to the source adapters:

- Eurostat's `fetch_raw` reuses an existing `data.tsv.gz`.
- OECD reuses a completed raw file or completed split manifest.
- FRED reuses the page list when `_pages.json` says the release is complete.

Thus a forced run may simply parse the same raw payload again. Refreshing a catalogue does not
invalidate those payloads. Do not treat a new processing timestamp as proof of newly downloaded data.

## Intended refresh design (not implemented yet)

A useful refresh command should first check which datasets changed, then explicitly refresh
their raw payloads and rebuild derived series. Download only changed datasets where change
metadata is reliable; provider-specific deltas require separate support. Append-only updates
are insufficient because historical economic observations can be revised or removed.

Record provider version/retrieval metadata, detect changed historical observations, and publish
a new versioned snapshot after verification. Keep earlier snapshots intact for reproducible
benchmarks. Until this exists, a deliberately separate data root can isolate a fresh full
acquisition from the current collection; it does not provide incremental updates.
