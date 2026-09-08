"""Where the corpus is hierarchical: series that are exact sums of other series.

A different kind of property from length or recency. Those describe one series; this describes a
*relation between* series, and it is the one that makes a corpus useful for hierarchical
forecasting and reconciliation — and dangerous for a naive train/test split, since an aggregate
and its parts carry the same information twice.

Two structures produce it here, and they are found in different ways.

**Explicit totals.** SDMX marks an aggregate with a reserved code, usually ``_T``. A dataset whose
``sex`` dimension holds ``{M, F, _T}`` states that ``_T = M + F`` exactly. Measured over the
datasets on disk: **71.6% carry an explicit total in at least one dimension**, 48.6% in two or
more. ``sex`` is the most common (2,445 datasets), then ``age``, then firm size class.

**Prefix-nested classifications.** NUTS regions, NACE activities, COICOP purposes and HS products
all encode depth in the code itself: ``DE`` contains ``DE1`` contains ``DE11``. Nesting is
therefore recoverable from the codes alone, with no external table — **53.6% of datasets have at
least one prefix-nested dimension**, almost always ``geo``.

**The catch, and it is a large one.** An aggregation identity only holds if the measure is
additive, and the most common unit in the corpus is *Percentage* (1,927 datasets), followed by
Number, Thousand persons, Euro and Thousand tonnes. A "total" row in a percentage or index dataset
is a differently-computed figure, not a sum, and treating it as one produces silent nonsense. So
every function here reports additivity beside the structure, and ``check_identity`` verifies the
arithmetic rather than assuming it.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import re
from pathlib import Path

import polars as pl

from terrastat.storage import data_dir

log = logging.getLogger(__name__)

# Reserved codes meaning "all of this dimension". `_T` is the SDMX convention; the others turn up
# in Eurostat and in OECD's older dataflows.
TOTAL_CODES = frozenset({"_T", "TOTAL", "T", "TOT", "ALL", "_Z"})
_TOTAL_LABELS = re.compile(r"^(total|all|all items|all activities|all ages)\b", re.I)

# Units that do not add up. A percentage, an index, a rate or an average over a subgroup is not a
# component of the same figure for the parent group, however the dimension is labelled.
_NON_ADDITIVE = re.compile(
    r"percent|per cent|\brate\b|index|ratio|average|median|\bper\s|per1|per 1|"
    r"purchasing power|standard|score|probability|share of",
    re.I,
)


def is_total(code: str | None, label: str | None = None) -> bool:
    """Whether this code means "everything in this dimension"."""
    if code and code.strip().upper() in TOTAL_CODES:
        return True
    return bool(label and _TOTAL_LABELS.match(label.strip()))


def is_additive(unit_label: str | None) -> bool | None:
    """Whether summing children to make the parent is arithmetically meaningful.

    ``None`` when there is nothing to judge. Deliberately conservative: anything that looks like a
    percentage, index, rate or average is treated as non-additive, because a false positive here
    produces a silently wrong identity rather than a visible error.
    """
    if not unit_label:
        return None
    return not bool(_NON_ADDITIVE.search(unit_label))


def code_levels(codes: list[str]) -> dict[str, int]:
    """Depth of each code, by counting how many of its own proper prefixes are also present.

    ``DE`` -> 0, ``DE1`` -> 1, ``DE11`` -> 2. Linear in the length of each code rather than
    quadratic in the size of the codelist, which matters: some geo dimensions hold thousands of
    NUTS codes and the naive pairwise version does not finish.
    """
    s = {c for c in codes if c and c.upper() not in TOTAL_CODES}
    return {c: sum(1 for i in range(1, len(c)) if c[:i] in s) for c in s}


def _dataset_files(root: Path | str | None = None, sources=None):
    base = (Path(root) if root else data_dir()) / "datasets"
    for src_dir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        if sources and src_dir.name not in sources:
            continue
        yield from sorted(src_dir.glob("*/dataset.json"))


def hierarchy_table(root: Path | str | None = None, sources=None) -> pl.DataFrame:
    """One row per dataset: where its hierarchy is, how deep, and whether it can legitimately sum.

    ``total_dims`` are dimensions holding an explicit aggregate; ``nested_dims`` are those whose
    codes nest by prefix. ``additive`` is the unit test, and a dataset with structure but
    ``additive = false`` is one where the totals exist but are not sums.
    """
    rows = []
    for f in _dataset_files(root, sources):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("skipping %s: %s", f, exc)
            continue
        totals, nested, depth = [], [], 0
        units = []
        for d in m.get("dimensions") or []:
            codes = d.get("codes") or {}
            if not codes:
                continue
            if any(is_total(c, v) for c, v in codes.items()):
                totals.append(d["id"])
            lv = code_levels(list(codes))
            deep = [v for v in lv.values() if v >= 1]
            if len(deep) >= 2:
                nested.append(d["id"])
                depth = max(depth, max(deep))
            if str(d["id"]).lower() in ("unit", "unit_measure"):
                units = list(codes.values())
        add = [is_additive(u) for u in units]
        rows.append({
            "source": m.get("source") or f.parent.parent.name,
            "dataset_id": m.get("dataset_id") or f.parent.name,
            "title": (m.get("title") or "")[:120],
            "n_series": m.get("n_series"),
            "total_dims": ",".join(totals),
            "n_total_dims": len(totals),
            "nested_dims": ",".join(nested),
            "max_depth": depth,
            "n_units": len(units),
            # a dataset mixing additive and non-additive units is only partly summable
            "additive": None if not add else (all(a for a in add if a is not None) or None if all(a is None for a in add) else all(add)),
            "units": " | ".join(str(u)[:30] for u in units[:4]),
        })
    return pl.DataFrame(rows).sort("n_series", descending=True)


def _pick_parent(totals: list[tuple[str, dict]]):
    """The one aggregate among the total-looking codes, or nothing if it cannot be told.

    A reserved SDMX code (``_T``) is unambiguous and wins outright. Several free-text labels
    beginning "Total" are not a hierarchy this function can resolve — they are usually different
    aggregates, sometimes different measures entirely — so it declines rather than guessing.
    """
    if not totals:
        return None
    reserved = [r for c, r in totals if c and c.strip().upper() in TOTAL_CODES]
    if len(reserved) == 1:
        return reserved[0]
    if len(totals) == 1:
        return totals[0][1]
    return None


def aggregation_sets(dataset_id: str, dim: str, root=None, max_groups: int | None = 200) -> pl.DataFrame:
    """Parent/child groups inside one dataset, along one dimension.

    Series are grouped by every dimension *except* ``dim``; within a group, the series whose
    ``dim`` code is a total is the parent and the rest are its children. That is the whole
    definition — it needs no external hierarchy table, only the SDMX convention.
    """
    from terrastat.quality import scan

    lf = scan(root, columns=["series_uid", "dataset_id", "frequency", "dimensions", "units",
                             "start_date", "end_date", "n_obs"])
    df = lf.filter(pl.col("dataset_id") == dataset_id).collect()
    if not df.height:
        return pl.DataFrame()
    groups: dict[tuple, dict] = collections.defaultdict(lambda: {"totals": [], "children": []})
    for r in df.iter_rows(named=True):
        dims = r["dimensions"] or []
        key, this = [], None
        for d in dims:
            if d["id"] == dim:
                this = d
            else:
                key.append((d["id"], d["code"]))
        if this is None:
            continue
        g = groups[(r["frequency"], tuple(key))]
        if is_total(this["code"], this["label"]):
            g["totals"].append((this["code"], r))
        else:
            g["children"].append(r)
    out, ambiguous = [], 0
    for (freq, key), g in groups.items():
        parent = _pick_parent(g["totals"])
        if parent is None:
            # Either no total at all, or several and no way to tell which is the aggregate. A
            # dimension can carry nested totals -- one Eurostat price dimension offers "Total crop
            # including fruit", "Total crop excluding fruit", "Total animal" and "Total
            # agricultural goods" -- and an earlier version silently kept whichever came last and
            # dropped the others from the children too, which quietly produced a wrong parent over
            # an incomplete set. 8.8% of total-carrying dimensions look like this.
            ambiguous += len(g["totals"]) > 1
            continue
        g = {"parent": parent, "children": g["children"]}
        if len(g["children"]) < 2:
            continue
        out.append({
            "frequency": freq,
            "parent_uid": g["parent"]["series_uid"],
            "n_children": len(g["children"]),
            "units": g["parent"]["units"],
            "additive": is_additive(g["parent"]["units"]),
            "child_uids": [c["series_uid"] for c in g["children"]],
            "group": " ".join(f"{k}={v}" for k, v in key)[:120],
        })
        if max_groups and len(out) >= max_groups:
            break
    return pl.DataFrame(out) if out else pl.DataFrame()


def check_identity(sets: pl.DataFrame, root=None, tol: float = 0.01, max_groups: int = 50) -> pl.DataFrame:
    """Does the parent actually equal the sum of its children?

    Structure says an identity should hold; only arithmetic says it does. A group can fail
    legitimately — the children may not exhaust the parent, the unit may not be additive, the
    aggregate may be computed rather than summed — so this reports the discrepancy rather than
    passing judgement.
    """
    from terrastat.quality import scan

    if not sets.height:
        return pl.DataFrame()
    sets = sets.head(max_groups)
    wanted = set(sets.get_column("parent_uid").to_list())
    for lst in sets.get_column("child_uids").to_list():
        wanted.update(lst)
    data = (
        scan(root, columns=["series_uid", "dates", "values"])
        .filter(pl.col("series_uid").is_in(list(wanted)))
        .collect()
    )
    by = {r["series_uid"]: dict(zip(r["dates"], r["values"])) for r in data.iter_rows(named=True)}
    rows = []
    for r in sets.iter_rows(named=True):
        p = by.get(r["parent_uid"]) or {}
        kids = [by.get(u) or {} for u in r["child_uids"]]
        dates = [d for d, v in p.items() if v is not None and all(k.get(d) is not None for k in kids)]
        if not dates:
            continue
        diffs = []
        for d in dates:
            tot = sum(k[d] for k in kids)
            denom = abs(p[d]) or 1.0
            diffs.append(abs(p[d] - tot) / denom)
        worst = max(diffs)
        rows.append({
            "parent_uid": r["parent_uid"], "n_children": r["n_children"],
            "units": r["units"], "additive": r["additive"],
            "n_dates": len(dates),
            "max_rel_error": round(worst, 6),
            "holds": bool(worst <= tol),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="terrastat hierarchy", description=__doc__.splitlines()[0])
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--dataset", default=None, help="inspect one dataset's aggregation groups")
    p.add_argument("--dim", default=None, help="with --dataset: the dimension holding the total")
    p.add_argument("--check", action="store_true", help="with --dataset: verify parent == sum(children)")
    p.add_argument("--out", default=None, help="write the dataset-level table here")
    p.add_argument("--root", default=None)
    args = p.parse_args(argv)

    if args.dataset:
        if not args.dim:
            h = hierarchy_table(args.root, args.sources).filter(pl.col("dataset_id") == args.dataset)
            print(h.select("dataset_id", "total_dims", "nested_dims", "max_depth", "additive", "units"))
            return 0
        sets = aggregation_sets(args.dataset, args.dim, args.root)
        print(f"{sets.height} aggregation groups on {args.dim!r}")
        if sets.height:
            print(sets.select("frequency", "n_children", "units", "additive", "group").head(10))
        if args.check and sets.height:
            res = check_identity(sets, args.root)
            if res.height:
                ok = res.get_column("holds").sum()
                print(f"\nidentity parent == sum(children): holds for {ok} of {res.height} groups checked")
                print(res.head(10))
        return 0

    t = hierarchy_table(args.root, args.sources)
    n = t.height
    print(f"{n:,} datasets\n")
    print(f"  with an explicit total   {t.filter(pl.col('n_total_dims') > 0).height:>6,}  "
          f"({t.filter(pl.col('n_total_dims') > 0).height / n:.1%})")
    print(f"  totals in 2+ dimensions  {t.filter(pl.col('n_total_dims') >= 2).height:>6,}")
    print(f"  prefix-nested dimension  {t.filter(pl.col('max_depth') > 0).height:>6,}  "
          f"({t.filter(pl.col('max_depth') > 0).height / n:.1%})")
    print(f"  ... and additive units   {t.filter((pl.col('n_total_dims') > 0) & pl.col('additive')).height:>6,}")
    with pl.Config(tbl_rows=15, tbl_cols=10, ascii_tables=True, tbl_formatting="ASCII_MARKDOWN",
                   tbl_hide_dataframe_shape=True, fmt_str_lengths=30):
        print("\n== largest datasets with a summable hierarchy ==")
        print(t.filter((pl.col("n_total_dims") > 0) & pl.col("additive"))
               .select("source", "dataset_id", "n_series", "total_dims", "max_depth", "units").head(12))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        t.write_parquet(args.out)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
