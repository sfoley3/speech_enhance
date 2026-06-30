#!/usr/bin/env python3
"""
P.835 results analyser
======================
Loads the responses produced by the survey, joins them to mapping.csv, and reports:

  1. A dsp-vs-raw table across all four conditions (orig + 3 models) with
     mean +/- std (and N, 95% CI) for each P.835 dimension SIG / BAK / OVRL.
  2. Inter-rater agreement (Krippendorff's alpha, ordinal) per dimension,
     plus mean per-stimulus SD as a simple dispersion check.
  3. Quality-control flags:
       - gold_high items rated too LOW and gold_low items rated too HIGH,
       - raters who fail the gold checks or the final attention question,
       - clean/noisy anchor means (sanity: clean should top, noisy should bottom).

Inputs
------
--responses   Either (a) a CSV exported from the Google Sheet collector that has a
              `responsesJSON` column, or (b) a directory / glob of the per-response
              JSON files the survey downloads (response_*.json).
--mapping     mapping.csv produced by make_p835_stimuli.py (the private key).
--out         Output directory for the CSV tables (default: ./analysis_out).
--keep-flagged  Include flagged raters in the main table (default: exclude them).

Usage
-----
  python analyze_results.py --responses sheet_export.csv --mapping mapping.csv
  python analyze_results.py --responses ./responses/ --mapping mapping.csv

Requires: pandas, numpy.
"""

import os
import re
import csv
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd

DIMS = ["SIG", "BAK", "OVRL"]
CONDITION_ORDER = ["orig", "META_denoiser", "NVIDIA_REUSE", "PASE"]
BASE_ORDER = ["raw_clean", "dsp"]


# --------------------------------------------------------------------------
# Loading responses (two accepted formats)
# --------------------------------------------------------------------------
def _rows_from_response_obj(obj):
    """One survey submission (the `results` object) -> list of rating rows."""
    rater = obj.get("completionCode") or obj.get("ParticipantInfo", {}).get("completionCode", "")
    info = obj.get("ParticipantInfo", {}) or {}
    final_check = info.get("finalCheck", "")
    block = info.get("block", "")
    rows = []
    for k, v in obj.items():
        if k in ("ParticipantInfo", "completionCode"):
            continue
        if not isinstance(v, dict):
            continue
        rows.append({
            "rater": rater, "block": block, "finalCheck": final_check,
            "id": k,
            "SIG": v.get("SIG"), "BAK": v.get("BAK"), "OVRL": v.get("OVRL"),
        })
    return rows


def load_responses(path):
    rows = []
    if os.path.isdir(path) or any(ch in path for ch in "*?["):
        files = sorted(glob.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) \
            else sorted(glob.glob(path))
        for f in files:
            with open(f) as fh:
                rows.extend(_rows_from_response_obj(json.load(fh)))
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if "responsesJSON" in df.columns:                 # Google Sheet export
            for _, r in df.iterrows():
                try:
                    obj = json.loads(r["responsesJSON"])
                except (TypeError, ValueError):
                    continue
                rows.extend(_rows_from_response_obj(obj))
        else:                                             # already-long CSV
            need = {"rater", "id", "SIG", "BAK", "OVRL"}
            if not need.issubset(df.columns):
                sys.exit(f"[ERROR] CSV lacks responsesJSON and {need} columns.")
            return df
    else:
        sys.exit(f"[ERROR] Unrecognised --responses path: {path}")
    if not rows:
        sys.exit("[ERROR] No ratings parsed from --responses.")
    out = pd.DataFrame(rows)
    for d in DIMS:
        out[d] = pd.to_numeric(out[d], errors="coerce")
    return out


# --------------------------------------------------------------------------
# Krippendorff's alpha (ordinal / interval / nominal), handles missing data
# --------------------------------------------------------------------------
def krippendorff_alpha(units, level="ordinal"):
    """
    units: list of lists; each inner list = the numeric ratings a single item
           received from however many raters (missing values already dropped).
    Returns alpha (float) or nan if undefined.
    """
    units = [[float(x) for x in u if x is not None and not (isinstance(x, float) and np.isnan(x))]
             for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")

    values = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    if V < 2:
        return float("nan")           # everyone agreed perfectly -> alpha undefined/1

    o = np.zeros((V, V))
    for u in units:
        m = len(u)
        w = 1.0 / (m - 1)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += w
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")

    # squared difference metric δ²(c,k)
    delta2 = np.zeros((V, V))
    for c in range(V):
        for k in range(V):
            if level == "nominal":
                delta2[c, k] = 0.0 if c == k else 1.0
            elif level == "interval":
                delta2[c, k] = (values[c] - values[k]) ** 2
            else:  # ordinal
                lo, hi = (c, k) if c <= k else (k, c)
                s = n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2.0
                delta2[c, k] = s ** 2

    Do = (o * delta2).sum() / n
    De = (np.outer(n_c, n_c) * delta2).sum() / (n * (n - 1))
    if De == 0:
        return float("nan")
    return 1.0 - Do / De


def alpha_for_dimension(df, dim):
    """Build per-stimulus rating lists and compute ordinal alpha for one dimension."""
    units = [g[dim].dropna().tolist() for _, g in df.groupby("id")]
    return krippendorff_alpha(units, level="ordinal")


# --------------------------------------------------------------------------
# QC: gold items + attention check
# --------------------------------------------------------------------------
def _expectation_fail(expected, value):
    """expected like '>=4' or '<=2'; return True if `value` violates it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    m = re.match(r"\s*(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)", str(expected))
    if not m:
        return False
    op, thr = m.group(1), float(m.group(2))
    if op == ">=": return not (value >= thr)
    if op == "<=": return not (value <= thr)
    if op == ">":  return not (value > thr)
    if op == "<":  return not (value < thr)
    if op == "=":  return not (value == thr)
    return False


def gold_checks(merged):
    """Per-rating pass/fail on gold items (uses OVRL vs expected_overall)."""
    gold = merged[merged["role"].isin(["gold_high", "gold_low"])].copy()
    if gold.empty:
        return gold, pd.DataFrame()
    gold["fail"] = gold.apply(
        lambda r: _expectation_fail(r.get("expected_overall", ""), r["OVRL"]), axis=1)
    per_rater = (gold.groupby("rater")
                 .agg(gold_items=("fail", "size"), gold_fails=("fail", "sum"))
                 .reset_index())
    per_rater["gold_fail_rate"] = per_rater["gold_fails"] / per_rater["gold_items"]
    return gold, per_rater


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def ci95(std, n):
    return 1.96 * std / np.sqrt(n) if n and n > 0 else np.nan


def condition_base_table(exp):
    """Long table: condition x base x dim -> mean, std, n, ci95."""
    recs = []
    for (cond, base), g in exp.groupby(["condition", "base"]):
        for d in DIMS:
            vals = g[d].dropna()
            recs.append({
                "condition": cond, "base": base, "dim": d,
                "mean": round(vals.mean(), 3) if len(vals) else np.nan,
                "std": round(vals.std(ddof=1), 3) if len(vals) > 1 else np.nan,
                "n": int(len(vals)),
                "ci95": round(ci95(vals.std(ddof=1), len(vals)), 3) if len(vals) > 1 else np.nan,
            })
    df = pd.DataFrame(recs)
    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER + sorted(
        set(df["condition"]) - set(CONDITION_ORDER)), ordered=True)
    return df.sort_values(["dim", "condition", "base"]).reset_index(drop=True)


def dsp_vs_raw_pivot(cb):
    """Headline wide table: rows=condition, cols=dim x base showing 'mean (std)'."""
    cb = cb.copy()
    cb["cell"] = cb.apply(
        lambda r: f"{r['mean']:.2f} ({r['std']:.2f})" if pd.notna(r["mean"]) else "-", axis=1)
    wide = cb.pivot_table(index="condition", columns=["dim", "base"],
                          values="cell", aggfunc="first", observed=False)
    # order columns dim then base
    cols = [(d, b) for d in DIMS for b in BASE_ORDER if (d, b) in wide.columns]
    return wide.reindex(columns=pd.MultiIndex.from_tuples(cols))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyse P.835 listening-test results.")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out", default="./analysis_out")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="keep flagged raters in the main table (default: exclude)")
    ap.add_argument("--gold-fail-thresh", type=float, default=0.5,
                    help="flag rater if gold fail-rate exceeds this (default 0.5)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    resp = load_responses(args.responses)
    mp = pd.read_csv(args.mapping)
    if "expected_overall" not in mp.columns:
        mp["expected_overall"] = ""
    keep_cols = ["id", "role", "base", "condition", "utterance", "expected_overall"]
    merged = resp.merge(mp[keep_cols], on="id", how="left")

    missing = merged["role"].isna().sum()
    if missing:
        print(f"[WARN] {missing} ratings had ids not found in mapping.csv (ignored).")
        merged = merged.dropna(subset=["role"])

    # ---- QC ----
    gold, gold_per_rater = gold_checks(merged)
    raters = merged[["rater", "finalCheck"]].drop_duplicates("rater").copy()
    raters = raters.merge(gold_per_rater, on="rater", how="left")
    raters["gold_fail_rate"] = raters["gold_fail_rate"].fillna(0.0)
    raters["attention_fail"] = raters["finalCheck"].apply(
        lambda x: str(x) not in ("quality", "") )  # 'quality' is the correct answer
    raters["flagged"] = (raters["gold_fail_rate"] > args.gold_fail_thresh) | raters["attention_fail"]

    flagged_ids = set(raters.loc[raters["flagged"], "rater"])
    n_raters = raters["rater"].nunique()
    print(f"[INFO] raters: {n_raters} | flagged: {len(flagged_ids)} "
          f"({'kept' if args.keep_flagged else 'excluded'} from main table)")

    analysis = merged if args.keep_flagged else merged[~merged["rater"].isin(flagged_ids)]
    exp = analysis[analysis["role"] == "experimental"].copy()

    # ---- main dsp-vs-raw table ----
    cb = condition_base_table(exp)
    pivot = dsp_vs_raw_pivot(cb)

    # ---- inter-rater agreement (on experimental items) ----
    irr = {d: alpha_for_dimension(exp, d) for d in DIMS}
    mean_sd = {d: round(exp.groupby("id")[d].std(ddof=1).mean(), 3) for d in DIMS}

    # ---- anchors sanity ----
    anchors = (merged[merged["role"].isin(["anchor_clean", "anchor_noisy"])]
               .groupby("role")[DIMS].mean().round(2))

    # ---- console report ----
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)
    print("\n================ dsp vs raw  —  mean (std) ================")
    print(pivot.to_string())

    print("\n================ inter-rater agreement ================")
    for d in DIMS:
        a = irr[d]
        tag = ("good" if a >= 0.8 else "tentative" if a >= 0.667 else "low") if a == a else "n/a"
        print(f"  {d:4s}  Krippendorff alpha (ordinal) = "
              f"{a:.3f} [{tag}]   mean per-stimulus SD = {mean_sd[d]}")

    print("\n================ QC: gold / attention ================")
    if gold.empty:
        print("  No gold items found in mapping (role gold_high/gold_low).")
    else:
        bad = gold[gold["fail"]]
        print(f"  gold ratings: {len(gold)} | failures: {len(bad)}")
        if len(bad):
            show = (bad.groupby(["role", "id"])
                    .agg(n_fail=("fail", "size"),
                         mean_OVRL=("OVRL", "mean"),
                         expected=("expected_overall", "first")).reset_index())
            print(show.to_string(index=False))
    if not anchors.empty:
        print("\n  anchor means (clean should be highest, noisy lowest):")
        print(anchors.to_string())
    flg = raters[raters["flagged"]]
    if len(flg):
        print("\n  flagged raters:")
        print(flg[["rater", "gold_items", "gold_fails", "gold_fail_rate", "attention_fail"]]
              .to_string(index=False))

    # ---- write files ----
    cb.to_csv(os.path.join(args.out, "summary_condition_base.csv"), index=False)
    pivot.to_csv(os.path.join(args.out, "dsp_vs_raw_table.csv"))
    raters.to_csv(os.path.join(args.out, "rater_qc.csv"), index=False)
    if not gold.empty:
        gold.to_csv(os.path.join(args.out, "gold_item_checks.csv"), index=False)
    pd.DataFrame([{"dim": d, "krippendorff_alpha_ordinal": irr[d],
                   "mean_per_stimulus_sd": mean_sd[d]} for d in DIMS]).to_csv(
        os.path.join(args.out, "inter_rater_agreement.csv"), index=False)

    print(f"\n[DONE] wrote tables to {args.out}/")


if __name__ == "__main__":
    main()