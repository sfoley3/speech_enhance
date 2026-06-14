"""Grouped ABX comparison on USC-LSS (raw/dsp) against EMA baseline.

Layout expected under ``--results-dir``:
        {results_dir}/raw/USC_LSS/{raw_mri,meta,nvidia,pase}
        {results_dir}/dsp/USC_LSS/{dsp_mri,meta,nvidia,pase}

For each (group, family) condition we build one shared ABX triplet set on the
token intersection across all loaded conditions and score all conditions on that
same set (paired by construction).

Outputs (in --out-dir):
    abx_lss_grouped_summary.csv            grouped condition summary
    abx_lss_raw_vs_dsp_paired_tests.csv    paired Wilcoxon per family
    abx_lss_vs_ema_tests.csv               per-condition tests vs EMA baseline
    abx_lss_paired_cells.csv               per-cell score table (tidy)
    abx_lss_compare_grouped.pdf            grouped bar plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats
from matplotlib.patches import Patch  # noqa: E402

import abx_utils as U


def _load_baseline(path: Path) -> dict | None:
    """Read the EMA baseline summary written by abx_timit_baseline.py, if any."""
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return row


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - rank))
    running = np.maximum.accumulate(adj[order])
    for k, idx in enumerate(order):
        adj[idx] = running[k]
    return adj


def _folder_name(group: str, family: str) -> str:
    if family == "mri":
        return f"{group}_mri"
    return family


def _condition_id(group: str, family: str) -> str:
    return f"{group}__{family}"


def _sem_from_cells(cells: pd.DataFrame) -> float:
    vals = cells["score"].to_numpy(dtype=float)
    n = len(vals)
    if n < 2:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(n))


def make_plot(summary: pd.DataFrame, baseline: dict | None, out_pdf: Path) -> None:
    """Grouped bars: EMA baseline + per-family raw/dsp bars with SEM."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    family_gap = 1.4
    centers = 1.2 + np.arange(len(U.FAMILIES), dtype=float) * family_gap
    bar_w = 0.34

    # Leftmost EMA bar (grey)
    ema_x = 0.0
    if baseline is not None:
        ema = float(baseline["accuracy_pooled"])
        lo = float(baseline.get("ci_lo", ema))
        hi = float(baseline.get("ci_hi", ema))
        ema_sem = max(0.0, min(ema - lo, hi - ema)) / 1.96 if hi >= lo else 0.0
        ax.bar(
            [ema_x], [ema], width=0.6,
            color=U.COND_COLORS["orig_ema"], edgecolor="black",
            yerr=[ema_sem], capsize=6,
            error_kw=dict(ecolor="black", lw=1.2),
        )

    # Per-family grouped bars
    raw_handle = None
    dsp_handle = None
    for i, family in enumerate(U.FAMILIES):
        c = centers[i]
        for group, dx in [("dsp", -bar_w / 2), ("raw_clean", bar_w / 2)]:
            row = summary[
                (summary["group"] == group) & (summary["family"] == family)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            h = ax.bar(
                [c + dx], [float(r["accuracy"])], width=bar_w,
                color=U.COND_COLORS[group], edgecolor="black",
                yerr=[float(r["sem"])], capsize=5,
                error_kw=dict(ecolor="black", lw=1.2),
                label=group if ((group == "raw_clean" and raw_handle is None)
                                or (group == "dsp" and dsp_handle is None)) else None,
            )
            if group == "raw_clean" and raw_handle is None:
                raw_handle = h
            if group == "dsp" and dsp_handle is None:
                dsp_handle = h

    ax.axhline(U.CHANCE, color="black", ls=":", lw=1.0)
    ax.set_xticks(np.concatenate([[ema_x], centers]))
    ax.set_xticklabels(["EMA", *[U.COND_LABELS[f] for f in U.FAMILIES]])
    ax.set_ylabel("ABX accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xlim(-0.6, centers[-1] + 0.8)
    handles = [Patch(facecolor=U.COND_COLORS["raw_clean"], edgecolor="black", label="Raw"),
               Patch(facecolor=U.COND_COLORS["dsp"], edgecolor="black", label="DSP")]
    ax.legend(handles=handles, title="", loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] wrote {out_pdf}")


def _raw_vs_dsp_tests(cells_by_cond: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired Wilcoxon tests of DSP vs RAW for each family."""
    key_cols = ["speaker", "phoneme_ax", "phoneme_b"]
    recs = []
    for family in U.FAMILIES:
        raw_id = _condition_id("raw_clean", family)
        dsp_id = _condition_id("dsp", family)
        if raw_id not in cells_by_cond or dsp_id not in cells_by_cond:
            continue
        raw = cells_by_cond[raw_id][key_cols + ["score"]].rename(
            columns={"score": "raw_clean"}
        )
        dsp = cells_by_cond[dsp_id][key_cols + ["score"]].rename(
            columns={"score": "dsp"}
        )
        m = raw.merge(dsp, on=key_cols, how="inner")
        x = m["dsp"].to_numpy(dtype=float)
        y = m["raw_clean"].to_numpy(dtype=float)
        try:
            W, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            W, p = float("nan"), float("nan")
        med_diff = float(np.median(x - y)) if len(m) else float("nan")
        direction = (
            "dsp_better" if med_diff > 0 else
            "raw_better" if med_diff < 0 else
            "no_diff"
        )
        recs.append({
            "family": family,
            "median_dsp": float(np.median(x)) if len(m) else float("nan"),
            "median_raw": float(np.median(y)) if len(m) else float("nan"),
            "median_diff_dsp_minus_raw": med_diff,
            "W": float(W),
            "p": float(p),
            "n_cells": int(len(m)),
            "direction": direction,
        })
    out = pd.DataFrame(recs)
    if len(out):
        valid = out["p"].notna().to_numpy()
        holm = np.full(len(out), np.nan)
        if valid.any():
            holm[valid] = _holm(out.loc[valid, "p"].to_numpy())
        out["p_holm"] = holm
    return out


def _vs_ema_tests(
    cells_by_cond: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    baseline: dict | None,
) -> pd.DataFrame:
    """Wilcoxon signed-rank tests of condition per-cell score vs EMA scalar."""
    if baseline is None:
        return pd.DataFrame()
    ema_acc = float(baseline["accuracy_pooled"])
    recs = []
    for _, r in summary.iterrows():
        cid = str(r["condition_id"])
        cells = cells_by_cond[cid]
        vals = cells["score"].to_numpy(dtype=float)
        diffs = vals - ema_acc
        try:
            W, p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            W, p = float("nan"), float("nan")
        med_diff = float(np.median(diffs)) if len(diffs) else float("nan")
        recs.append({
            "condition_id": cid,
            "group": str(r["group"]),
            "family": str(r["family"]),
            "ema_accuracy": ema_acc,
            "median_score": float(np.median(vals)) if len(vals) else float("nan"),
            "median_diff_vs_ema": med_diff,
            "W": float(W),
            "p": float(p),
            "n_cells": int(len(vals)),
            "direction": (
                "better_than_ema" if med_diff > 0 else
                "worse_than_ema" if med_diff < 0 else
                "no_diff"
            ),
            "test_type": "wilcoxon_signed_rank_vs_ema_scalar",
            "notes": (
                "EMA comes from USC-TIMIT baseline summary; this is not a strict "
                "item-level paired LSS-vs-EMA test."
            ),
        })
    out = pd.DataFrame(recs)
    if len(out):
        valid = out["p"].notna().to_numpy()
        holm = np.full(len(out), np.nan)
        if valid.any():
            holm[valid] = _holm(out.loc[valid, "p"].to_numpy())
        out["p_holm"] = holm
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results-dir", type=Path, default=U.LSS_RESULTS_DIR,
                    help="root containing raw/USC_LSS and dsp/USC_LSS")
    ap.add_argument("--out-dir", type=Path, default=U.DEFAULT_OUT_DIR)
    ap.add_argument(
        "--baseline-summary", type=Path, default=None,
        help=("path to abx_timit_baseline_summary.csv; defaults to "
              "{out_dir}/abx_timit_baseline_summary.csv"),
    )
    ap.add_argument("--max-file", type=int, default=37,
                    help="keep only LSS files usc_s1_1 .. this number; drops "
                         "usc_s1_38 (Rainbow Passage) so the sentence set "
                         "matches TIMIT sentences 1-455. 0 = no cap.")
    ap.add_argument("--max-per-cell", type=int, default=5,
                    help="cap on triplets per (speaker x A/X-vowel x B-vowel) "
                         "cell; sampled without replacement. Same seed across "
                         "conditions keeps the comparison paired. Realized "
                         "~= max_per_cell * n_vowels * (n_vowels-1). 0 = no cap.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    max_per_cell = args.max_per_cell if args.max_per_cell > 0 else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] results_dir = {args.results_dir}")
    print(f"[cfg] out_dir     = {args.out_dir}")

    # load every grouped condition, keep those that actually have tracks
    tracks_by_cond: dict[str, dict] = {}
    meta_rows: list[dict] = []
    for group in U.GROUPS:
        for family in U.FAMILIES:
            cond_dir = args.results_dir / group / "USC_LSS"
            cond_name = _folder_name(group, family)
            cid = _condition_id(group, family)
            tracks = U.load_condition_tracks(cond_dir, cond_name)
            if tracks and args.max_file > 0:
                tracks = U.filter_lss_files(tracks, args.max_file)
            if tracks:
                tracks_by_cond[cid] = tracks
                meta_rows.append({
                    "condition_id": cid,
                    "group": group,
                    "family": family,
                    "folder": cond_name,
                })
            else:
                print(f"[warn] no tracks for {group}/{cond_name}; skipping")
    if not tracks_by_cond:
        raise SystemExit("[err] no conditions loaded")

    # form one shared triplet set and score it against every condition, so the
    # comparison is genuinely paired (same triplets, different audio).
    cells_by_cond, flags_by_cond = U.run_abx_paired(
        tracks_by_cond, speakers=None,  # LSS is a single speaker
        max_per_cell=max_per_cell, seed=args.seed,
    )

    summary_rows = []
    meta = pd.DataFrame(meta_rows)
    for _, m in meta.iterrows():
        cid = str(m["condition_id"])
        flags = flags_by_cond[cid]
        if len(flags) == 0:
            print(f"[warn] no triplets for {cid}; skipping")
            continue
        acc, (lo, hi) = U.bootstrap_ci(flags, n_boot=args.n_boot, seed=args.seed)
        cells = cells_by_cond[cid]
        summary_rows.append({
            "condition_id": cid,
            "group": str(m["group"]),
            "family": str(m["family"]),
            "folder": str(m["folder"]),
            "accuracy": acc,
            "sem": _sem_from_cells(cells),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_cells": int(len(cells)),
            "n_triplets": int(len(flags)),
            "chance": U.CHANCE,
        })
        print(f"[abx] {cid}: {len(cells)} cells, "
              f"{len(flags)} triplets  accuracy={acc:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")

    if not summary_rows:
        raise SystemExit("[err] no conditions produced ABX results")

    summary = pd.DataFrame(summary_rows)
    summary["family"] = pd.Categorical(
        summary["family"], categories=U.FAMILIES, ordered=True,
    )
    summary["group"] = pd.Categorical(
        summary["group"], categories=["dsp", "raw_clean"], ordered=True,
    )
    summary = summary.sort_values(["family", "group"]).reset_index(drop=True)
    summary.to_csv(args.out_dir / "abx_lss_grouped_summary.csv", index=False)

    # Save per-cell table in tidy format for downstream checks/plots.
    per_cell = []
    for _, r in summary.iterrows():
        cid = str(r["condition_id"])
        d = cells_by_cond[cid].copy()
        d.insert(0, "condition_id", cid)
        d.insert(1, "group", str(r["group"]))
        d.insert(2, "family", str(r["family"]))
        per_cell.append(d)
    pd.concat(per_cell, ignore_index=True).to_csv(
        args.out_dir / "abx_lss_paired_cells.csv",
        index=False, encoding="utf-8",
    )

    # Paired tests requested: DSP vs RAW within each family.
    raw_dsp_tests = _raw_vs_dsp_tests(cells_by_cond)
    raw_dsp_tests.to_csv(
        args.out_dir / "abx_lss_raw_vs_dsp_paired_tests.csv", index=False,
    )
    if len(raw_dsp_tests):
        print("\n--- DSP vs RAW paired Wilcoxon (per family) ---")
        print(raw_dsp_tests.to_string(index=False))

    baseline_path = (
        args.baseline_summary
        if args.baseline_summary is not None
        else args.out_dir / "abx_timit_baseline_summary.csv"
    )
    baseline = _load_baseline(baseline_path)
    if baseline is None:
        print("[note] no EMA baseline summary found; run abx_timit_baseline.py "
              "first to draw the grey baseline.")
    vs_ema = _vs_ema_tests(cells_by_cond, summary, baseline)
    vs_ema.to_csv(args.out_dir / "abx_lss_vs_ema_tests.csv", index=False)
    if len(vs_ema):
        print("\n--- condition vs EMA tests ---")
        print(vs_ema.to_string(index=False))

    make_plot(summary, baseline, args.out_dir / "abx_lss_compare_grouped.pdf")

    print("\n--- grouped per-condition summary ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
