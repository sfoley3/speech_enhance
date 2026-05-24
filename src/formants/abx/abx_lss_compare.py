"""
ABX on single-speaker USC-LSS: orig MRI vs the two enhanced versions.

Task: ON = vowel (phoneme), BY = speaker (LSS is one speaker, so one BY level)
for each of the three conditions -- orig_mri, meta, nvidia.  A single triplet
set is formed on the token set shared by all three conditions and scored
against each condition's own formants, so every condition is evaluated on
exactly the same (A, B, X) triplets and the comparison is genuinely paired.

Goal: the enhanced versions (meta, nvidia) should be *no worse* than orig MRI
at preserving vowel discriminability.  We report per-condition ABX accuracy
with triplet bootstrap CIs and paired Wilcoxon (Holm-corrected) between
conditions, and plot the three against the clean-EMA baseline (grey) produced
by abx_timit_baseline.py if its summary CSV is present.

Outputs (in --out-dir):
  abx_lss_summary.csv        per-condition accuracy + CI
  abx_lss_paired_tests.csv   pairwise Wilcoxon + Holm
  abx_lss_paired_cells.csv   per-cell accuracy, all conditions merged
  abx_lss_compare.pdf        bar plot vs grey EMA baseline

Usage:
  python abx_lss_compare.py
  python abx_lss_compare.py --results-dir /path/to/fave_results/USC_LSS \
                            --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import abx_utils as U


def _load_baseline(out_dir: Path) -> dict | None:
    """Read the EMA baseline summary written by abx_timit_baseline.py, if any."""
    path = out_dir / "abx_timit_baseline_summary.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return row


def make_plot(summary: pd.DataFrame, baseline: dict | None, out_pdf: Path) -> None:
    """Bar plot: ABX accuracy with CI error bars.

    The clean-EMA baseline is the leftmost bar (grey), followed by the
    comparator conditions in their own colors.  No legend; the x-axis labels
    carry the condition names.
    """
    labels: list[str] = []
    vals: list[float] = []
    los: list[float] = []
    his: list[float] = []
    colors: list[str] = []

    # EMA baseline first (grey), if available
    if baseline is not None:
        labels.append(U.COND_LABELS["orig_ema"])
        vals.append(float(baseline["accuracy_pooled"]))
        los.append(float(baseline["ci_lo"]))
        his.append(float(baseline["ci_hi"]))
        colors.append(U.COND_COLORS["orig_ema"])

    for c in summary["condition"]:
        row = summary[summary["condition"] == c].iloc[0]
        labels.append(U.COND_LABELS.get(c, c))
        vals.append(float(row["accuracy"]))
        los.append(float(row["ci_lo"]))
        his.append(float(row["ci_hi"]))
        colors.append(U.COND_COLORS.get(c, "#4c72b0"))

    vals = np.asarray(vals)
    yerr = np.vstack([vals - np.asarray(los), np.asarray(his) - vals])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.6, color=colors, edgecolor="black",
           yerr=yerr, capsize=6, error_kw=dict(ecolor="black", lw=1.2))

    ax.axhline(U.CHANCE, color="black", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ABX accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] wrote {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results-dir", type=Path, default=U.LSS_RESULTS_DIR,
                    help="root containing {condition}/*_tracks.csv")
    ap.add_argument("--out-dir", type=Path, default=U.DEFAULT_OUT_DIR)
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

    # load every condition, keep those that actually have tracks
    tracks_by_cond: dict[str, dict] = {}
    for cond in U.COMPARATORS:
        tracks = U.load_condition_tracks(args.results_dir, cond)
        if tracks and args.max_file > 0:
            tracks = U.filter_lss_files(tracks, args.max_file)
        if tracks:
            tracks_by_cond[cond] = tracks
        else:
            print(f"[warn] no tracks for {cond}; skipping")
    if not tracks_by_cond:
        raise SystemExit("[err] no conditions loaded")

    # form one shared triplet set and score it against every condition, so the
    # comparison is genuinely paired (same triplets, different audio).
    cells_by_cond, flags_by_cond = U.run_abx_paired(
        tracks_by_cond, speakers=None,  # LSS is a single speaker
        max_per_cell=max_per_cell, seed=args.seed,
    )

    summary_rows = []
    for cond in tracks_by_cond:  # COMPARATORS order
        flags = flags_by_cond[cond]
        if len(flags) == 0:
            print(f"[warn] no triplets for {cond}; skipping")
            continue
        acc, (lo, hi) = U.bootstrap_ci(flags, n_boot=args.n_boot, seed=args.seed)
        summary_rows.append({
            "condition": cond,
            "accuracy": acc,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_triplets": int(len(flags)),
            "chance": U.CHANCE,
        })
        print(f"[abx] {cond}: {len(cells_by_cond[cond])} cells, "
              f"{len(flags)} triplets  accuracy={acc:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")

    if not summary_rows:
        raise SystemExit("[err] no conditions produced ABX results")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "abx_lss_summary.csv", index=False)

    # paired tests across the conditions actually run
    conds_run = [r["condition"] for r in summary_rows]
    if len(conds_run) >= 2:
        merged, tests = U.paired_condition_tests(cells_by_cond, conds_run)
        tests.to_csv(args.out_dir / "abx_lss_paired_tests.csv", index=False)
        merged.to_csv(
            args.out_dir / "abx_lss_paired_cells.csv",
            index=False, encoding="utf-8",
        )
        print("\n--- paired Wilcoxon (per-cell accuracy, Holm-corrected) ---")
        print(tests.to_string(index=False))

    baseline = _load_baseline(args.out_dir)
    if baseline is None:
        print("[note] no EMA baseline summary found; run abx_timit_baseline.py "
              "first to draw the grey baseline.")
    make_plot(summary, baseline, args.out_dir / "abx_lss_compare.pdf")

    print("\n--- per-condition summary ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
