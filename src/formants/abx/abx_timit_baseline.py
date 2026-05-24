"""
ABX baseline (ceiling) on clean USC-TIMIT EMA formants.

Task: ON = vowel (phoneme), BY = speaker -- within-speaker vowel-category
discriminability of the F1_s/F2_s/F3_s tracks under the DTW + framewise-KL
distance.  Computed per speaker (F1, F5, M1, M3) and averaged across speakers
to give the grey "clean" baseline that the LSS comparison plots against.

Outputs (in --out-dir):
  abx_timit_baseline_per_speaker.csv   per-speaker accuracy
  abx_timit_baseline_per_phoneme.csv   per (speaker x vowel-contrast) cells
  abx_timit_baseline_summary.csv       overall accuracy + bootstrap CI

Usage:
  python abx_timit_baseline.py
  python abx_timit_baseline.py --results-dir /path/to/fave_results/USC-TIMIT \
                               --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import abx_utils as U


def _weighted_accuracy(df: pd.DataFrame) -> float:
    """Triplet-weighted mean of the per-cell ABX accuracy."""
    n = df["n"].to_numpy(dtype=float)
    s = df["score"].to_numpy(dtype=float)
    return float((s * n).sum() / n.sum()) if n.sum() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results-dir", type=Path, default=U.TIMIT_RESULTS_DIR,
                    help="root containing {condition}/{spk}/*_tracks.csv")
    ap.add_argument("--out-dir", type=Path, default=U.DEFAULT_OUT_DIR)
    ap.add_argument("--max-sentence", type=int, default=455,
                    help="keep only sentence ids <= this; drops the last range "
                         "file per speaker (usctimit_*_456_460) so the sentence "
                         "set matches LSS files 1-37. 0 = no cap.")
    ap.add_argument("--max-per-cell", type=int, default=5,
                    help="cap on triplets per (speaker x A/X-vowel x B-vowel) "
                         "cell; sampled without replacement. Realized total "
                         "~= max_per_cell * n_vowels * (n_vowels-1) per speaker. "
                         "0 = no cap.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    max_per_cell = args.max_per_cell if args.max_per_cell > 0 else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] results_dir = {args.results_dir}")
    print(f"[cfg] out_dir     = {args.out_dir}")

    # 1. load clean EMA tracks, restricted to the shared sentence set
    tracks = U.load_condition_tracks(args.results_dir, U.REFERENCE)
    if args.max_sentence > 0:
        tracks = U.filter_timit_sentences(tracks, args.max_sentence)

    # 2. run the in-memory ABX (BY=speaker keeps every triplet within a speaker)
    cells, flags = U.run_abx(
        tracks, speakers=set(U.TIMIT_SPKS),
        max_per_cell=max_per_cell, seed=args.seed,
    )
    if len(flags) == 0:
        raise SystemExit("[err] no triplets formed; check --results-dir layout")
    print(f"[abx] {len(cells)} cells, {len(flags)} triplets")

    # 3. per-speaker accuracy + across-speaker mean
    per_spk = (
        cells.groupby("speaker", sort=True)
        .apply(_weighted_accuracy)
        .rename("accuracy")
        .reset_index()
    )
    per_spk["n_triplets"] = (
        cells.groupby("speaker", sort=True)["n"].sum().reset_index(drop=True)
    )
    across_spk_mean = float(per_spk["accuracy"].mean())

    # 4. triplet-level bootstrap CI (pooled over all speakers)
    acc, (lo, hi) = U.bootstrap_ci(flags, n_boot=args.n_boot, seed=args.seed)

    per_spk.to_csv(args.out_dir / "abx_timit_baseline_per_speaker.csv", index=False)
    cells.to_csv(
        args.out_dir / "abx_timit_baseline_per_phoneme.csv",
        index=False, encoding="utf-8",
    )
    pd.DataFrame([{
        "condition": U.REFERENCE,
        "accuracy_pooled": acc,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_across_speakers": across_spk_mean,
        "n_triplets": int(len(flags)),
        "chance": U.CHANCE,
    }]).to_csv(args.out_dir / "abx_timit_baseline_summary.csv", index=False)

    print("\n--- per-speaker ABX accuracy (clean EMA) ---")
    print(per_spk.to_string(index=False))
    print(f"\n[baseline] across-speaker mean = {across_spk_mean:.4f}")
    print(f"[baseline] pooled triplet accuracy = {acc:.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]  (chance={U.CHANCE})")


if __name__ == "__main__":
    main()
