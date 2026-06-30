"""
Diagnostic formant-tracking plots: Raw vs DSP (USC_LSS), one sentence per plot.

LSS scans (``usc_s1_NN``, single speaker ``s1``) are multi-sentence. Sentence
boundaries are recovered by aligning each scan's transcript
(``transcripts/{scan}.txt``, split on ``.!?;``) against its word tier
(``textgrids/{scan}.TextGrid``) using the same aligner that builds the MOS
stimuli. Because every group/family is time-aligned to the same recording, one
``[start, end]`` window applies to all 8 cells.

For each randomly selected sentence segment we render a 4x2 PDF. Columns are the
audio groups (Raw = ``raw`` on the left, DSP = ``dsp`` on the right). Rows
are the four formant families (Original = ``mri``, Denoiser = ``meta``,
REUSE = ``nvidia``, PASE = ``pase``). Each panel shows that cell's wideband
spectrogram cropped to the sentence window with its smoothed formant tracks
(F1_s, F2_s, F3_s, in Hz) overlaid as scatter points.

Inputs (cluster layout):
    ORIG_DIR     /project2/shrikann_35/sfoley/data/single_spk_corpus
                   audio_{group}/usc_s1_NN.wav   (mri family)
                   textgrids/usc_s1_NN.TextGrid  transcripts/usc_s1_NN.txt
    ENHANCED_DIR /project2/shrikann_35/sfoley/data/enhanced_audio
                   {group}/<MODEL>/USC_LSS_<family>/usc_s1_NN.wav
    RESULTS_DIR  /project2/shrikann_35/sfoley/data/fave_results
                   {group}/USC_LSS/<folder>/usc_s1_NN_tracks.csv

Outputs:
    {OUT_DIR}/diagnostic_{scan}_seg{seg}.pdf  (one per selected sentence)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import librosa.display

import matplotlib.pyplot as plt


ORIG_DIR = Path("/project2/shrikann_35/sfoley/data/single_spk_corpus")
ENHANCED_DIR = Path("/project2/shrikann_35/sfoley/data/enhanced_audio")
RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results")

CORPUS = "USC_LSS"
GROUPS = ["raw", "dsp"]
FAMILIES = ["mri", "meta", "nvidia", "pase"]

# Row labels (one per family) and column titles (one per group).
FAMILY_LABELS = {
    "mri": "Original", "meta": "Denoiser",
    "nvidia": "REUSE", "pase": "PASE",
}
GROUP_TITLES = {"raw": "Raw", "dsp": "DSP"}
GROUP_PLOT_ORDER = ["raw", "dsp"]            # raw left, dsp right

# Per-scan transcripts + textgrids drive sentence splitting. The same time base
# applies to every group/family because all conditions are aligned to the same
# recording, so one window works for all 8 cells.
TG_DIR = ORIG_DIR / "textgrids"
TXT_DIR = ORIG_DIR / "transcripts"
ALIGN_WINDOW = 5                            # gappy-match search window (words)

# Keep sentence segments within this duration window (seconds) and cap how many
# come from any single scan.
MIN_DUR = 2.0
MAX_DUR = 10.0
MAX_PER_SCAN = 2

# Seconds of spectrogram context padded around each sentence window.
SENTENCE_PAD = 0.05

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figs" / "diagnostic" / "dsp_raw"

FORMANTS = ["F1_s", "F2_s", "F3_s"]
FORMANT_COLORS = {"F1_s": "#e41a1c", "F2_s": "#ff7f00", "F3_s": "#4daf4a"}

# Reuse the exact sentence aligner that builds the MOS stimuli, so diagnostic
# windows match real sentence-segment boundaries.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from sentence_stim_split import sentence_segments  # noqa: E402


def folder_name(group: str, family: str) -> str:
    """FAVE results subfolder: the mri family lives in ``{group}_mri``; the
    other families use the bare family name."""
    return f"{group}_mri" if family == "mri" else family


def audio_dir(group: str, family: str) -> Path:
    """Directory holding the wavs for one (group, family) cell."""
    if family == "mri":
        return ORIG_DIR / f"audio_{group}"
    if family == "meta":
        return ENHANCED_DIR / group / "META_denoiser" / "USC_LSS_meta"
    if family == "nvidia":
        return ENHANCED_DIR / group / "NVIDIA_REUSE" / "USC_LSS_nvidia"
    if family == "pase":
        return ENHANCED_DIR / group / "PASE" / "USC_LSS_pase"
    raise ValueError(f"unknown family: {family}")


def paths_for(group: str, family: str, stem: str) -> Tuple[Path, Path]:
    """Return (wav, tracks_csv) for one (group, family, file stem)."""
    wav = audio_dir(group, family) / f"{stem}.wav"
    tracks = (
        RESULTS_DIR / group / CORPUS / folder_name(group, family)
        / f"{stem}_tracks.csv"
    )
    return wav, tracks


def all_cells_exist(scan: str) -> bool:
    """True only if every (group, family) cell has both a wav and a
    ``*_tracks.csv`` on disk for ``scan``."""
    for group in GROUPS:
        for family in FAMILIES:
            wav, tracks = paths_for(group, family, scan)
            if not (wav.is_file() and tracks.is_file()):
                return False
    return True


def build_segment_pool() -> List[dict]:
    """Discover sentence segments across all scans that have a transcript, a
    textgrid, and all 8 (group, family) cells. Returns dicts with
    ``scan``/``seg``/``start``/``end``/``text``/``dur`` (duration-filtered)."""
    if not TG_DIR.is_dir():
        raise RuntimeError(f"missing textgrid dir: {TG_DIR}")
    pool: List[dict] = []
    for tg in sorted(TG_DIR.glob("*.TextGrid")):
        scan = tg.stem
        txt = TXT_DIR / f"{scan}.txt"
        if not txt.is_file():
            continue
        if not all_cells_exist(scan):
            continue
        try:
            segs = sentence_segments(str(tg), str(txt), ALIGN_WINDOW)
        except Exception as e:
            print(f"[warn] align fail {scan}: {e}", file=sys.stderr)
            continue
        for i, ch in enumerate(segs):
            dur = ch["end"] - ch["start"]
            if MIN_DUR <= dur <= MAX_DUR:
                pool.append({"scan": scan, "seg": i, "start": ch["start"],
                             "end": ch["end"], "text": ch["text"], "dur": dur})
    return pool


# ---------- plotting ----------


# Drop formant points within this many seconds of the sentence-window edges.
# The window is padded (see ``SENTENCE_PAD``) for spectrogram context, so this
# trim removes stray points from neighboring sentences that fall just inside the
# padded boundary while keeping the spectrogram context.
TRACK_EDGE_TRIM = 0.06


def _load_tracks_in_window(
    tracks_csv: Path, t0: float, t1: float
) -> pd.DataFrame:
    df = pd.read_csv(tracks_csv, usecols=["time", *FORMANTS])
    for c in ["time", *FORMANTS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"])
    lo = t0 + TRACK_EDGE_TRIM
    hi = t1 - TRACK_EDGE_TRIM
    if hi <= lo:
        lo, hi = t0, t1
    return df[(df["time"] >= lo) & (df["time"] <= hi)].copy()


def _plot_panel(
    ax: plt.Axes,
    wav_path: Path,
    tracks_csv: Path,
    t0: float,
    t1: float,
    title: str,
    show_ylabel: bool,
    show_legend: bool,
) -> int:
    """Render mel spectrogram + formant overlay onto ``ax``. Returns vowel-row
    count plotted (for logging)."""
    # load + crop audio
    y, sr = librosa.load(str(wav_path), sr=None, offset=t0, duration=max(0.0, t1 - t0))
    if y.size == 0:
        ax.set_title(f"{title}\n[empty audio in window]")
        return 0
    fmax = min(8000.0, sr / 2.0)
    # Wideband STFT spectrogram (~8 ms window) - smears harmonics into formant
    # bands while keeping good time resolution. Linear-Hz axis so formant
    # overlays sit on the true frequency they represent (no mel remap, no
    # empty-mel-filter artifacts).
    win_ms = 0.008
    n_fft = max(128, int(2 ** round(np.log2(win_ms * sr))))
    hop = max(1, n_fft // 8)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="Blues",
    )
    ax.set_ylim(0.0, fmax)
    ax.set_yticks(np.arange(0, fmax + 1, 2000))
    ax.tick_params(axis="both", labelsize=15)
    # x-axis is relative to the sentence window (0..t1-t0); librosa's
    # specshow already plots from 0, so leave the ticks alone.
    ax.set_xlim(0.0, t1 - t0)

    # formant overlay. librosa's mel y-axis takes Hz values directly and
    # handles the nonlinear mapping internally, so plot raw Hz (not mel).
    df = _load_tracks_in_window(tracks_csv, t0, t1)
    n_pts = 0
    for f_col in FORMANTS:
        hz = df[f_col].to_numpy()
        ok = np.isfinite(hz) & (hz > 0) & (hz <= fmax)
        if not ok.any():
            continue
        t_local = df["time"].to_numpy()[ok] - t0
        ax.scatter(
            t_local,
            hz[ok],
            s=10,
            c=FORMANT_COLORS[f_col],
            label=f_col.replace("_s", ""),
            edgecolors="none",
            alpha=0.9,
        )
        n_pts += int(ok.sum())

    ax.set_title(title, fontsize=15)
    if show_ylabel:
        ax.set_ylabel("Frequency (Hz)", fontsize=15)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Time (s)", fontsize=15)
    # if show_legend and n_pts:
    #     ax.legend(loc="upper right", fontsize=8, framealpha=0.6, markerscale=2)
    return n_pts


def render_segment_pdf(
    scan: str,
    seg: int,
    start: float,
    end: float,
    text: str,
    out_dir: Path,
) -> Path:
    """Make one 4x2 PDF for a single sentence segment of ``scan``: rows are
    families (Original, Denoiser, REUSE, PASE); columns are groups (Raw, DSP).
    Each panel shows that cell's wideband spectrogram cropped to the sentence
    window with its F1/F2/F3 formant tracks overlaid."""
    # one shared window for all 8 cells (the recordings are time-aligned)
    t0 = max(0.0, start - SENTENCE_PAD)
    t1 = end + SENTENCE_PAD
    max_dur = t1 - t0

    fig, axes = plt.subplots(
        len(FAMILIES), len(GROUP_PLOT_ORDER),
        figsize=(13, 15), constrained_layout=True,
        sharex=True, sharey=True,
    )

    per_panel_counts: Dict[Tuple[str, str], int] = {}
    last_row = len(FAMILIES) - 1
    for r, family in enumerate(FAMILIES):
        for c, group in enumerate(GROUP_PLOT_ORDER):
            ax = axes[r, c]
            wav, tracks = paths_for(group, family, scan)
            n_pts = _plot_panel(
                ax,
                wav,
                tracks,
                t0,
                t1,
                title=(GROUP_TITLES[group] if r == 0 else ""),
                show_ylabel=(c == 0),
                show_legend=(r == 0 and c == 0),
            )
            ax.set_xlim(0.0, max_dur)
            if r != last_row:
                ax.set_xlabel("")
            per_panel_counts[(group, family)] = n_pts
        # family row label on the left of each row
        axes[r, 0].annotate(
            FAMILY_LABELS[family],
            xy=(-0.22, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center",
            fontsize=15, fontweight="bold",
        )

    fig.suptitle(f'"{text}"', fontsize=16)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diagnostic_{scan}_seg{seg:02d}.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"[plot] {scan} seg{seg:02d} -> {out_path.name} "
        f"(formant points per panel: " +
        ", ".join(
            f"{g}/{fam}={per_panel_counts[(g, fam)]}"
            for fam in FAMILIES for g in GROUP_PLOT_ORDER
        ) + ")",
        flush=True,
    )
    return out_path


# ---------- selection driver ----------


def pick_segments(seed: int, n: int) -> List[dict]:
    """Randomly pick up to ``n`` sentence segments, capping how many come from
    any single scan (see ``MAX_PER_SCAN``)."""
    pool = build_segment_pool()
    if not pool:
        return []
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen: List[dict] = []
    per_scan: Dict[str, int] = {}
    for u in pool:
        if per_scan.get(u["scan"], 0) >= MAX_PER_SCAN:
            continue
        chosen.append(u)
        per_scan[u["scan"]] = per_scan.get(u["scan"], 0) + 1
        if len(chosen) >= n:
            break
    return chosen


# ---------- CLI ----------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sentence selection (default 0)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default {DEFAULT_OUT_DIR})")
    p.add_argument("--n-sentences", type=int, default=4,
                   help="Number of sentences to plot (default 4)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # wipe any prior diagnostic PDFs so the output directory only reflects
    # the current run
    if args.out_dir.is_dir():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    picks = pick_segments(args.seed, args.n_sentences)
    if not picks:
        print("[err] no sentence segments with all (group, family) cells present",
              file=sys.stderr)
        return 1
    print("[select] " + ", ".join(f"{u['scan']}#{u['seg']}" for u in picks))
    for u in picks:
        try:
            render_segment_pdf(
                u["scan"], u["seg"], u["start"], u["end"], u["text"],
                args.out_dir,
            )
        except Exception as e:
            print(f"[err] render fail {u['scan']} seg{u['seg']}: {e}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())