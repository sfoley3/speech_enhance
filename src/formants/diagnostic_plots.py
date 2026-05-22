"""
Diagnostic formant-tracking plots.

For each of 4 speakers (F1, F5, M1, M3) pick one random utterance (a multi-
sentence USC-TIMIT wav). Within that utterance, pick one sentence id from the
filename's [start, end] range and locate its time window in the FAVE
``_recoded.TextGrid`` word tier. Render a 2x2 PDF (one panel per condition:
orig_ema, orig_mri, meta, nvidia) with a mel spectrogram cropped to that
sentence window and the smoothed formant tracks (F1_s, F2_s, F3_s, in Hz,
projected onto the mel y-axis) overlaid as scatter points.

Inputs (cluster layout):
    ORIG_DIR     /project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT
    ENHANCED_DIR /project2/shrikann_35/sfoley/data/enhanced_audio
    RESULTS_DIR  /project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT
    SENTENCES    {ORIG_DIR}/list_of_sentences.txt

Outputs:
    {OUT_DIR}/diagnostic_{spk}_{rec_stem}_sent{sid}.pdf  (one per speaker)
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import librosa.display

import matplotlib.pyplot as plt


ORIG_DIR = Path("/project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT")
ENHANCED_DIR = Path("/project2/shrikann_35/sfoley/data/enhanced_audio")
RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT")
SENTENCES_FILE = ORIG_DIR / "list_of_sentences.txt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figs" / "diagnostic"


SPKS = ["F1", "F5", "M1", "M3"]
CONDITIONS = ["orig_ema", "orig_mri", "meta", "nvidia"]
COND_TITLES = {
    "orig_ema": "Original EMA (clean)",
    "orig_mri": "Original MRI (noisy)",
    "meta": "META denoiser",
    "nvidia": "NVIDIA REUSE",
}

FORMANTS = ["F1_s", "F2_s", "F3_s"]
FORMANT_COLORS = {"F1_s": "#e41a1c", "F2_s": "#ff7f00", "F3_s": "#4daf4a"}

# Modality token in the wav/textgrid stems: "ema" for orig_ema, "mri" for
# orig_mri/meta/nvidia (the denoised audio is built from the MRI recordings).
MODALITY = {
    "orig_ema": "ema",
    "orig_mri": "mri",
    "meta": "mri",
    "nvidia": "mri",
}


def wav_dir(condition: str, spk: str) -> Path:
    if condition == "orig_ema":
        return ORIG_DIR / "EMA" / "Data" / spk / "wav"
    if condition == "orig_mri":
        return ORIG_DIR / "MRI" / "Data" / spk / "wav"
    if condition == "meta":
        return ENHANCED_DIR / "META_denoiser" / "USC-TIMIT_denoiser" / spk / "wav"
    if condition == "nvidia":
        return ENHANCED_DIR / "NVIDIA_REUSE" / "USC-TIMIT_reuse" / spk
    raise ValueError(f"unknown condition: {condition}")


# Filename: usctimit_{mod}_{spk}_{startID}_{endID}.wav
WAV_RE = re.compile(
    r"^usctimit_(?P<mod>ema|mri)_(?P<spk>[a-z]\d+)_(?P<start>\d{3})_(?P<end>\d{3})$",
    flags=re.IGNORECASE,
)


# ---------- sentence list ----------


_PUNCT_RE = re.compile(r"[^a-z0-9' ]+")


def _norm_text(s: str) -> List[str]:
    """Lowercase, strip punctuation (keep apostrophes), split on whitespace."""
    s = s.lower().replace("\u2019", "'")
    s = _PUNCT_RE.sub(" ", s)
    return [w for w in s.split() if w]


def load_sentences(path: Path) -> Dict[int, List[str]]:
    """Parse ``NNN : sentence text`` lines into ``{id: [tokens]}``."""
    out: Dict[int, List[str]] = {}
    line_re = re.compile(r"^\s*(\d{1,4})\s*:\s*(.+?)\s*$")
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            m = line_re.match(raw)
            if not m:
                continue
            sid = int(m.group(1))
            out[sid] = _norm_text(m.group(2))
    if not out:
        raise RuntimeError(f"no sentences parsed from {path}")
    return out


# ---------- minimal Praat TextGrid (long-form) parser ----------


_TG_NUM = r"-?\d+(?:\.\d+)?"
_TG_INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]\s*:\s*"
    rf"xmin\s*=\s*(?P<xmin>{_TG_NUM})\s*"
    rf"xmax\s*=\s*(?P<xmax>{_TG_NUM})\s*"
    r"text\s*=\s*\"(?P<text>(?:[^\"]|\"\")*)\"",
    flags=re.DOTALL,
)
_TG_TIER_RE = re.compile(
    r"item\s*\[\d+\]\s*:\s*"
    r"class\s*=\s*\"(?P<class>[^\"]+)\"\s*"
    r"name\s*=\s*\"(?P<name>[^\"]+)\"",
    flags=re.DOTALL,
)


def load_words_tier(tg_path: Path) -> List[Tuple[float, float, str]]:
    """Return the ``words`` interval tier as a list of ``(xmin, xmax, text)``.

    Only handles Praat long-form TextGrid files (what FAVE writes). Empty/`sp`/
    `sil` intervals are preserved; the caller decides whether to skip them.
    """
    txt = tg_path.read_text(encoding="utf-8", errors="replace")
    # find each "item [n]:" block, then within each block its intervals
    # easiest: split on "item [" occurrences and inspect each chunk
    chunks = re.split(r"item\s*\[\d+\]\s*:", txt)
    for chunk in chunks[1:]:
        m = re.search(
            r"class\s*=\s*\"(?P<class>[^\"]+)\"\s*"
            r"name\s*=\s*\"(?P<name>[^\"]+)\"",
            chunk,
        )
        if not m:
            continue
        if m.group("name") != "words":
            continue
        intervals: List[Tuple[float, float, str]] = []
        for im in _TG_INTERVAL_RE.finditer(chunk):
            xmin = float(im.group("xmin"))
            xmax = float(im.group("xmax"))
            text = im.group("text").replace("\"\"", "\"").strip()
            intervals.append((xmin, xmax, text))
        return intervals
    raise RuntimeError(f"no 'words' tier in {tg_path}")


def _is_silence(token: str) -> bool:
    t = token.strip().lower()
    return t in ("", "sp", "sil", "{sl}", "{ns}", "{ls}", "{lg}", "{cg}")


def find_sentence_window(
    words: List[Tuple[float, float, str]],
    target_tokens: List[str],
    pad: float = 0.05,
) -> Optional[Tuple[float, float, int, int]]:
    """Locate a contiguous (silence-skipping) run of word intervals whose text
    sequence equals ``target_tokens`` (normalized). Returns
    ``(t_start, t_end, first_idx, last_idx)`` or ``None`` if not found.
    Returns the *first* match.
    """
    if not target_tokens:
        return None
    norm = [_norm_text(w[2])[0] if _norm_text(w[2]) else "" for w in words]
    n = len(words)
    for i in range(n):
        if _is_silence(words[i][2]):
            continue
        j = i
        ti = 0
        while j < n and ti < len(target_tokens):
            if _is_silence(words[j][2]):
                j += 1
                continue
            if norm[j] != target_tokens[ti]:
                break
            ti += 1
            j += 1
        if ti == len(target_tokens):
            last = j - 1
            t_start = max(0.0, words[i][0] - pad)
            t_end = words[last][1] + pad
            return t_start, t_end, i, last
    return None


# ---------- per-condition path resolution ----------


def stem_for(condition: str, spk: str, start: str, end: str) -> str:
    return f"usctimit_{MODALITY[condition]}_{spk.lower()}_{start}_{end}"


def paths_for(
    condition: str, spk: str, start: str, end: str
) -> Tuple[Path, Path, Path]:
    """Return (wav, textgrid, tracks_csv) for one (condition, recording)."""
    stem = stem_for(condition, spk, start, end)
    wav = wav_dir(condition, spk) / f"{stem}.wav"
    tg = RESULTS_DIR / condition / f"{stem}_recoded.TextGrid"
    tracks = RESULTS_DIR / condition / f"{stem}_tracks.csv"
    return wav, tg, tracks


def candidate_recordings(spk: str) -> List[Tuple[str, str]]:
    """List (start, end) id ranges for which all four conditions have the
    required wav, textgrid, and tracks files on disk."""
    ema_dir = wav_dir("orig_ema", spk)
    if not ema_dir.is_dir():
        return []
    out: List[Tuple[str, str]] = []
    for wav in sorted(ema_dir.glob("*.wav")):
        m = WAV_RE.match(wav.stem)
        if not m or m.group("mod").lower() != "ema":
            continue
        if m.group("spk").upper() != spk.upper():
            continue
        start, end = m.group("start"), m.group("end")
        ok = True
        for cond in CONDITIONS:
            w, tg, tr = paths_for(cond, spk, start, end)
            if not (w.is_file() and tg.is_file() and tr.is_file()):
                ok = False
                break
        if ok:
            out.append((start, end))
    return out


# ---------- plotting ----------


# Drop formant points within this many seconds of the sentence window edges.
# The sentence window is padded (see ``find_sentence_window``) so adjacent
# vowels from neighboring sentences sometimes fall just inside the boundary;
# this trim removes those stray edge points while still keeping the spectrogram
# context.
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


def render_utterance_pdf(
    spk: str,
    start: str,
    end: str,
    sid: int,
    sentence_text: str,
    out_dir: Path,
) -> Path:
    """Make one 2x2 PDF for this (speaker, recording, sentence)."""
    # per-condition sentence windows (EMA and MRI textgrids differ)
    windows: Dict[str, Tuple[float, float]] = {}
    tokens = _norm_text(sentence_text)
    for cond in CONDITIONS:
        _, tg, _ = paths_for(cond, spk, start, end)
        words = load_words_tier(tg)
        win = find_sentence_window(words, tokens)
        if win is None:
            raise RuntimeError(
                f"sentence {sid} tokens {tokens!r} not found in {tg}"
            )
        windows[cond] = (win[0], win[1])

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 7.5), constrained_layout=True,
        sharex=True, sharey=True,
    )
    # Shared time axis spans the longest per-condition window (EMA and MRI
    # textgrids can differ slightly).
    max_dur = max(t1 - t0 for (t0, t1) in windows.values())
    panel_order = [
        ("orig_ema", axes[0, 0]),
        ("orig_mri", axes[0, 1]),
        ("meta", axes[1, 0]),
        ("nvidia", axes[1, 1]),
    ]
    per_panel_counts: Dict[str, int] = {}
    for i, (cond, ax) in enumerate(panel_order):
        wav, _, tracks = paths_for(cond, spk, start, end)
        t0, t1 = windows[cond]
        n_pts = _plot_panel(
            ax,
            wav,
            tracks,
            t0,
            t1,
            title=COND_TITLES[cond],
            show_ylabel=(i % 2 == 0),
            show_legend=(i == 0),
        )
        ax.set_xlim(0.0, max_dur)
        per_panel_counts[cond] = n_pts

    fig.suptitle(
        f'"{sentence_text.capitalize()}"',
        fontsize=15,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rec_stem = f"{spk.lower()}_{start}_{end}"
    out_path = out_dir / f"diagnostic_{spk}_{rec_stem}_sent{sid:03d}.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"[plot] {spk} rec={start}-{end} sent={sid:03d} -> {out_path.name} "
        f"(formant points per panel: " +
        ", ".join(f"{c}={per_panel_counts[c]}" for c in CONDITIONS) + ")",
        flush=True,
    )
    return out_path


# ---------- selection driver ----------


def pick_utterances_per_speaker(
    sentences: Dict[int, List[str]],
    seed: int,
    n_per_spk: int = 3,
) -> List[Tuple[str, str, str, int, str]]:
    """Return up to ``n_per_spk`` ``(spk, start, end, sid, sentence_text)``
    tuples per speaker. Iterates shuffled recordings then shuffled sentence
    ids within each, skipping sids that can't be located in both EMA and MRI
    textgrids. Picks have distinct sids per speaker."""
    rng = random.Random(seed)
    picks: List[Tuple[str, str, str, int, str]] = []
    for spk in SPKS:
        cands = candidate_recordings(spk)
        if not cands:
            print(f"[warn] no complete recordings for {spk}; skipping",
                  file=sys.stderr)
            continue
        rng.shuffle(cands)
        spk_picks: List[Tuple[str, str, str, int, str]] = []
        used_sids: set = set()
        for start, end in cands:
            if len(spk_picks) >= n_per_spk:
                break
            sid_range = list(range(int(start), int(end) + 1))
            rng.shuffle(sid_range)
            # cache textgrids once per recording
            _, tg_ema, _ = paths_for("orig_ema", spk, start, end)
            _, tg_mri, _ = paths_for("orig_mri", spk, start, end)
            try:
                words_ema = load_words_tier(tg_ema)
                words_mri = load_words_tier(tg_mri)
            except Exception as e:
                print(f"[warn] tg read fail for {spk} {start}-{end}: {e}",
                      file=sys.stderr)
                continue
            for sid in sid_range:
                if sid in used_sids or sid not in sentences:
                    continue
                tokens = sentences[sid]
                if find_sentence_window(words_ema, tokens) is None:
                    continue
                if find_sentence_window(words_mri, tokens) is None:
                    continue
                spk_picks.append((spk, start, end, sid, " ".join(tokens)))
                used_sids.add(sid)
                break  # one sentence per recording
        if not spk_picks:
            print(f"[warn] no matchable sentence for {spk}", file=sys.stderr)
            continue
        if len(spk_picks) < n_per_spk:
            print(f"[warn] only {len(spk_picks)}/{n_per_spk} picks for {spk}",
                  file=sys.stderr)
        picks.extend(spk_picks)
    return picks


# ---------- CLI ----------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for utterance/sentence selection (default 0)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default {DEFAULT_OUT_DIR})")
    p.add_argument("--sentences-file", type=Path, default=SENTENCES_FILE,
                   help="Path to list_of_sentences.txt")
    p.add_argument("--n-per-speaker", type=int, default=3,
                   help="Number of utterances to plot per speaker (default 3)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sentences = load_sentences(args.sentences_file)
    print(f"[load] {len(sentences)} sentences from {args.sentences_file}")
    # wipe any prior diagnostic PDFs so the output directory only reflects
    # the current run
    if args.out_dir.is_dir():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    picks = pick_utterances_per_speaker(sentences, args.seed, args.n_per_speaker)
    if not picks:
        print("[err] nothing to plot", file=sys.stderr)
        return 1
    for spk, start, end, sid, text in picks:
        try:
            render_utterance_pdf(spk, start, end, sid, text, args.out_dir)
        except Exception as e:
            print(f"[err] render fail {spk} {start}-{end} sent{sid}: {e}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())