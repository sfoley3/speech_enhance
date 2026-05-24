"""
Base utilities for the formant ABX evaluation.

A small, self-contained reimplementation of the pieces of the ABX
discrimination task we need -- no ABXpy / h5py / h5features dependency.  Vowel
formant tracks loaded by ``calc_rmse.load_condition_tracks`` are scored in
memory.

ABX task design
---------------
ON = ``phoneme`` (vowel identity)   -- A and X are the same vowel, B differs.
BY = ``speaker``                    -- every triplet stays within one speaker.
No ACROSS axis.

A triplet (A, B, X) scores 1 if X is closer to A than to B, 0 if closer to B,
0.5 on a tie.  Chance = 0.5.

Distance
--------
``abx_distance`` == length-normalized DTW with a symmetrized, frame-normalized
Kullback-Leibler frame cost ("DTW + framewise KL on normalized formants").  Per
frame the 3 formant values (F1_s, F2_s, F3_s) are renormalized to a 3-bin
distribution; the (Tx, Ty) pairwise symmetric-KL matrix is the DTW local cost,
and the accumulated path cost is divided by the warping-path length so vowels
of different durations stay comparable.  The DTW recurrence (steps (1,1),
(1,0), (0,1), unit weights) matches ABXpy's.

Sampling
--------
A *cell* is one (speaker, A/X-vowel, B-vowel).  ``make_triplets`` caps the
number of triplets per cell at ``max_per_cell`` (sampled without replacement,
seeded).  With identical item structure across conditions (as in the three LSS
conditions) the same seed yields the same triplets, keeping the cross-condition
comparison paired.

Result helpers: ``bootstrap_ci``, ``paired_condition_tests`` (Wilcoxon + Holm).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import librosa

# Reuse the track loader / VowelTrack / formant columns from the sibling
# calc_rmse.py (one directory up) so the ABX side keys vowels exactly the same
# way the RMSE analysis does (allophone->IPA mapping, per-vowel grouping, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calc_rmse import (  # noqa: E402
    VowelTrack,
    load_condition_tracks,
    FORMANTS,
)

# ---------- config ----------

TIMIT_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT")
LSS_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC_LSS")
DEFAULT_OUT_DIR = Path("/scratch1/seanfole/speech_enhance/src/formants/abx")

REFERENCE = "orig_ema"                       # USC-TIMIT clean EMA (baseline)
COMPARATORS = ["orig_mri", "meta", "nvidia"]  # USC-LSS noisy + two enhanced
TIMIT_SPKS = ["F1", "F5", "M1", "M3"]

COND_LABELS = {
    "orig_ema": "EMA",
    "orig_mri": "Orig MRI",
    "meta": "Denoiser",
    "nvidia": "RE-USE",
}
COND_COLORS = {
    "orig_ema": "#bdbdbd",
    "orig_mri": "#7f7f7f",
    "meta": "#4c72b0",
    "nvidia": "#dd8452",
}

CHANCE = 0.5  # 2-alternative ABX


# ---------- recording filters ----------
#
# Restrict the analysis to the sentence set shared by both corpora by dropping
# the trailing file in each dataset:
#   * LSS  -- keep files usc_s1_1 .. usc_s1_37, drop usc_s1_38 (Rainbow Passage).
#   * TIMIT-- keep sentence ids <= 455, dropping the last range file per speaker
#     (usctimit_*_456_460), which lies beyond the shared inventory.
# Both operate on the modality-stripped ``recording_key`` produced by
# ``calc_rmse._normalize_recording``.

_FILE_NUM_RE = re.compile(r"(\d+)\Z")                 # trailing integer
_STIM_RE = re.compile(r"(?<!\d)(\d+)_(\d+)\Z")        # trailing "<a>_<b>" range


def _file_number(rec_key: str) -> int | None:
    """Trailing file number, e.g. 'usc_s1_20' -> 20."""
    m = _FILE_NUM_RE.search(rec_key)
    return int(m.group(1)) if m else None


def _max_sentence_id(rec_key: str) -> int | None:
    """Highest sentence id in a TIMIT stimulus range, e.g.
    'usctimit_f1_456_460' -> 460.  Falls back to a lone trailing integer."""
    m = _STIM_RE.search(rec_key)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    m2 = _FILE_NUM_RE.search(rec_key)
    return int(m2.group(1)) if m2 else None


def filter_lss_files(
    tracks: Dict[Tuple, VowelTrack], max_file: int = 37,
) -> Dict[Tuple, VowelTrack]:
    """Keep LSS vowels whose recording file number is <= ``max_file``."""
    out = {k: vt for k, vt in tracks.items()
           if (n := _file_number(vt.recording_key)) is not None and n <= max_file}
    dropped = {vt.recording_key for vt in tracks.values()} - {
        vt.recording_key for vt in out.values()}
    print(f"[filter] LSS file<= {max_file}: kept {len(out)}/{len(tracks)} vowels"
          f"; dropped recordings: {sorted(dropped)}")
    return out


def filter_timit_sentences(
    tracks: Dict[Tuple, VowelTrack], max_sentence: int = 455,
) -> Dict[Tuple, VowelTrack]:
    """Keep TIMIT vowels whose stimulus range stays at or below ``max_sentence``
    (drops the last range file per speaker, e.g. usctimit_*_456_460)."""
    out = {k: vt for k, vt in tracks.items()
           if (m := _max_sentence_id(vt.recording_key)) is not None
           and m <= max_sentence}
    n_rec_in = len({vt.recording_key for vt in tracks.values()})
    n_rec_out = len({vt.recording_key for vt in out.values()})
    print(f"[filter] TIMIT sentence<= {max_sentence}: kept {len(out)}/{len(tracks)}"
          f" vowels across {n_rec_out}/{n_rec_in} recordings")
    return out


# ---------- items ----------


def build_items(
    tracks: Dict[Tuple, VowelTrack],
    speakers: set | None = None,
) -> List[dict]:
    """Flatten the track dict into a list of vowel items.

    Each item is ``{id, speaker, phoneme, formants}`` where ``formants`` is the
    token's (T, 3) F1_s/F2_s/F3_s array (sorted by time).  ``id`` is unique per
    token.  Items are emitted in a deterministic order (sorted by id) so that
    triplet sampling is reproducible and identical across conditions that share
    item structure.
    """
    items: List[dict] = []
    seen: set = set()
    for vt in tracks.values():
        if speakers is not None and vt.speaker not in speakers:
            continue
        vid = f"{vt.speaker}__{vt.recording_key}__{vt.vowel_id}"
        if vid in seen:
            continue
        seen.add(vid)
        t = np.asarray(vt.rel_time, dtype=np.float64)
        feat = np.asarray(vt.formants, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        items.append({
            "id": vid,
            "speaker": vt.speaker,
            "phoneme": vt.phoneme,
            "formants": feat[order],
        })
    items.sort(key=lambda it: it["id"])
    return items


# ---------- distance: symmetric KL + length-normalized DTW ----------


def _normalize_frames(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Renormalize each frame (row) to a probability distribution.

    The 3 formant values per frame are floored at ``eps`` and divided by their
    sum, turning each (T, 3) track into T 3-bin distributions for the KL frame
    cost.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, eps)
    return x / x.sum(axis=1, keepdims=True)


def _sym_kl_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise symmetric KL between the frames of two tracks.

    Returns a (Tx, Ty) matrix where entry (i, j) is
    ``0.5 * (KL(x_i || y_j) + KL(y_j || x_i))`` over frame-normalized
    distributions.  Vectorized: for normalized p, q,
    ``KL(p||q) = sum_d p_d (log p_d - log q_d)``.
    """
    x = _normalize_frames(x)
    y = _normalize_frames(y)
    lx = np.log(x)
    ly = np.log(y)
    xlx = (x * lx).sum(axis=1)          # (Tx,)
    yly = (y * ly).sum(axis=1)          # (Ty,)
    kl_xy = xlx[:, None] - x @ ly.T     # (Tx, Ty): KL(x_i || y_j)
    kl_yx = yly[None, :] - lx @ y.T     # (Tx, Ty): KL(y_j || x_i)
    return 0.5 * (kl_xy + kl_yx)


def abx_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    """Length-normalized DTW distance between two (T, 3) formant tracks.

    The local cost is the symmetric-KL frame matrix; DTW uses the default
    librosa recurrence (steps (1,1), (1,0), (0,1), unit weights, matching
    ABXpy), and the accumulated cost ``D[-1, -1]`` is divided by the
    warping-path length so different vowel durations stay comparable.
    """
    C = _sym_kl_matrix(fa, fb)
    D, wp = librosa.sequence.dtw(C=C, subseq=False, backtrack=True)
    return float(D[-1, -1] / len(wp))


# ---------- triplet formation ----------


def make_triplets(
    items: List[dict],
    max_per_cell: int | None = None,
    seed: int = 0,
) -> List[Tuple[str, str, str, int, int, int]]:
    """Form ABX triplets within each speaker.

    For each speaker, for each phoneme ``p`` with >= 2 tokens (so A != X is
    possible) used as the A/X vowel, and each other phoneme ``q`` (the B vowel),
    the *cell* (speaker, p, q) contains every ordered (X, A, B) with X, A drawn
    from ``p`` (X != A) and B drawn from ``q``.  When a cell has more than
    ``max_per_cell`` such triplets they are sub-sampled without replacement.

    Returns a list of ``(speaker, p, q, xi, ai, bi)`` where xi/ai/bi index
    ``items``.  Cells are visited in sorted order and token indices are sorted,
    so the output is deterministic and identical across conditions that share
    item structure (when ``seed`` matches).
    """
    rng = np.random.default_rng(seed)

    # speaker -> phoneme -> sorted list of item indices
    by_spk: Dict[str, Dict[str, List[int]]] = {}
    for idx, it in enumerate(items):
        by_spk.setdefault(it["speaker"], {}).setdefault(it["phoneme"], []).append(idx)

    triplets: List[Tuple[str, str, str, int, int, int]] = []
    for spk in sorted(by_spk):
        phon = by_spk[spk]
        phonemes = sorted(phon)
        for p in phonemes:
            ax = phon[p]
            if len(ax) < 2:
                continue
            for q in phonemes:
                if q == p:
                    continue
                b = phon[q]
                n_ax, n_b = len(ax), len(b)
                total = n_ax * (n_ax - 1) * n_b
                if total == 0:
                    continue
                if max_per_cell is None or total <= max_per_cell:
                    for xi in ax:
                        for ai in ax:
                            if ai == xi:
                                continue
                            for bi in b:
                                triplets.append((spk, p, q, xi, ai, bi))
                else:
                    # rejection-sample distinct (X, A, B) triplets
                    picked: set = set()
                    while len(picked) < max_per_cell:
                        xi = ax[rng.integers(n_ax)]
                        ai = ax[rng.integers(n_ax)]
                        if ai == xi:
                            continue
                        bi = b[rng.integers(n_b)]
                        picked.add((xi, ai, bi))
                    for xi, ai, bi in sorted(picked):
                        triplets.append((spk, p, q, xi, ai, bi))
    return triplets


def score_triplets(
    items: List[dict],
    triplets: List[Tuple[str, str, str, int, int, int]],
) -> np.ndarray:
    """Per-triplet ABX flag in {0, 0.5, 1}.

    1 if d(X, A) < d(X, B), 0 if greater, 0.5 on a tie.  Pairwise distances are
    memoized (the symmetric distance is cached on the sorted index pair) since
    the same token pair recurs across many triplets.
    """
    cache: Dict[Tuple[int, int], float] = {}

    def dist(i: int, j: int) -> float:
        key = (i, j) if i <= j else (j, i)
        d = cache.get(key)
        if d is None:
            d = abx_distance(items[i]["formants"], items[j]["formants"])
            cache[key] = d
        return d

    flags = np.empty(len(triplets), dtype=np.float64)
    for k, (_spk, _p, _q, xi, ai, bi) in enumerate(triplets):
        d_xa = dist(xi, ai)
        d_xb = dist(xi, bi)
        flags[k] = 1.0 if d_xa < d_xb else (0.0 if d_xa > d_xb else 0.5)
    return flags


def aggregate_cells(
    triplets: List[Tuple[str, str, str, int, int, int]],
    flags: np.ndarray,
) -> pd.DataFrame:
    """Collapse per-triplet flags into per-cell accuracy.

    Returns ``DataFrame[speaker, phoneme_ax, phoneme_b, score, n]`` where
    ``score`` is the mean flag in the cell and ``n`` its triplet count.
    """
    df = pd.DataFrame(
        [(s, p, q) for (s, p, q, _xi, _ai, _bi) in triplets],
        columns=["speaker", "phoneme_ax", "phoneme_b"],
    )
    df["flag"] = np.asarray(flags, dtype=np.float64)
    cells = (
        df.groupby(["speaker", "phoneme_ax", "phoneme_b"], sort=True)["flag"]
        .agg(score="mean", n="size")
        .reset_index()
    )
    return cells


def run_abx(
    tracks: Dict[Tuple, VowelTrack],
    speakers: set | None = None,
    max_per_cell: int | None = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """End-to-end ABX: tracks -> (per-cell accuracy table, per-triplet flags).

    ``cells`` is the aggregated table (speaker, phoneme_ax, phoneme_b, score,
    n); ``flags`` is the raw per-triplet {0, 0.5, 1} array for bootstrap CIs.
    """
    items = build_items(tracks, speakers=speakers)
    triplets = make_triplets(items, max_per_cell=max_per_cell, seed=seed)
    flags = score_triplets(items, triplets)
    cells = aggregate_cells(triplets, flags)
    return cells, flags


def run_abx_paired(
    tracks_by_cond: Dict[str, Dict[Tuple, VowelTrack]],
    speakers: set | None = None,
    max_per_cell: int | None = None,
    seed: int = 0,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Score one shared triplet set against several conditions (truly paired).

    Triplets are formed *once* on the token set shared by every condition
    (intersection of item ids), then scored against each condition's own
    formants -- so every condition is evaluated on exactly the same (A, B, X)
    triplets and any accuracy difference is attributable to the audio, not to a
    different triplet sample.  This removes the fragile assumption that the
    per-condition item lists happen to be index-aligned.

    Tokens missing from any condition (e.g. dropped by formant-extraction
    failures) are excluded from all conditions.  The canonical speaker/phoneme
    labels are taken from the first condition.

    Returns ``(cells_by_cond, flags_by_cond)`` -- dicts keyed by condition name,
    each value matching :func:`run_abx`'s ``cells`` / ``flags`` outputs.
    """
    conds = list(tracks_by_cond)
    if not conds:
        return {}, {}

    items_by_cond = {
        c: build_items(tracks_by_cond[c], speakers=speakers) for c in conds
    }
    id_sets = [{it["id"] for it in items_by_cond[c]} for c in conds]
    shared = set.intersection(*id_sets)

    # canonical shared items (build_items is sorted by id; filtering keeps order)
    canonical = [it for it in items_by_cond[conds[0]] if it["id"] in shared]
    for c in conds:
        dropped = len(items_by_cond[c]) - len(shared)
        print(f"[paired] {c}: {len(items_by_cond[c])} items, "
              f"{dropped} not shared")
    print(f"[paired] shared token set = {len(canonical)} items")

    # one triplet list on the shared structure
    triplets = make_triplets(canonical, max_per_cell=max_per_cell, seed=seed)

    cells_by_cond: Dict[str, pd.DataFrame] = {}
    flags_by_cond: Dict[str, np.ndarray] = {}
    for c in conds:
        by_id = {it["id"]: it["formants"] for it in items_by_cond[c]}
        cond_items = [{"formants": by_id[it["id"]]} for it in canonical]
        flags = score_triplets(cond_items, triplets)
        flags_by_cond[c] = flags
        cells_by_cond[c] = aggregate_cells(triplets, flags)
    return cells_by_cond, flags_by_cond


# ---------- result helpers ----------


def bootstrap_ci(
    flags: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Mean accuracy and percentile bootstrap CI over per-triplet flags."""
    flags = np.asarray(flags, dtype=np.float64)
    n = len(flags)
    if n == 0:
        nan = float("nan")
        return nan, (nan, nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = flags[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2.0))
    return float(flags.mean()), (lo, hi)


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone)."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - rank))
    # enforce monotonic non-decreasing along the sorted order
    running = np.maximum.accumulate(adj[order])
    for k, idx in enumerate(order):
        adj[idx] = running[k]
    return adj


def paired_condition_tests(
    cells_by_cond: Dict[str, pd.DataFrame],
    conditions: List[str],
    alternative: str = "two-sided",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the per-cell accuracies across conditions and run Wilcoxon + Holm.

    The conditions share the same item structure, so cells (identified by
    speaker, phoneme_ax, phoneme_b) line up one-to-one.  We inner-merge on those
    keys, then run a pairwise Wilcoxon signed-rank test on the per-cell ``score``
    for each condition pair, Holm-correcting the p-values.

    Returns (merged_wide, tests) where ``merged_wide`` has one row per cell with
    a column per condition, and ``tests`` is a tidy table of the pairwise
    results (W, p, p_holm, median difference).
    """
    from scipy import stats

    key_cols = ["speaker", "phoneme_ax", "phoneme_b"]

    merged: pd.DataFrame | None = None
    for cond in conditions:
        d = cells_by_cond[cond][key_cols + ["score"]].rename(columns={"score": cond})
        merged = d if merged is None else merged.merge(d, on=key_cols, how="inner")
    assert merged is not None

    pairs = [(a, b) for i, a in enumerate(conditions) for b in conditions[i + 1:]]
    recs = []
    for a, b in pairs:
        x = merged[a].to_numpy(dtype=float)
        y = merged[b].to_numpy(dtype=float)
        try:
            W, p = stats.wilcoxon(
                x, y, zero_method="wilcox", alternative=alternative,
            )
        except ValueError:
            W, p = float("nan"), float("nan")
        recs.append({
            "cond_a": a,
            "cond_b": b,
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "median_diff": float(np.median(x - y)),
            "W": float(W),
            "p": float(p),
            "n_cells": int(len(x)),
        })
    tests = pd.DataFrame(recs)
    if len(tests):
        valid = tests["p"].notna().to_numpy()
        holm = np.full(len(tests), np.nan)
        if valid.any():
            holm[valid] = _holm(tests.loc[valid, "p"].to_numpy())
        tests["p_holm"] = holm
    return merged, tests
