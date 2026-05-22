"""
RMSE between formant tracks: EMA (clean) vs. {orig_mri, meta, nvidia}.

Pipeline
--------
1. Load fave-extract `*_tracks.csv` files (one row every ~2 ms per vowel) for
   the four conditions:  orig_ema, orig_mri, meta, nvidia.
2. Group rows into per-vowel tracks keyed by (recording, vowel index).
   FAVE's `id` column is a within-file vowel index. The same textgrid prompts
   are spoken under both modalities, so vowel order/labels match — we still
   verify on (word, label) before pairing.
3. For each matched vowel, DTW-align the comparator track (orig_mri / meta /
   nvidia) to the EMA reference and compute RMSE per formant.  We use the
   DCT-smoothed tracks F1_s, F2_s, F3_s as specified.
4. Pool RMSE across vowels and speakers; produce a boxplot with F1/F2/F3 on x
   and three boxes (orig_mri, meta, nvidia) per formant.
5. Run an omnibus test (Friedman, paired/non-parametric across the three
   conditions per vowel) and post-hoc Wilcoxon signed-rank with Holm
   correction.  Also reports a Shapiro-Wilk normality check; if data look
   normal, repeated-measures ANOVA + paired t-tests are reported alongside.
6. Save boxplot PDF and a CSV of per-vowel RMSE values for downstream use.

Usage
-----
    python formant_rmse.py                # default paths (cluster layout)
    python formant_rmse.py --results-dir /path/to/fave_results/USC-TIMIT \
                           --out-dir     /path/to/output

Assumes layout produced by setup_fave.py:
    {RESULTS_DIR}/{condition}/{spk}/*_tracks.csv
or a flat:
    {RESULTS_DIR}/{condition}/*_tracks.csv
Both are handled.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import librosa 

import matplotlib.pyplot as plt  

from scipy import stats 
from tqdm import tqdm 

plt.rc('font', size=20)
plt.rc('legend', fontsize=15)  

# ---------- config ----------

DEFAULT_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT")
DEFAULT_OUT_DIR = Path("/scratch1/seanfole/speech_enhance/src/formants")

REFERENCE = "orig_ema"
COMPARATORS = ["orig_mri", "meta", "nvidia"]
CONDITIONS = [REFERENCE, *COMPARATORS]

FORMANTS = ["F1_s", "F2_s", "F3_s"]
FORMANT_LABELS = {"F1_s": "F1", "F2_s": "F2", "F3_s": "F3"}

SPKS = ["F1", "F5", "M1", "M3"]

# Match the recording key across modalities by stripping the modality token
# e.g. "usctimit_mri_m3_446_450" -> "usctimit_m3_446_450"
MODALITY_RE = re.compile(r"_(mri|ema)_", flags=re.IGNORECASE)

# Allophone (FAVE label) -> phoneme (IPA). Collapse modality-specific
# allophonic variants so EMA/MRI/denoised tracks key on the same phoneme
# before matching. Anything not in this map is treated as a non-vowel and
# dropped from the analysis (covers "sil", "sp", stops, fricatives, ...).
ALLOPHONE_TO_IPA: Dict[str, str] = {
    "ay":  "a\u026a",
    "ay0": "a\u026a",
    "ae":  "\u00e6",
    "@":   "\u0259",
    "\u028c": "\u0259",
    "o":   "\u0251",   # vowel in "pot"
    "oh":  "\u0254",   # open-o
    "ow":  "o\u028a",
    "owf": "o\u028a",
    "i":   "\u026a",
    "iy":  "i",
    "iyf": "i",
    "e":   "\u025b",
    "ey":  "e\u026a",
    "eyf": "e\u026a",
    "u":   "\u028a",
    "uw":  "u",
    "uwr": "u",
    "oy":  "\u0254\u026a",
    "aw":  "a\u028a",
    "tuw": "u",
    "ahr": "\u0251",
    "*hr": "\u0259r",
    "iyr": "i",
    "ah":  "\u0259",
    "a":   "\u0251",
}


def map_label(raw: str) -> str | None:
    """Look up a FAVE allophone label in ``ALLOPHONE_TO_IPA``.

    Lookup is case-insensitive and strips any trailing ARPABET stress digit
    (``AE1`` -> ``ae``). Returns ``None`` for unmapped labels so callers can
    drop non-vowel rows (silence, consonants, ...).
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    # exact match first so explicit stress-bearing keys like "ay0" win
    if s in ALLOPHONE_TO_IPA:
        return ALLOPHONE_TO_IPA[s]
    # fall back: strip a single trailing ARPABET stress digit and retry
    if s[-1].isdigit() and len(s) > 1:
        s2 = s[:-1]
        if s2 in ALLOPHONE_TO_IPA:
            return ALLOPHONE_TO_IPA[s2]
    return None


def _norm_word(w: object) -> str:
    return str(w).strip().lower() if w is not None else ""


# ---------- data loading ----------


@dataclass
class VowelTrack:
    """One vowel: time-series of (F1_s, F2_s, F3_s) sampled ~ every 2 ms."""
    recording_key: str          # modality-stripped recording id
    speaker: str
    vowel_id: str               # FAVE's `id` within the recording
    word: str                   # normalized (lower/stripped) word
    pre_word: str               # normalized preceding word
    fol_word: str               # normalized following word
    raw_label: str              # original FAVE label (e.g. "ae")
    phoneme: str                # mapped IPA phoneme (e.g. "\u00e6")
    in_word_index: int          # 0-based occurrence within the word triplet
    rel_time: np.ndarray        # shape (T,)
    formants: np.ndarray        # shape (T, 3) for F1_s F2_s F3_s

    @property
    def match_key(self) -> Tuple[str, str, str, str, str, str, int]:
        return (
            self.speaker,
            self.recording_key,
            self.pre_word,
            self.word,
            self.fol_word,
            self.phoneme,
            self.in_word_index,
        )


def _normalize_recording(file_name: str) -> str:
    """Strip the modality token so EMA and MRI versions share a key."""
    stem = Path(str(file_name)).stem
    # tracks file convention: usctimit_<mod>_<spk>_<s>_<e>_tracks
    stem = stem.replace("_tracks", "")
    return MODALITY_RE.sub("_", stem, count=1).lower()


def _iter_tracks_csvs(results_dir: Path, condition: str) -> Iterable[Tuple[str, Path]]:
    """Yield (speaker, csv_path).  Supports {cond}/{spk}/*.csv and {cond}/*.csv."""
    cond_dir = results_dir / condition
    if not cond_dir.is_dir():
        print(f"[warn] missing condition dir: {cond_dir}", file=sys.stderr)
        return
    # speaker subdirs
    spk_dirs = [d for d in cond_dir.iterdir() if d.is_dir()]
    if spk_dirs:
        for d in sorted(spk_dirs):
            for p in sorted(d.glob("*_tracks.csv")):
                yield d.name, p
    else:
        for p in sorted(cond_dir.glob("*_tracks.csv")):
            # try to infer speaker from filename: usctimit_<mod>_<spk>_<...>
            m = re.match(r"usctimit_[a-z]+_([a-zA-Z0-9]+)_", p.stem)
            spk = m.group(1).upper() if m else "?"
            yield spk, p


def load_condition_tracks(
    results_dir: Path, condition: str
) -> Dict[Tuple[str, str, str, str, str, str, int], VowelTrack]:
    """Load all vowel tracks for one condition.

    Returns a dict keyed by
    ``(speaker, rec_key, pre_word, word, fol_word, phoneme, in_word_index)``
    -> ``VowelTrack``. Rows whose label does not map to an IPA phoneme via
    ``ALLOPHONE_TO_IPA`` (silence, consonants, ...) are dropped.
    """
    needed_cols = {
        "file_name", "id", "word", "pre_word", "fol_word",
        "label", "rel_time", *FORMANTS,
    }
    out: Dict[Tuple[str, str, str, str, str, str, int], VowelTrack] = {}
    n_unmapped = 0
    n_short = 0

    for spk, csv_path in _iter_tracks_csvs(results_dir, condition):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[err] reading {csv_path}: {e}", file=sys.stderr)
            continue
        missing = needed_cols - set(df.columns)
        if missing:
            print(f"[warn] {csv_path.name} missing cols {missing}; skipping",
                  file=sys.stderr)
            continue

        df = df[list(needed_cols)].copy()
        for c in FORMANTS + ["rel_time"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=FORMANTS + ["rel_time"])
        df["rec_key"] = df["file_name"].map(_normalize_recording)
        df["phoneme"] = df["label"].map(map_label)
        unmapped_mask = df["phoneme"].isna()
        # count unique (rec_key, id) units dropped, not raw frames
        n_unmapped += df.loc[unmapped_mask, ["rec_key", "id"]].drop_duplicates().shape[0]
        df = df[~unmapped_mask].copy()
        df["word_n"] = df["word"].map(_norm_word)
        df["pre_word_n"] = df["pre_word"].map(_norm_word)
        df["fol_word_n"] = df["fol_word"].map(_norm_word)

        # build per-vowel tracks (one row per FAVE `id` within a recording)
        per_vowel: List[VowelTrack] = []
        for (rec_key, vid), grp in df.groupby(["rec_key", "id"], sort=False):
            grp = grp.sort_values("rel_time")
            if len(grp) < 3:
                n_short += 1
                continue
            per_vowel.append(VowelTrack(
                recording_key=str(rec_key),
                speaker=spk,
                vowel_id=str(vid),
                word=str(grp["word_n"].iloc[0]),
                pre_word=str(grp["pre_word_n"].iloc[0]),
                fol_word=str(grp["fol_word_n"].iloc[0]),
                raw_label=str(grp["label"].iloc[0]),
                phoneme=str(grp["phoneme"].iloc[0]),
                in_word_index=0,  # filled in below
                rel_time=grp["rel_time"].to_numpy(dtype=float),
                formants=grp[FORMANTS].to_numpy(dtype=float),
            ))

        # within each (rec_key, pre_word, word, fol_word) bucket, assign an
        # ordinal index by start time so repeated vowels pair 1st<->1st etc.
        buckets: Dict[Tuple[str, str, str, str], List[VowelTrack]] = {}
        for t in per_vowel:
            buckets.setdefault(
                (t.recording_key, t.pre_word, t.word, t.fol_word), []
            ).append(t)
        for tracks in buckets.values():
            tracks.sort(key=lambda t: (float(t.rel_time[0]), t.vowel_id))
            # group by mapped phoneme so the ordinal is per-phoneme within word
            counters: Dict[str, int] = {}
            for t in tracks:
                idx = counters.get(t.phoneme, 0)
                t.in_word_index = idx
                counters[t.phoneme] = idx + 1
                out[t.match_key] = t

    print(f"[load] {condition}: {len(out)} vowels across "
          f"{len({k[0] for k in out})} speakers "
          f"(dropped {n_unmapped} unmapped, {n_short} too-short)")
    return out


# ---------- DTW + RMSE ----------


def _rescale_for_dtw(x: np.ndarray, mode: str) -> np.ndarray:
    """
    Rescale formants for the DTW *cost only* — RMSE is always reported on raw Hz.

    Modes
    -----
    "zscore" : per-formant z-score (default; pure scale balance)
    "log"    : natural log (linguistically motivated, near-Mel below ~1 kHz)
    "none"   : raw Hz — F2 will dominate the alignment
    """
    if mode == "none":
        return x
    if mode == "log":
        return np.log(np.maximum(x, 1.0))
    # zscore
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def dtw_path(ref: np.ndarray, hyp: np.ndarray, scale_mode: str = "zscore"
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joint multivariate DTW across F1/F2/F3 using librosa.sequence.dtw.
    `ref`, `hyp`: (T, 3) in raw Hz.  Returns aligned index arrays
    (i_ref, j_hyp) in time order (start -> end of vowel).
    """
    a = _rescale_for_dtw(ref, scale_mode)
    b = _rescale_for_dtw(hyp, scale_mode)
    # librosa expects (features, time)
    _, wp = librosa.sequence.dtw(
        X=a.T, Y=b.T,
        metric="euclidean",
        subseq=False,
        backtrack=True,
    )
    # wp is returned in reverse (end -> start); flip so it's start -> end
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1]


def vowel_rmse(ref: VowelTrack, hyp: VowelTrack, scale_mode: str = "zscore"
               ) -> np.ndarray:
    """Return RMSE per formant (shape (3,)) after DTW-aligning hyp to ref."""
    i_ref, j_hyp = dtw_path(ref.formants, hyp.formants, scale_mode=scale_mode)
    a = ref.formants[i_ref]   # raw Hz
    b = hyp.formants[j_hyp]   # raw Hz
    err = a - b
    return np.sqrt(np.mean(err * err, axis=0))


# ---------- pairing + computation ----------


def compute_rmse_table(
    ref_tracks: Dict[Tuple[str, str, str, str, str, str, int], VowelTrack],
    comp_tracks: Dict[str, Dict[Tuple[str, str, str, str, str, str, int], VowelTrack]],
    scale_mode: str = "zscore",
    diagnostic_dir: Path | None = None,
    diagnostic_sample_n: int = 100,
    diagnostic_seed: int = 0,
) -> pd.DataFrame:
    """
    For every match key present in the reference, look up the same key in
    each comparator condition. Keys are
    ``(speaker, rec_key, pre_word, word, fol_word, phoneme, in_word_index)``
    so files must agree on speaker+stimulus, words must agree on the
    (pre_word, word, fol_word) triplet, and phonemes must agree after
    allophone->IPA mapping; repeated phonemes within a word pair by ordinal.
    Vowels missing from any comparator are skipped so post-hoc tests stay
    paired. Returns a long-form RMSE table.

    If ``diagnostic_dir`` is supplied, also write two JSON files there:
    ``rmse_pairing_included.json`` (raw included count + up to
    ``diagnostic_sample_n`` paired reference vowels) and
    ``rmse_pairing_excluded.json`` (raw excluded count + up to
    ``diagnostic_sample_n`` reference vowels dropped because >=1 comparator
    was missing the key, each annotated with a mismatch reason). Since
    orig_mri / meta / nvidia share audio + textgrids, the diagnostic
    reports only the orig_mri side as the MRI representative.
    """
    # Silence / non-vowel labels we never want appearing inside a word's
    # phone string. Anything not in ALLOPHONE_TO_IPA already drops out
    # before tracks are built, but be defensive.
    SILENCE_LABELS = {"sil", "sp", "spn", ""}

    def _is_silence(raw_label: str, phoneme: str) -> bool:
        return (
            not phoneme
            or not raw_label
            or raw_label.lower() in SILENCE_LABELS
            or phoneme.lower() in SILENCE_LABELS
        )

    def _reconstruct_sentence(
        tracks: Dict[Tuple[str, str, str, str, str, str, int], VowelTrack],
    ) -> Dict[Tuple[str, str], str]:
        """(speaker, rec_key) -> single sentence string built by walking
        the recording's vowels in time order and de-duplicating
        consecutive pre_word/word/fol_word tokens."""
        by_rec: Dict[Tuple[str, str], List[VowelTrack]] = {}
        for vt in tracks.values():
            by_rec.setdefault((vt.speaker, vt.recording_key), []).append(vt)
        out: Dict[Tuple[str, str], str] = {}
        for rec, vts in by_rec.items():
            vts.sort(key=lambda v: (
                float(v.rel_time[0]) if len(v.rel_time) else 0.0,
                v.vowel_id,
            ))
            toks: List[str] = []
            for v in vts:
                for w in (v.pre_word, v.word, v.fol_word):
                    if not w:
                        continue
                    if toks and toks[-1] == w:
                        continue
                    toks.append(w)
            out[rec] = " ".join(toks)
        return out

    def _word_phone_strings(
        tracks: Dict[Tuple[str, str, str, str, str, str, int], VowelTrack],
    ) -> Dict[Tuple[str, str, str, str, str], str]:
        """(speaker, rec_key, pre_word, word, fol_word) -> IPA string of
        the word's vowel phones in in_word_index order, silences filtered."""
        bucket: Dict[Tuple[str, str, str, str, str], List[Tuple[int, str, str]]] = {}
        for vt in tracks.values():
            slot = (vt.speaker, vt.recording_key,
                    vt.pre_word, vt.word, vt.fol_word)
            bucket.setdefault(slot, []).append(
                (vt.in_word_index, vt.raw_label, vt.phoneme)
            )
        out: Dict[Tuple[str, str, str, str, str], str] = {}
        for slot, entries in bucket.items():
            entries.sort()
            out[slot] = "".join(
                ph for _, lab, ph in entries if not _is_silence(lab, ph)
            )
        return out

    # MRI representative: orig_mri / meta / nvidia share audio + textgrids.
    mri_repr = "orig_mri" if "orig_mri" in comp_tracks else next(iter(comp_tracks))

    # MRI-side word-slot candidate index for excluded-reason classification.
    word_slot_index: Dict[
        Tuple[str, str, str, str, str],
        List[Tuple[str, str, int, str]],
    ] = {}
    for vt in comp_tracks[mri_repr].values():
        slot = (vt.speaker, vt.recording_key,
                vt.pre_word, vt.word, vt.fol_word)
        word_slot_index.setdefault(slot, []).append(
            (vt.phoneme, vt.raw_label, vt.in_word_index, vt.vowel_id)
        )
    for slot in word_slot_index:
        word_slot_index[slot].sort(key=lambda t: (t[2], t[3]))

    if diagnostic_dir is not None:
        ema_sentences = _reconstruct_sentence(ref_tracks)
        mri_sentences = _reconstruct_sentence(comp_tracks[mri_repr])
        ema_word_phones = _word_phone_strings(ref_tracks)
        mri_word_phones = _word_phone_strings(comp_tracks[mri_repr])

    rows = []
    included_diag: List[Dict[str, object]] = []
    excluded_diag: List[Dict[str, object]] = []
    skipped = {"missing_in_comparator": 0}

    for key, ref in ref_tracks.items():
        missing = [c for c, d in comp_tracks.items() if key not in d]
        if missing:
            skipped["missing_in_comparator"] += 1
            if diagnostic_dir is not None:
                slot = (ref.speaker, ref.recording_key,
                        ref.pre_word, ref.word, ref.fol_word)
                alts = word_slot_index.get(slot, [])
                if not alts:
                    reason = "different_word_context"
                elif any(a[0] == ref.phoneme for a in alts):
                    reason = "phoneme_count_mismatch"
                else:
                    reason = "different_phoneme"

                mri_word = ref.word if alts else None
                mri_phones = mri_word_phones.get(slot, "") if alts else ""
                excluded_diag.append({
                    "speaker": ref.speaker,
                    "recording": ref.recording_key,
                    "reason": reason,
                    "ema": {
                        "sentence": ema_sentences.get(
                            (ref.speaker, ref.recording_key), ""),
                        "word": ref.word,
                        "phones": ema_word_phones.get(slot, ""),
                        "phoneme": {
                            "label": ref.raw_label,
                            "ipa": ref.phoneme,
                        },
                    },
                    "mri": {
                        "sentence": mri_sentences.get(
                            (ref.speaker, ref.recording_key), ""),
                        "word": mri_word,
                        "phones": mri_phones,
                        "phonemes": [
                            {"label": lab, "ipa": phon}
                            for phon, lab, _, _ in alts
                        ],
                    },
                })
            continue
        hyps = {c: comp_tracks[c][key] for c in comp_tracks}

        if diagnostic_dir is not None:
            slot = (ref.speaker, ref.recording_key,
                    ref.pre_word, ref.word, ref.fol_word)
            included_diag.append({
                "speaker": ref.speaker,
                "sentence": ema_sentences.get(
                    (ref.speaker, ref.recording_key), ""),
                "word": ref.word,
                "phones": ema_word_phones.get(slot, ""),
                "phoneme": {
                    "label": ref.raw_label,
                    "ipa": ref.phoneme,
                },
            })

        for cond, hyp in hyps.items():
            rmse = vowel_rmse(ref, hyp, scale_mode=scale_mode)
            for f_col, val in zip(FORMANTS, rmse):
                rows.append({
                    "speaker": ref.speaker,
                    "recording": ref.recording_key,
                    "vowel_id": ref.vowel_id,
                    "pre_word": ref.pre_word,
                    "word": ref.word,
                    "fol_word": ref.fol_word,
                    "raw_label": ref.raw_label,
                    "phoneme": ref.phoneme,
                    "in_word_index": ref.in_word_index,
                    "condition": cond,
                    "formant": FORMANT_LABELS[f_col],
                    "rmse_hz": float(val),
                })

    n_pairs = len(rows) // max(1, (len(comp_tracks) * len(FORMANTS)))
    print(f"[pair] kept {n_pairs} vowels; skipped {skipped}")

    if diagnostic_dir is not None:
        rng = random.Random(diagnostic_seed)

        def _sample(rows_in: List[Dict[str, object]]) -> List[Dict[str, object]]:
            if len(rows_in) <= diagnostic_sample_n:
                return list(rows_in)
            return rng.sample(rows_in, diagnostic_sample_n)

        n_included = len(included_diag)
        n_excluded = len(excluded_diag)
        n_total = n_included + n_excluded

        inc_payload = {
            "counts": {
                "included": n_included,
                "excluded": n_excluded,
                "total": n_total,
            },
            "sample": _sample(included_diag),
        }
        exc_payload = {
            "counts": {
                "included": n_included,
                "excluded": n_excluded,
                "total": n_total,
            },
            "sample": _sample(excluded_diag),
        }

        inc_path = diagnostic_dir / "rmse_pairing_included.json"
        inc_path.write_text(
            json.dumps(inc_payload, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"[diag] included: {len(inc_payload['sample'])}/{n_included}"
              f" -> {inc_path}")

        exc_path = diagnostic_dir / "rmse_pairing_excluded.json"
        exc_path.write_text(
            json.dumps(exc_payload, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"[diag] excluded: {len(exc_payload['sample'])}/{n_excluded}"
              f" -> {exc_path}")

    return pd.DataFrame(rows)


# ---------- stats ----------


def run_stats(df: pd.DataFrame) -> str:
    """
    Per formant: Shapiro-Wilk normality (on residuals per condition), Friedman
    omnibus across the three conditions (paired by vowel), pairwise Wilcoxon
    signed-rank with Holm correction.  If all conditions look normal, also
    report repeated-measures ANOVA + paired t-tests.
    """
    lines: List[str] = []
    conds = COMPARATORS

    for fmt in ["F1", "F2", "F3"]:
        sub = df[df["formant"] == fmt]
        # pivot so each row is a vowel, columns are conditions; drop rows
        # missing any condition.
        wide = sub.pivot_table(
            index=["speaker", "recording", "vowel_id"],
            columns="condition",
            values="rmse_hz",
            aggfunc="first",
        ).dropna(subset=conds)
        if len(wide) < 5:
            lines.append(f"\n=== {fmt}: insufficient paired data (n={len(wide)})")
            continue

        n = len(wide)
        lines.append(f"\n=== {fmt}  (n={n} paired vowels) ===")
        for c in conds:
            x = wide[c].to_numpy()
            lines.append(f"  {c:>9s}: mean={x.mean():7.2f}  median={np.median(x):7.2f}  "
                         f"sd={x.std(ddof=1):7.2f}")

        # normality on each condition
        normal = True
        for c in conds:
            x = wide[c].to_numpy()
            if len(x) >= 5000:
                W, p = stats.shapiro(np.random.choice(x, 5000, replace=False))
            else:
                W, p = stats.shapiro(x)
            lines.append(f"  Shapiro-Wilk {c}: W={W:.3f}  p={p:.3g}"
                         + ("  (normal)" if p > 0.05 else "  (non-normal)"))
            if p <= 0.05:
                normal = False

        # Friedman (non-parametric repeated measures)
        stat_f, p_f = stats.friedmanchisquare(*(wide[c].to_numpy() for c in conds))
        lines.append(f"  Friedman chi2={stat_f:.3f}  p={p_f:.3g}")

        # pairwise Wilcoxon signed-rank, Holm-Bonferroni corrected
        pairs = [(a, b) for i, a in enumerate(conds) for b in conds[i + 1:]]
        raw = []
        for a, b in pairs:
            try:
                W, p = stats.wilcoxon(wide[a], wide[b], zero_method="wilcox",
                                      alternative="two-sided")
            except ValueError as e:
                W, p = float("nan"), float("nan")
                lines.append(f"  Wilcoxon {a} vs {b}: error: {e}")
                continue
            raw.append((a, b, W, p))
        if raw:
            ps = np.array([r[3] for r in raw])
            order = np.argsort(ps)
            holm = np.empty_like(ps)
            for rank, idx in enumerate(order):
                holm[idx] = min(1.0, ps[idx] * (len(ps) - rank))
            # enforce monotonicity
            holm_sorted = np.maximum.accumulate(holm[order])
            for k, idx in enumerate(order):
                holm[idx] = holm_sorted[k]
            for (a, b, W, p), p_adj in zip(raw, holm):
                lines.append(f"  Wilcoxon {a:>9s} vs {b:<9s}: W={W:.1f}  "
                             f"p={p:.3g}  p_holm={p_adj:.3g}")

        # optional parametric companion
        if normal:
            try:
                F, p = stats.f_oneway(*(wide[c].to_numpy() for c in conds))
                # this is between-groups; for RM-ANOVA use a within model:
                # use a simple paired form by differencing — report 1-way for ref
                lines.append(f"  (one-way ANOVA, between, F={F:.3f}  p={p:.3g})")
            except Exception as e:
                lines.append(f"  ANOVA error: {e}")
            for a, b in pairs:
                t, p = stats.ttest_rel(wide[a], wide[b])
                lines.append(f"  paired t  {a:>9s} vs {b:<9s}: t={t:.3f}  p={p:.3g}")

    return "\n".join(lines)


# ---------- plotting ----------


def make_boxplot(df: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    formants = ["F1", "F2", "F3"]
    conds = COMPARATORS
    colors = {"orig_mri": "#bdbdbd", "meta": "#4c72b0", "nvidia": "#dd8452"}

    width = 0.22
    positions = []
    box_data = []
    box_colors = []
    xticks = []
    for i, fmt in enumerate(formants):
        for j, c in enumerate(conds):
            vals = df[(df["formant"] == fmt) & (df["condition"] == c)]["rmse_hz"].to_numpy()
            pos = i + (j - 1) * (width + 0.04)
            positions.append(pos)
            box_data.append(vals)
            box_colors.append(colors[c])
        xticks.append(i)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        notch=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor("black")

    ax.set_xticks(xticks)
    ax.set_xticklabels(formants)
    ax.set_ylabel("RMSE")
    #ax.set_xlabel("Formant")

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[c], edgecolor="black",
                     label={"orig_mri": "orig MRI",
                            "meta": "META denoiser",
                            "nvidia": "NVIDIA reuse"}[c])
               for c in conds]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] wrote {out_pdf}")


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="root containing {condition}/{spk}/*_tracks.csv")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="output dir for the boxplot pdf, rmse csv, stats txt")
    ap.add_argument("--dtw-scale", choices=["zscore", "log", "none"],
                    default="zscore",
                    help="rescaling applied to formants for the DTW cost only "
                         "(RMSE is always reported on raw Hz). new-fave's F*_s "
                         "tracks are DCT-smoothed but NOT normalized, so "
                         "without rescaling F2 dominates the joint alignment.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] results_dir = {args.results_dir}")
    print(f"[cfg] out_dir     = {args.out_dir}")

    ref = load_condition_tracks(args.results_dir, REFERENCE)
    comp = {c: load_condition_tracks(args.results_dir, c) for c in COMPARATORS}

    print(f"[cfg] dtw_scale  = {args.dtw_scale}")
    df = compute_rmse_table(ref, comp, scale_mode=args.dtw_scale,
                            diagnostic_dir=args.out_dir)
    if df.empty:
        sys.exit("[err] no paired vowels found; check matching keys / file layout")

    rmse_csv = args.out_dir / "formant_rmse_per_vowel.csv"
    df.to_csv(rmse_csv, index=False)
    print(f"[save] {rmse_csv}  ({len(df)} rows)")

    # quick summary printed and dumped
    summary = (
        df.groupby(["formant", "condition"])["rmse_hz"]
          .agg(["count", "mean", "median", "std"])
          .reset_index()
    )
    print("\n--- summary (Hz) ---")
    print(summary.to_string(index=False))
    summary.to_csv(args.out_dir / "formant_rmse_summary.csv", index=False)

    stats_txt = run_stats(df)
    print(stats_txt)
    (args.out_dir / "formant_rmse_stats.txt").write_text(stats_txt + "\n")

    make_boxplot(df, args.out_dir / "formant_rmse_boxplot.pdf")


if __name__ == "__main__":
    main()
