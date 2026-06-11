"""
Print audio sampling rate(s) for every (condition, group/speaker) combination
that feeds new-fave -- using the real source audio dirs (the same ones the
fave_work symlink builders point at), before any symlinking.

LSS  (groups dsp/raw):
    {g}_mri : {lss_orig}/audio_{g}
    meta    : {enhanced}/{g}/META_denoiser/USC_LSS_meta
    nvidia  : {enhanced}/{g}/NVIDIA_REUSE/USC_LSS_nvidia
    pase    : {enhanced}/{g}/PASE/USC_LSS_pase

USC-TIMIT (speakers F1/F5/M1/M3; enhanced not group-split):
    orig_mri: {timit_orig}/MRI/Data/{spk}/wav
    orig_ema: {timit_orig}/EMA/Data/{spk}/wav
    meta    : {enhanced}/META_denoiser/USC-TIMIT_denoiser/{spk}/wav
    nvidia  : {enhanced}/NVIDIA_REUSE/USC-TIMIT_reuse/{spk}

Usage:
    python check_sample_rates.py                 # both corpora, default paths
    python check_sample_rates.py --corpus lss
    python check_sample_rates.py --enhanced /alt/enhanced_audio
"""

from __future__ import annotations

import argparse
import wave
from collections import Counter
from pathlib import Path

try:
    import soundfile as sf          # wav/flac/...
    def _rate(p: Path) -> int:
        return sf.info(str(p)).samplerate
except Exception:                   # stdlib fallback (wav only)
    def _rate(p: Path) -> int:
        with wave.open(str(p), "rb") as w:
            return w.getframerate()

# ---- default source roots (from the fave_work builders) ----
LSS_ORIG = Path("/project2/shrikann_35/sfoley/data/single_spk_corpus")
ENHANCED = Path("/project2/shrikann_35/sfoley/data/enhanced_audio")
TIMIT_ORIG = Path("/project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT")

GROUPS = ["dsp", "raw"]
TIMIT_SPKS = ["F1", "F5", "M1", "M3"]
EXTS = ("*.wav", "*.flac")


def rates_in(d: Path) -> Counter:
    c = Counter()
    if not d.is_dir():
        return c
    for ext in EXTS:
        for p in sorted(d.glob(ext)):
            try:
                c[_rate(p)] += 1
            except Exception as e:
                print(f"[err] {p}: {e}")
    return c


def report(name: str, d: Path) -> Counter:
    c = rates_in(d)
    if not c:
        print(f"  {name:<24s} (no audio)   {d}")
    else:
        rates = ", ".join(f"{r} Hz x{n}" for r, n in sorted(c.items()))
        print(f"  {name:<24s} {rates}")
    return c


def lss_conditions(g: str, enhanced: Path, lss_orig: Path) -> dict[str, Path]:
    return {
        f"{g}_mri": lss_orig / f"audio_{g}",
        "meta": enhanced / g / "META_denoiser" / "USC_LSS_meta",
        "nvidia": enhanced / g / "NVIDIA_REUSE" / "USC_LSS_nvidia",
        "pase": enhanced / g / "PASE" / "USC_LSS_pase",
    }


def timit_conditions(spk: str, enhanced: Path, timit_orig: Path) -> dict[str, Path]:
    return {
        "orig_mri": timit_orig / "MRI" / "Data" / spk / "wav",
        "orig_ema": timit_orig / "EMA" / "Data" / spk / "wav",
        "meta": enhanced / "META_denoiser" / "USC-TIMIT_denoiser" / spk / "wav",
        "nvidia": enhanced / "NVIDIA_REUSE" / "USC-TIMIT_reuse" / spk,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--corpus", choices=["lss", "timit", "both"], default="both")
    ap.add_argument("--lss-orig", type=Path, default=LSS_ORIG)
    ap.add_argument("--timit-orig", type=Path, default=TIMIT_ORIG)
    ap.add_argument("--enhanced", type=Path, default=ENHANCED)
    args = ap.parse_args()

    total = Counter()

    if args.corpus in ("lss", "both"):
        print("=== USC_LSS ===")
        for g in GROUPS:
            for cond, d in lss_conditions(g, args.enhanced, args.lss_orig).items():
                total += report(f"{g}/{cond}", d)

    if args.corpus in ("timit", "both"):
        print("\n=== USC-TIMIT ===")
        for spk in TIMIT_SPKS:
            for cond, d in timit_conditions(spk, args.enhanced, args.timit_orig).items():
                total += report(f"{cond}/{spk}", d)

    print("\n--- overall ---")
    for r, n in sorted(total.items()):
        print(f"  {r} Hz : {n} files")
    if len(total) > 1:
        print("  [!] mixed sampling rates present")


if __name__ == "__main__":
    main()