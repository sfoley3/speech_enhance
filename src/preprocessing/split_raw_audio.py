"""
Trim raw LSS audio to the length of the matching DSP file.

    audio_raw/usc_s1_1.wav  (len L_raw)
    audio_dsp/usc_s1_1.wav  (len L_dsp <= L_raw)
 -> audio_raw_clean/usc_s1_1.wav  (first L_dsp samples of raw)

"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

ORIG_DIR = Path("/project2/shrikann_35/sfoley/data/single_spk_corpus")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--raw-dir", type=Path, default=ORIG_DIR / "audio_raw")
    ap.add_argument("--dsp-dir", type=Path, default=ORIG_DIR / "audio_dsp")
    ap.add_argument("--out-dir", type=Path, default=ORIG_DIR / "audio_raw_clean")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for raw_path in sorted(args.raw_dir.glob("*.wav")):
        dsp_path = args.dsp_dir / raw_path.name
        if not dsp_path.is_file():
            print(f"[skip] no DSP match for {raw_path.name}")
            n_skip += 1
            continue

        raw, sr_raw = sf.read(str(raw_path))
        dsp_len = sf.info(str(dsp_path)).frames
        sr_dsp = sf.info(str(dsp_path)).samplerate

        if sr_raw != sr_dsp:
            print(f"[skip] sample-rate mismatch {raw_path.name}: "
                  f"raw {sr_raw} vs dsp {sr_dsp}")
            n_skip += 1
            continue

        if len(raw) < dsp_len:
            print(f"[warn] raw shorter than dsp ({len(raw)} < {dsp_len}); "
                  f"copying raw as-is: {raw_path.name}")
            trimmed = raw
        else:
            trimmed = raw[:dsp_len]

        sf.write(str(args.out_dir / raw_path.name), trimmed, sr_raw)
        cut = (len(raw) - len(trimmed)) / sr_raw
        print(f"[ok] {raw_path.name}: {len(raw)} -> {len(trimmed)} samples "
              f"(cut {cut:.2f}s)")
        n_ok += 1

    print(f"\ndone: {n_ok} trimmed, {n_skip} skipped -> {args.out_dir}")


if __name__ == "__main__":
    main()