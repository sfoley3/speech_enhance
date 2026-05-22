"""
Build per-(condition, speaker) working directories for new-fave.

Each working dir contains symlinks to the audio for that condition + symlinks to
the appropriate TextGrids (MRI textgrids for orig_mri/meta/nvidia, EMA textgrids
for orig_ema). Run new-fave once per condition over the per-speaker subdirs.

Layout produced under WORK_DIR:
    fave_work/
        orig_mri/{spk}/  -> *.wav (MRI orig) + *.TextGrid (MRI)
        orig_ema/{spk}/  -> *.wav (EMA orig) + *.TextGrid (EMA)
        meta/{spk}/      -> *.wav (META)     + *.TextGrid (MRI)
        nvidia/{spk}/    -> *.wav (NVIDIA)   + *.TextGrid (MRI)
"""

from __future__ import annotations

import sys
from pathlib import Path

ORIG_DIR = Path("/project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT")
ENHANCED_DIR = Path("/project2/shrikann_35/sfoley/data/enhanced_audio")
WORK_DIR = Path("/project2/shrikann_35/sfoley/data/fave_work/USC-TIMIT")
RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT")

SPKS = ["F1", "F5", "M1", "M3"]  # only spk with both MRI and EMA


# textgrid sources:
#   MRI textgrids cover orig_mri + both denoised conditions (same recordings, same alignments)
#   EMA textgrids cover orig_ema only
def mri_tg(spk):
    return ORIG_DIR / "MRI" / "Data" / spk / "textgrids"


def ema_tg(spk):
    return ORIG_DIR / "EMA" / "Data" / spk / "textgrids"


CONDITIONS = {
    "orig_mri": {
        "wav_dir": lambda spk: ORIG_DIR / "MRI" / "Data" / spk / "wav",
        "tg_dir": mri_tg,
    },
    "orig_ema": {
        "wav_dir": lambda spk: ORIG_DIR / "EMA" / "Data" / spk / "wav",
        "tg_dir": ema_tg,
    },
    "meta": {
        "wav_dir": lambda spk: (
            ENHANCED_DIR / "META_denoiser" / "USC-TIMIT_denoiser" / spk / "wav"
        ),
        "tg_dir": mri_tg,
    },
    "nvidia": {
        "wav_dir": lambda spk: ENHANCED_DIR / "NVIDIA_REUSE" / "USC-TIMIT_reuse" / spk,
        "tg_dir": mri_tg,
    },
}


def link_dir(src_dir: Path, pattern: str, dest_dir: Path) -> int:
    """Symlink all files matching pattern in src_dir into dest_dir. Returns count."""
    if not src_dir.is_dir():
        print(f"  [warn] missing: {src_dir}", file=sys.stderr)
        return 0
    n = 0
    for f in sorted(src_dir.glob(pattern)):
        link = dest_dir / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f)
        n += 1
    return n


def check_pairs(work_subdir: Path) -> tuple[int, list[str]]:
    """Return (n_pairs, orphans). Orphans are basenames missing a wav or TextGrid."""
    stems_wav = {p.stem for p in work_subdir.glob("*.wav")}
    stems_tg = {p.stem for p in work_subdir.glob("*.TextGrid")}
    paired = stems_wav & stems_tg
    orphans = sorted((stems_wav ^ stems_tg))
    return len(paired), orphans


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for cond, spec in CONDITIONS.items():
        for spk in SPKS:
            sub = WORK_DIR / cond / spk
            sub.mkdir(parents=True, exist_ok=True)
            wav_dir = spec["wav_dir"](spk)
            tg_dir = spec["tg_dir"](spk)
            n_wav = link_dir(wav_dir, "*.wav", sub)
            n_tg = link_dir(tg_dir, "*.TextGrid", sub)
            n_pair, orphans = check_pairs(sub)
            line = f"[{cond}/{spk}] wav={n_wav} tg={n_tg} paired={n_pair}"
            if orphans:
                line += f"  ORPHANS({len(orphans)}): " + ", ".join(orphans[:5])
                if len(orphans) > 5:
                    line += f", ... (+{len(orphans) - 5})"
            print(line)
            summary.append((cond, spk, n_wav, n_tg, n_pair, len(orphans)))

    # print fave-extract commands
    print("\n# --- fave-extract commands ---")
    for cond in CONDITIONS:
        print(
            f"fave-extract subcorpora {WORK_DIR}/{cond}/* "
            f"--destination {RESULTS_DIR}/{cond}"
        )


if __name__ == "__main__":
    main()
