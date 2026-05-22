"""
Convert USC-TIMIT-style .trans files to Praat TextGrids for FAVE / new-fave.

Input format (one phone per line, comma-separated):
    start,end,phone,word,sentence
Plus a first line with the speaker ID (e.g. "s").

Output: a .TextGrid with two IntervalTiers:
    - "{speaker} - phone"
    - "{speaker} - word"
Empty word fields become "sp" (short pause) on the word tier, which is
what FAVE expects.

Usage:
    python trans_to_textgrid.py input.trans [output.TextGrid] [--speaker NAME]
    # batch (single trans directory):
    python trans_to_textgrid.py --batch /path/to/trans_dir /path/to/out_dir
    # walk a USC-TIMIT-style tree (Data/{spk}/trans -> Data/{spk}/textgrids):
    python trans_to_textgrid.py --walk /path/to/MRI/Data
    python trans_to_textgrid.py --walk /path/to/EMA/Data
"""

from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


def parse_trans(path: Path) -> Tuple[str, List[Interval], List[Interval]]:
    """Return (speaker, phone_intervals, word_intervals)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    speaker = lines[0].strip()
    phone_rows: List[Tuple[float, float, str, str]] = []

    for ln in lines[1:]:
        # split on commas, but the sentence field may itself contain commas — rare in TIMIT prompts.
        # Use maxsplit so the sentence stays intact if it has commas.
        parts = ln.split(",", 4)
        if len(parts) < 4:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        phone = parts[2].strip()
        word = parts[3].strip()  # strips the leading space you have in " MOVE"
        phone_rows.append((start, end, phone, word))

    if not phone_rows:
        raise ValueError(f"No phone rows parsed from {path}")

    # Build phone tier directly
    phone_intervals = [Interval(s, e, p if p else "sp") for (s, e, p, _w) in phone_rows]

    # Build word tier by merging consecutive rows with the same word label.
    # Empty word -> "sp" (silence/pause), which FAVE skips during vowel extraction.
    word_intervals: List[Interval] = []
    cur_word = None
    cur_start = None
    cur_end = None
    for (s, e, _p, w) in phone_rows:
        label = w if w else "sp"
        if cur_word is None:
            cur_word, cur_start, cur_end = label, s, e
        elif label == cur_word and abs(s - cur_end) < 1e-6:
            # contiguous, same word — extend
            cur_end = e
        else:
            word_intervals.append(Interval(cur_start, cur_end, cur_word))
            cur_word, cur_start, cur_end = label, s, e
    if cur_word is not None:
        word_intervals.append(Interval(cur_start, cur_end, cur_word))

    # Sanity: fill any tiny gaps (floating-point drift) with sp so the tier is gapless
    phone_intervals = _seal_gaps(phone_intervals)
    word_intervals = _seal_gaps(word_intervals)

    return speaker, phone_intervals, word_intervals


def _seal_gaps(intervals: List[Interval], eps: float = 1e-6) -> List[Interval]:
    """Insert sp intervals to fill any gaps between consecutive intervals."""
    if not intervals:
        return intervals
    sealed = [intervals[0]]
    for nxt in intervals[1:]:
        prev = sealed[-1]
        if nxt.xmin - prev.xmax > eps:
            sealed.append(Interval(prev.xmax, nxt.xmin, "sp"))
        elif nxt.xmin < prev.xmax:
            # overlap — clamp
            nxt = Interval(prev.xmax, max(nxt.xmax, prev.xmax), nxt.text)
        sealed.append(nxt)
    return sealed


def write_textgrid(out_path: Path, speaker: str,
                   phones: List[Interval], words: List[Interval]) -> None:
    xmin = min(phones[0].xmin, words[0].xmin)
    xmax = max(phones[-1].xmax, words[-1].xmax)

    def fmt_tier(name: str, ivs: List[Interval]) -> str:
        lines = []
        lines.append('    class = "IntervalTier"')
        lines.append(f'    name = "{name}"')
        lines.append(f'    xmin = {xmin}')
        lines.append(f'    xmax = {xmax}')
        lines.append(f'    intervals: size = {len(ivs)}')
        for i, iv in enumerate(ivs, 1):
            text = iv.text.replace('"', '""')
            lines.append(f'    intervals [{i}]:')
            lines.append(f'        xmin = {iv.xmin}')
            lines.append(f'        xmax = {iv.xmax}')
            lines.append(f'        text = "{text}"')
        return "\n".join(lines)

    body = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        f'xmin = {xmin}',
        f'xmax = {xmax}',
        'tiers? <exists>',
        'size = 2',
        'item []:',
        '    item [1]:',
        fmt_tier('words', words),
        '    item [2]:',
        fmt_tier('phones', phones),
        '',
    ]
    out_path.write_text("\n".join(body), encoding="utf-8")


def convert_one(in_path: Path, out_path: Path, speaker_override: str | None = None) -> None:
    speaker, phones, words = parse_trans(in_path)
    if speaker_override:
        speaker = speaker_override
    write_textgrid(out_path, speaker, phones, words)
    print(f"[ok] {in_path.name} -> {out_path.name} "
          f"({len(phones)} phones, {len(words)} words, speaker='{speaker}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input .trans file or directory (with --batch)")
    ap.add_argument("output", nargs="?", help="output .TextGrid or directory")
    ap.add_argument("--batch", action="store_true",
                    help="treat input as a directory of .trans files")
    ap.add_argument("--walk", action="store_true",
                    help="walk a USC-TIMIT-style tree: input is a Data/ root "
                         "containing per-speaker subdirs each with a trans/ folder; "
                         "outputs go to Data/{spk}/textgrids/")
    ap.add_argument("--speaker", default=None,
                    help="override speaker name written into the tier names "
                         "(in --walk mode the per-speaker dir name is used instead)")
    args = ap.parse_args()

    in_p = Path(args.input)

    if args.walk:
        if not in_p.is_dir():
            sys.exit("--walk requires input to be a Data/ directory")
        spk_dirs = sorted([d for d in in_p.iterdir() if d.is_dir()])
        if not spk_dirs:
            sys.exit(f"no speaker subdirectories found under {in_p}")
        total_files = 0
        for spk_dir in spk_dirs:
            trans_dir = spk_dir / "trans"
            if not trans_dir.is_dir():
                print(f"[skip] {spk_dir.name}: no trans/ subdir", file=sys.stderr)
                continue
            out_dir = spk_dir / "textgrids"
            out_dir.mkdir(parents=True, exist_ok=True)
            spk_name = args.speaker or spk_dir.name
            tfs = sorted(trans_dir.glob("*.trans"))
            if not tfs:
                print(f"[skip] {spk_dir.name}: no .trans files", file=sys.stderr)
                continue
            print(f"[{spk_dir.name}] {len(tfs)} files -> {out_dir}")
            for tf in tfs:
                out_path = out_dir / (tf.stem + ".TextGrid")
                try:
                    convert_one(tf, out_path, spk_name)
                    total_files += 1
                except Exception as e:
                    print(f"  [err] {tf.name}: {e}", file=sys.stderr)
        print(f"\ndone — converted {total_files} files across {len(spk_dirs)} speaker dirs")
    elif args.batch:
        if not in_p.is_dir():
            sys.exit("--batch requires input to be a directory")
        out_dir = Path(args.output) if args.output else in_p
        out_dir.mkdir(parents=True, exist_ok=True)
        for tf in sorted(in_p.glob("*.trans")):
            out_path = out_dir / (tf.stem + ".TextGrid")
            try:
                convert_one(tf, out_path, args.speaker)
            except Exception as e:
                print(f"[err] {tf.name}: {e}", file=sys.stderr)
    else:
        out_path = Path(args.output) if args.output else in_p.with_suffix(".TextGrid")
        convert_one(in_p, out_path, args.speaker)


if __name__ == "__main__":
    main()
