#!/usr/bin/env python3
"""
Combined sentence-splitting + P.835 stimuli generator
=====================================================

Replaces the two-step pipeline (split every scan into utterances on disk, then
build stimuli) with a single pass that NEVER stores intermediate split audio.

For each scan we compute sentence boundaries ONCE from the TextGrid + transcript
(reusing the alignment logic from get_sentence_chunks.py). Because every
condition (orig + 3 enhancement models) in both base forms (dsp, raw_clean) is
time-aligned to the same recording, the same [start, end] applies to all of
them. We then slice each of the 8 full-scan source WAVs at those timestamps and
pipe it straight through ffmpeg loudness-normalisation, writing only the final
opaque-named stimulus.

Design (matches the P.835 survey):
    4 conditions (orig + META_denoiser + NVIDIA_REUSE + PASE)
  x 2 bases      (dsp, raw_clean)        = 8 cells
  x N utterances (default 12)            = 96 stimuli

Outputs (into --output_dir, default ./):
    audio/stim_XXXX.wav     loudness-normalised stimuli (opaque names)
    mapping.csv             PRIVATE key: id -> scan/seg/text/base/condition/time
    audioSamples.js.txt     paste into index.html (id, url, utt)
    audioSamples.json       same, JSON
    mismatches.json         sentences skipped by the aligner (if any)

Usage (run on the cluster where the data lives):
    python make_p835_stimuli.py \
        --corpus_dir   /project2/shrikann_35/sfoley/data/single_spk_corpus \
        --enhanced_dir /project2/shrikann_35/sfoley/data/enhanced_audio \
        --output_dir   . \
        --num_utterances 12 --seed 42

    # preview selection + mapping without rendering any audio:
    python make_p835_stimuli.py --corpus_dir ... --enhanced_dir ... --dry_run

Requires: python3, ffmpeg on PATH, and the `textgrid` package
    (pip install textgrid).
"""

import os
import re
import csv
import json
import glob
import random
import argparse
import subprocess

import textgrid

# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------
BASES      = ["dsp", "raw_clean"]                                   # raw_clean is noisiest
CONDITIONS = ["orig", "META_denoiser", "NVIDIA_REUSE", "PASE"]      # orig + 3 models
# orig audio lives in the corpus dir; map each base form to its sub-folder.
ORIG_SUBDIR = {"dsp": "audio_dsp", "raw_clean": "audio_raw_clean"}

# Loudness target (exactly the recipe from the README).
LOUDNORM = "loudnorm=I=-23:LRA=7:TP=-2"
SAMPLE_RATE = "16000"
FORCE_MONO = True  # set False to keep original channel count

# Keep only utterances within this duration window (seconds) as MOS stimuli.
MIN_DUR = 2.0
MAX_DUR = 10.0
MAX_PER_SCAN = 2  # avoid pulling too many utterances from one recording

# --------------------------------------------------------------------------
# Anchors + gold/trap items (flagged in mapping via the `role` column;
# the survey rates them like any other sample).
#   anchor_clean : clean studio reference -> calibrates the TOP of the scale
#   anchor_noisy : heavily degraded clip  -> calibrates the BOTTOM of the scale
#   gold_high    : obviously good (clean); expected OVRL high  -> reject raters who score it low
#   gold_low     : obviously bad (degraded); expected OVRL low -> reject raters who score it high
# --------------------------------------------------------------------------
CLEAN_EX_DIR = "/project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT/EMA/Data"
CLEAN_SPK    = ["F1", "F5", "M1", "M3"]   # clean reference speakers (studio USC-TIMIT)

N_CLEAN_ANCHOR = 4    # clean high-anchor items
N_NOISY_ANCHOR = 4    # degraded low-anchor items
N_GOLD_HIGH    = 2    # clean gold items (expected high)
N_GOLD_LOW     = 2    # degraded gold items (expected low)

# Degradation recipe for low-anchor / gold_low (applied before loudnorm).
ANCHOR_LOWPASS_HZ = 3000   # band-limit to sound clearly worse
GOLD_LOWPASS_HZ   = 2000   # gold_low is even more obviously bad

EXPECTED = {  # coarse expected OVRL for gold scoring (1-5)
    "gold_high": ">=4",
    "gold_low":  "<=2",
}

# --------------------------------------------------------------------------
# Alignment logic (ported verbatim from get_sentence_chunks.py, audio-only)
# --------------------------------------------------------------------------
sentence_split_re = re.compile(r"[.!?;]")
word_clean_re     = re.compile(r"[^\w']+")
mismatch_log = []


def clean(tok: str) -> str:
    return tok.lower().strip("'\"")


def read_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    return [s.strip() for s in sentence_split_re.split(raw) if s.strip()]


def word_intervals(path):
    tg = textgrid.TextGrid.fromFile(path)
    tier = tg.getFirst("words")
    return [(clean(itv.mark), itv.minTime, itv.maxTime) for itv in tier if itv.mark.strip()]


def try_match(words, start, target, win, extra_tg=8):
    """Gappy subsequence matcher (see original docstring)."""
    def norm(s):
        return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    def eq(a, b):
        a = norm(a).replace("-", "")
        b = norm(b).replace("-", "")
        if a == b:
            return True
        if a.endswith("s") and a[:-1] == b: return True
        if b.endswith("s") and b[:-1] == a: return True
        if a.endswith("'s") and a[:-2] == b: return True
        if b.endswith("'s") and b[:-2] == a: return True
        return False

    def merge_match(j, tok):
        if j >= len(words):
            return 0
        w1 = words[j][0]
        if eq(tok, w1):
            return 1
        if j + 1 < len(words):
            w12 = words[j][0] + words[j + 1][0]
            if eq(tok, w12):
                return 2
        return 0

    n = len(words)
    for off in range(win + 1):
        j = start + off
        i = 0
        if j >= n:
            break
        first_j = None
        last_j  = None
        consumed = 0
        max_consume = len(target) + extra_tg
        while i < len(target) and j < n and consumed <= max_consume:
            step = merge_match(j, target[i])
            if step:
                if first_j is None:
                    first_j = j
                j += step
                consumed += step
                last_j = j - 1
                i += 1
            else:
                j += 1
                consumed += 1
        if i == len(target) and first_j is not None and last_j is not None:
            return first_j, words[first_j:last_j + 1]
    return None, None


def sentence_segments(tg, txt, win):
    """Return [{'start','end','text'}] for sentences whose last word aligns."""
    def normalize_quotes(s: str) -> str:
        return (s.replace("’", "'").replace("‘", "'")
                 .replace("“", '"').replace("”", '"'))

    def pluralish_equal(a: str, b: str) -> bool:
        if a == b:
            return True
        if a.endswith("s") and a[:-1] == b: return True
        if b.endswith("s") and b[:-1] == a: return True
        if a.endswith("'s") and a[:-2] == b: return True
        if b.endswith("'s") and b[:-2] == a: return True
        return False

    words = word_intervals(tg)
    sentences = read_sentences(txt)
    segs, idx = [], 0

    for sent in sentences:
        toks = [clean(w) for w in re.split(word_clean_re, normalize_quotes(sent)) if w]
        if not toks:
            continue
        match_idx, span = try_match(words, idx, toks, win)
        if match_idx is None:
            mismatch_log.append({"file": os.path.basename(txt), "sentence": sent,
                                 "reason": "no_match", "start_idx": idx})
            idx += 1
            continue
        span_words = [w for (w, _, _) in span]
        last_ok = toks and span_words and pluralish_equal(toks[-1], span_words[-1])
        if not last_ok:
            mismatch_log.append({"file": os.path.basename(txt), "sentence": sent,
                                 "reason": "last_word_mismatch", "start_idx": idx,
                                 "target_last": toks[-1], "tg_last": span_words[-1] if span_words else ""})
            idx += 1
            continue
        segs.append({"start": span[0][1], "end": span[-1][2], "text": sent})
        idx = match_idx + len(toks)
    return segs


# --------------------------------------------------------------------------
# Source-path resolution
# --------------------------------------------------------------------------
_subdir_cache = {}


def enhanced_subdir(enhanced_dir, base, condition):
    """Each model dir holds exactly one USC_LSS_* folder; find it once."""
    key = (base, condition)
    if key not in _subdir_cache:
        pat = os.path.join(enhanced_dir, base, condition, "USC_LSS*")
        hits = sorted(glob.glob(pat))
        _subdir_cache[key] = hits[0] if hits else None
    return _subdir_cache[key]


def source_wav(args, base, condition, scan):
    """Full-scan source WAV for a (base, condition, scan) cell, or None."""
    if condition == "orig":
        p = os.path.join(args.corpus_dir, ORIG_SUBDIR[base], f"{scan}.wav")
    else:
        sub = enhanced_subdir(args.enhanced_dir, base, condition)
        p = os.path.join(sub, f"{scan}.wav") if sub else None
    return p if (p and os.path.isfile(p)) else None


def all_cells_exist(args, scan):
    """True only if every one of the 8 (base, condition) sources exists."""
    for base in BASES:
        for cond in CONDITIONS:
            if source_wav(args, base, cond, scan) is None:
                return False
    return True


# --------------------------------------------------------------------------
# Rendering: slice + loudness-normalise in one ffmpeg call (no temp files)
# --------------------------------------------------------------------------
def render(src, start, end, out_path):
    dur = max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
           "-af", LOUDNORM, "-ar", SAMPLE_RATE]
    if FORCE_MONO:
        cmd += ["-ac", "1"]
    cmd += [out_path]
    # NOTE: -ss before -i is sample-accurate for PCM/WAV input. If a source is
    # ever a compressed format, move -ss to AFTER -i for accurate seeking.
    subprocess.run(cmd, check=True)


def render_degraded(src, start, end, out_path, lowpass_hz):
    """Slice, band-limit, then loudness-normalise (one ffmpeg call). No added noise."""
    dur = max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
           "-af", f"lowpass=f={lowpass_hz},{LOUDNORM}", "-ar", SAMPLE_RATE]
    if FORCE_MONO:
        cmd += ["-ac", "1"]
    cmd += [out_path]
    subprocess.run(cmd, check=True)


def ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def gather_clean_pool(rng):
    """
    Collect clean studio reference clips for the high anchors / gold_high.

    USC-TIMIT clean files are typically one sentence per file, so we use the
    whole file (capped at MAX_DUR). Adjust the glob if your layout differs.
    Returns [{'source_wav', 'start', 'end', 'text', 'scan'}].
    """
    if not os.path.isdir(CLEAN_EX_DIR):
        print(f"[WARN] CLEAN_EX_DIR not found: {CLEAN_EX_DIR} -> skipping clean anchors/gold_high")
        return []
    items = []
    for spk in CLEAN_SPK:
        spk_dir = os.path.join(CLEAN_EX_DIR, spk)
        if not os.path.isdir(spk_dir):
            print(f"[WARN] missing speaker dir for clean pool: {spk_dir} -> skipping")
            continue
        for w in sorted(glob.glob(os.path.join(spk_dir, "**", "*.wav"), recursive=True)):
            dur = ffprobe_duration(w)
            if dur is None or dur < MIN_DUR:
                continue
            items.append({"source_wav": w, "start": 0.0,
                          "end": min(dur, MAX_DUR), "text": "", "scan": f"clean_{spk}"})
    rng.shuffle(items)
    return items


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Combined splitter + P.835 stimuli generator")
    p.add_argument("--corpus_dir", default="/project2/shrikann_35/sfoley/data/single_spk_corpus",
                   help="single_spk_corpus dir (textgrids, transcripts, audio_dsp, audio_raw_clean)")
    p.add_argument("--enhanced_dir", default="/project2/shrikann_35/sfoley/data/enhanced_audio",
                   help="enhanced_audio dir (dsp/<MODEL>/USC_LSS_*, raw_clean/<MODEL>/USC_LSS_*)")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--num_utterances", type=int, default=12)
    p.add_argument("--align_window", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="select utterances + write mapping, but render no audio")
    return p.parse_args()


def build_pool(args):
    """All valid utterances across scans that have a full set of 8 sources."""
    tg_dir  = os.path.join(args.corpus_dir, "textgrids")
    txt_dir = os.path.join(args.corpus_dir, "transcripts")
    scans = sorted(
        os.path.splitext(f)[0] for f in os.listdir(tg_dir)
        if f.endswith(".TextGrid")
        and os.path.isfile(os.path.join(txt_dir, os.path.splitext(f)[0] + ".txt"))
    )
    pool = []
    for scan in scans:
        if not all_cells_exist(args, scan):
            continue
        tg  = os.path.join(tg_dir,  f"{scan}.TextGrid")
        txt = os.path.join(txt_dir, f"{scan}.txt")
        try:
            segs = sentence_segments(tg, txt, args.align_window)
        except Exception as e:
            print(f"[WARN] {scan}: alignment failed ({e})")
            continue
        for i, ch in enumerate(segs):
            dur = ch["end"] - ch["start"]
            if MIN_DUR <= dur <= MAX_DUR:
                pool.append({"scan": scan, "seg": i, "start": ch["start"],
                             "end": ch["end"], "text": ch["text"], "dur": dur})
    return pool


def select_utterances(pool, n, rng):
    """Randomly pick n utterances, capping how many come from one scan."""
    rng.shuffle(pool)
    chosen, per_scan = [], {}
    for u in pool:
        if per_scan.get(u["scan"], 0) >= MAX_PER_SCAN:
            continue
        chosen.append(u)
        per_scan[u["scan"]] = per_scan.get(u["scan"], 0) + 1
        if len(chosen) == n:
            break
    return chosen


def main():
    args = get_args()
    rng = random.Random(args.seed)
    audio_out = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)

    pool = build_pool(args)
    print(f"[INFO] candidate utterances (after duration filter): {len(pool)}")
    if len(pool) < args.num_utterances:
        print(f"[WARN] only {len(pool)} candidates for {args.num_utterances} requested")

    utts = select_utterances(pool, args.num_utterances, rng)
    if len(utts) < args.num_utterances:
        raise SystemExit(f"[ERROR] could only select {len(utts)} utterances; "
                         f"loosen MIN_DUR/MAX_DUR/MAX_PER_SCAN or check sources.")

    # ------------------------------------------------------------------
    # Build a unified item list. Each item has a `role` and a render plan:
    #   plan = ("plain", src) | ("degraded", src, lowpass, noise)
    # ------------------------------------------------------------------
    items = []

    # (1) Experimental cells: 8 per utterance.
    for k, u in enumerate(utts, start=1):
        utt_label = f"utt{k:02d}"
        for base in BASES:
            for cond in CONDITIONS:
                src = source_wav(args, base, cond, u["scan"])
                items.append({
                    "role": "experimental", "utt": utt_label,
                    "base": base, "condition": cond,
                    "scan": u["scan"], "seg": u["seg"],
                    "start": u["start"], "end": u["end"], "dur": u["dur"],
                    "text": u["text"], "expected": "",
                    "plan": ("plain", src),
                })

    # (2) Clean high-anchors + gold_high (drawn from the clean studio corpus).
    clean_pool = gather_clean_pool(rng)
    n_clean_needed = N_CLEAN_ANCHOR + N_GOLD_HIGH
    if clean_pool and len(clean_pool) < n_clean_needed:
        print(f"[WARN] only {len(clean_pool)} clean clips for {n_clean_needed} needed; reusing.")
    ci = 0
    def next_clean():
        nonlocal ci
        if not clean_pool:
            return None
        c = clean_pool[ci % len(clean_pool)]
        ci += 1
        return c
    for j in range(N_CLEAN_ANCHOR):
        c = next_clean()
        if not c: break
        items.append({"role": "anchor_clean", "utt": f"anc_clean{j+1:02d}",
                      "base": "clean", "condition": "clean_ref",
                      "scan": c["scan"], "seg": -1,
                      "start": c["start"], "end": c["end"], "dur": c["end"] - c["start"],
                      "text": c["text"], "expected": "",
                      "plan": ("plain", c["source_wav"])})
    for j in range(N_GOLD_HIGH):
        c = next_clean()
        if not c: break
        items.append({"role": "gold_high", "utt": f"gold_hi{j+1:02d}",
                      "base": "clean", "condition": "clean_ref",
                      "scan": c["scan"], "seg": -1,
                      "start": c["start"], "end": c["end"], "dur": c["end"] - c["start"],
                      "text": c["text"], "expected": EXPECTED["gold_high"],
                      "plan": ("plain", c["source_wav"])})

    # (3) Degraded low-anchors + gold_low (degrade the noisiest experimental source:
    #     orig / raw_clean of the selected utterances).
    deg_sources = [u for u in utts]  # reuse the selected segments
    for j in range(N_NOISY_ANCHOR):
        u = deg_sources[j % len(deg_sources)]
        src = source_wav(args, "raw_clean", "orig", u["scan"])
        items.append({"role": "anchor_noisy", "utt": f"anc_noisy{j+1:02d}",
                      "base": "raw_clean", "condition": "degraded",
                      "scan": u["scan"], "seg": u["seg"],
                      "start": u["start"], "end": u["end"], "dur": u["dur"],
                      "text": u["text"], "expected": "",
                      "plan": ("degraded", src, ANCHOR_LOWPASS_HZ)})
    for j in range(N_GOLD_LOW):
        u = deg_sources[(j + N_NOISY_ANCHOR) % len(deg_sources)]
        src = source_wav(args, "raw_clean", "orig", u["scan"])
        items.append({"role": "gold_low", "utt": f"gold_lo{j+1:02d}",
                      "base": "raw_clean", "condition": "degraded",
                      "scan": u["scan"], "seg": u["seg"],
                      "start": u["start"], "end": u["end"], "dur": u["dur"],
                      "text": u["text"], "expected": EXPECTED["gold_low"],
                      "plan": ("degraded", src, GOLD_LOWPASS_HZ)})

    # Opaque, shuffled IDs across ALL items so role can't be inferred from name/order.
    ids = [f"stim_{i:04d}" for i in range(1, len(items) + 1)]
    rng.shuffle(ids)
    for it, sid in zip(items, ids):
        it["id"] = sid
    items.sort(key=lambda c: c["id"])

    # Render + collect mapping rows.
    rows = []
    for it in items:
        plan = it["plan"]
        src = plan[1]
        out_path = os.path.join(audio_out, f"{it['id']}.wav")
        if not args.dry_run:
            if src is None or not os.path.isfile(src):
                print(f"[WARN] missing source for {it['id']} "
                      f"({it['role']}/{it['condition']}/{it['base']}/{it['scan']}); skipping")
                continue
            if plan[0] == "plain":
                render(src, it["start"], it["end"], out_path)
            else:
                _, _, lp = plan
                render_degraded(src, it["start"], it["end"], out_path, lp)
        rows.append({
            "id": it["id"], "role": it["role"], "scan": it["scan"], "seg": it["seg"],
            "utterance": it["utt"], "base": it["base"], "condition": it["condition"],
            "start": round(it["start"], 3), "end": round(it["end"], 3),
            "dur": round(it["dur"], 3), "expected_overall": it["expected"],
            "text": it["text"], "source_wav": src or "",
            "filename": f"{it['id']}.wav", "url": f"audio/{it['id']}.wav",
        })

    rows.sort(key=lambda r: r["id"])

    # mapping.csv (PRIVATE - keep out of any public survey repo)
    map_path = os.path.join(args.output_dir, "mapping.csv")
    with open(map_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "role", "scan", "seg", "utterance", "base",
                                          "condition", "start", "end", "dur",
                                          "expected_overall", "text",
                                          "source_wav", "filename", "url"])
        w.writeheader()
        w.writerows(rows)

    # audioSamples for index.html
    js_items = ",\n".join(
        f'    {{ id: "{r["id"]}", url: "{r["url"]}", utt: "{r["utterance"]}" }}' for r in rows
    )
    with open(os.path.join(args.output_dir, "audioSamples.js.txt"), "w") as f:
        f.write("const audioSamples = [\n" + js_items + "\n  ];\n")
    with open(os.path.join(args.output_dir, "audioSamples.json"), "w") as f:
        json.dump([{"id": r["id"], "url": r["url"], "utt": r["utterance"]} for r in rows],
                  f, indent=2)

    if mismatch_log:
        with open(os.path.join(args.output_dir, "mismatches.json"), "w") as f:
            json.dump(mismatch_log, f, indent=2, ensure_ascii=False)
        print(f"[INFO] {len(mismatch_log)} sentences skipped by aligner; see mismatches.json")

    from collections import Counter
    role_counts = Counter(r["role"] for r in rows)
    mode = "DRY RUN (no audio written)" if args.dry_run else "rendered"
    print(f"[DONE] {mode}: {len(rows)} stimuli total")
    print(f"       experimental: {role_counts.get('experimental', 0)} "
          f"({len(utts)} utt x {len(BASES)} bases x {len(CONDITIONS)} conditions)")
    for role in ("anchor_clean", "anchor_noisy", "gold_high", "gold_low"):
        if role_counts.get(role):
            print(f"       {role}: {role_counts[role]}")
    print(f"       audio:   {audio_out}")
    print(f"       mapping: {map_path}")


if __name__ == "__main__":
    main()