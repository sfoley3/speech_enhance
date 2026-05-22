"""Verify that enhanced audio files match across both models and the original audio.

For each speaker, compares the set of .wav filenames in:
  - ORIG_DIR/EMA/Data/<SPK>/wav/
  - ENHANCED_DIR/META_denoiser/USC-TIMIT_denoiser/<SPK>/wav/
  - ENHANCED_DIR/NVIDIA_REUSE/USC-TIMIT_reuse/<SPK>/

Also compares word and phone tier label sequences between the EMA and MRI
TextGrids under ORIG_DIR/{EMA,MRI}/Data/<SPK>/textgrids/, matched by the
same ema<->mri filename normalization used for the wav check.

Prints per-speaker counts and any mismatches.
"""

import json
import os
import re
import sys

ORIG_DIR = "/project2/shrikann_35/xuanshi/DATA/SPAN/USC-TIMIT/USC-TIMIT"
ENHANCED_DIR = "/project2/shrikann_35/sfoley/data/enhanced_audio"

SPKS = ["F1", "F5", "M1", "M3"]

SOURCES = {
    "orig":   lambda spk: os.path.join(ORIG_DIR, "EMA", "Data", spk, "wav"),
    "meta":   lambda spk: os.path.join(ENHANCED_DIR, "META_denoiser", "USC-TIMIT_denoiser", spk, "wav"),
    "nvidia": lambda spk: os.path.join(ENHANCED_DIR, "NVIDIA_REUSE", "USC-TIMIT_reuse", spk),
}

# TextGrid sources (enhanced models reuse the MRI textgrids, so only EMA vs MRI exist).
TG_SOURCES = {
    "ema": lambda spk: os.path.join(ORIG_DIR, "EMA", "Data", spk, "textgrids"),
    "mri": lambda spk: os.path.join(ORIG_DIR, "MRI", "Data", spk, "textgrids"),
}


def normalize(fname):
    """Normalize filenames so 'ema' (orig) and 'mri' (meta/nvidia) variants match."""
    return fname.replace("ema", "<SET>").replace("mri", "<SET>")


def list_wavs(path):
    if not os.path.isdir(path):
        return None
    return {normalize(f) for f in os.listdir(path) if f.endswith(".wav")}


def list_textgrids(path):
    """Return {normalized_filename: full_path} for .TextGrid files, or None if missing."""
    if not os.path.isdir(path):
        return None
    out = {}
    for f in os.listdir(path):
        if f.lower().endswith(".textgrid"):
            out[normalize(f)] = os.path.join(path, f)
    return out


def parse_textgrid(path):
    """Parse a Praat long-form TextGrid.

    Returns {tier_name: [(xmin, xmax, label), ...]} for IntervalTier tiers.
    Empty-label intervals are retained so timings line up for downstream
    grouping; callers filter them as needed.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-16") as fh:
            text = fh.read()

    tiers = {}
    current_tier = None
    current_class = None
    cur_xmin = None
    cur_xmax = None

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("item ["):
            current_class = None
            current_tier = None
            cur_xmin = cur_xmax = None
        elif line.startswith("class = "):
            current_class = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
            if current_class == "IntervalTier":
                current_tier = name
                tiers.setdefault(current_tier, [])
            else:
                current_tier = None
        elif line.startswith("intervals [") and current_tier is not None:
            cur_xmin = cur_xmax = None
        elif line.startswith("xmin = ") and current_tier is not None:
            try:
                cur_xmin = float(line.split("=", 1)[1].strip())
            except ValueError:
                cur_xmin = None
        elif line.startswith("xmax = ") and current_tier is not None:
            try:
                cur_xmax = float(line.split("=", 1)[1].strip())
            except ValueError:
                cur_xmax = None
        elif line.startswith("text = ") and current_tier is not None:
            val = line[len("text = "):].strip()
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                label = val[1:-1]
            else:
                label = val.strip('"')
            label = label.replace('""', '"')
            tiers[current_tier].append((cur_xmin, cur_xmax, label))
            cur_xmin = cur_xmax = None

    return tiers


def group_phones_by_word(words, phones, eps=1e-6):
    """For each non-empty word interval, list non-empty phones contained in it.

    Word labels are lowercased; 'sp' labels are treated as silence and ignored.
    Returns [(word_label, [phone_label, ...]), ...] in word-tier order.
    """
    out = []
    for w_xmin, w_xmax, w_label in words:
        wl = w_label.lower() if w_label else w_label
        if not wl or wl == "sp":
            continue
        if w_xmin is None or w_xmax is None:
            out.append((wl, []))
            continue
        ph = [
            p_label
            for p_xmin, p_xmax, p_label in phones
            if p_label and p_label != "sp"
            and p_xmin is not None and p_xmax is not None
            and p_xmin >= w_xmin - eps
            and p_xmax <= w_xmax + eps
        ]
        out.append((wl, ph))
    return out


def _word_labels(intervals):
    """Extract non-empty, non-'sp' word labels (lowercased) from interval tuples."""
    return [lab.lower() for _, _, lab in intervals if lab and lab.lower() != "sp"]


_STIMULUS_RE = re.compile(r"(?<!\d)(\d+)_(\d+)\Z")


def _stimulus(stem):
    """Extract the trailing '#_#' stimulus id from a filename stem.

    Anchored to the end of the stem with a non-digit lookbehind so speaker
    prefixes like 'f1_' don't pollute the match
    (e.g., 'usctimit_<SET>_f1_116_120' -> '116_120').
    """
    m = _STIMULUS_RE.search(stem)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def _positional_diffs(a, b):
    """Count positions where two sequences differ, treating length diff as misses."""
    n = max(len(a), len(b))
    return sum(1 for i in range(n)
               if i >= len(a) or i >= len(b) or a[i] != b[i])


def _edit_counts(ref, hyp):
    """Levenshtein S/D/I counts between ref and hyp; returns (S, D, I, N_ref).

    Used for WER/PER: error_rate = (S + D + I) / N_ref.
    Backtracks the standard DP table to attribute each error.
    """
    n, m = len(ref), len(hyp)
    # DP table of edit distances.
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1],  # sub
                                  d[i - 1][j],      # del
                                  d[i][j - 1])      # ins
    # Backtrack to count S/D/I.
    s = di = ii = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            di += 1
            i -= 1
        else:
            ii += 1
            j -= 1
    return s, di, ii, n


def _pct(num, denom):
    return f"{(100.0 * num / denom):.2f}%" if denom else "n/a"


def compare_textgrids(spk, allowed_keys=None):
    """Compare EMA vs MRI textgrid contents for one speaker.

    If allowed_keys is provided, only TextGrid filenames (normalized) in that
    set are probed; others are counted as skipped.

    Returns (ema_count, mri_count, matched_count, skipped_count,
             word_mm, phone_word_mm, missing_dirs, records).
    records is a list of structured mismatch dicts suitable for JSON logging.
    """
    ema_dir = TG_SOURCES["ema"](spk)
    mri_dir = TG_SOURCES["mri"](spk)
    ema_map = list_textgrids(ema_dir)
    mri_map = list_textgrids(mri_dir)

    missing = []
    if ema_map is None:
        missing.append(("ema", ema_dir))
    if mri_map is None:
        missing.append(("mri", mri_dir))
    if missing:
        return {
            "ema_count": 0, "mri_count": 0, "matched": 0, "skipped": 0,
            "word_seq_mismatches": 0, "files_with_phone_mismatch": 0,
            "wer": {"S": 0, "D": 0, "I": 0, "N_ref": 0, "errors": 0, "rate": None},
            "per": {"S": 0, "D": 0, "I": 0, "N_ref": 0, "errors": 0, "rate": None},
            "missing": missing, "records": [],
        }

    ema_keys = set(ema_map)
    mri_keys = set(mri_map)
    matched_all = ema_keys & mri_keys

    if allowed_keys is not None:
        allowed_stems = {os.path.splitext(k)[0] for k in allowed_keys}
        matched = {k for k in matched_all if os.path.splitext(k)[0] in allowed_stems}
        skipped = len(matched_all) - len(matched)
        only_ema = sorted(k for k in (ema_keys - mri_keys) if os.path.splitext(k)[0] in allowed_stems)
        only_mri = sorted(k for k in (mri_keys - ema_keys) if os.path.splitext(k)[0] in allowed_stems)
    else:
        matched = matched_all
        skipped = 0
        only_ema = sorted(ema_keys - mri_keys)
        only_mri = sorted(mri_keys - ema_keys)

    records = []
    if only_ema:
        records.append({"spk": spk, "type": "only_in_ema", "files": only_ema})
    if only_mri:
        records.append({"spk": spk, "type": "only_in_mri", "files": only_mri})

    word_mm = 0
    phone_word_mm = 0
    word_ref_tokens = 0
    word_sub = word_del = word_ins = 0
    phone_ref_tokens = 0
    phone_sub = phone_del = phone_ins = 0
    for key in sorted(matched):
        stem = os.path.splitext(key)[0]
        stim = _stimulus(stem)
        try:
            ema_tiers = parse_textgrid(ema_map[key])
            mri_tiers = parse_textgrid(mri_map[key])
        except Exception as e:
            records.append({
                "spk": spk, "type": "parse_error",
                "file": key, "stimulus": stim, "error": str(e),
            })
            continue

        ema_words = ema_tiers.get("words", [])
        mri_words = mri_tiers.get("words", [])
        ema_phones = ema_tiers.get("phones", [])
        mri_phones = mri_tiers.get("phones", [])

        ema_word_seq = _word_labels(ema_words)
        mri_word_seq = _word_labels(mri_words)
        ws, wd, wi, wn = _edit_counts(ema_word_seq, mri_word_seq)
        word_sub += ws; word_del += wd; word_ins += wi; word_ref_tokens += wn
        if ema_word_seq != mri_word_seq:
            word_mm += 1
            records.append({
                "spk": spk, "type": "word_sequence_mismatch",
                "file": key, "stimulus": stim,
                "ema_words": ema_word_seq,
                "mri_words": mri_word_seq,
                "wer": {"S": ws, "D": wd, "I": wi, "N_ref": wn,
                        "rate": (ws + wd + wi) / wn if wn else None},
            })

        # Per-word phoneme comparison (occurrence-keyed).
        ema_by_word = {}
        for w, ph in group_phones_by_word(ema_words, ema_phones):
            ema_by_word.setdefault(w, []).append(ph)
        mri_by_word = {}
        for w, ph in group_phones_by_word(mri_words, mri_phones):
            mri_by_word.setdefault(w, []).append(ph)

        phone_diffs = []
        for w in sorted(set(ema_by_word) & set(mri_by_word)):
            ema_occs = ema_by_word[w]
            mri_occs = mri_by_word[w]
            n = min(len(ema_occs), len(mri_occs))
            for i in range(n):
                ps, pd_, pi, pn = _edit_counts(ema_occs[i], mri_occs[i])
                phone_sub += ps; phone_del += pd_; phone_ins += pi; phone_ref_tokens += pn
                if ema_occs[i] != mri_occs[i]:
                    phone_diffs.append({
                        "word": w,
                        "occurrence": i,
                        "ema_phones": ema_occs[i],
                        "mri_phones": mri_occs[i],
                        "per": {"S": ps, "D": pd_, "I": pi, "N_ref": pn,
                                "rate": (ps + pd_ + pi) / pn if pn else None},
                    })
        if phone_diffs:
            phone_word_mm += 1
            records.append({
                "spk": spk, "type": "phone_mismatch",
                "file": key, "stimulus": stim,
                "diffs": phone_diffs,
            })

    word_err = word_sub + word_del + word_ins
    phone_err = phone_sub + phone_del + phone_ins
    return {
        "ema_count": len(ema_map),
        "mri_count": len(mri_map),
        "matched": len(matched),
        "skipped": skipped,
        "word_seq_mismatches": word_mm,
        "files_with_phone_mismatch": phone_word_mm,
        "wer": {
            "S": word_sub, "D": word_del, "I": word_ins,
            "N_ref": word_ref_tokens, "errors": word_err,
            "rate": (word_err / word_ref_tokens) if word_ref_tokens else None,
        },
        "per": {
            "S": phone_sub, "D": phone_del, "I": phone_ins,
            "N_ref": phone_ref_tokens, "errors": phone_err,
            "rate": (phone_err / phone_ref_tokens) if phone_ref_tokens else None,
        },
        "missing": [],
        "records": records,
    }


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "verify_files.log.json")
SENT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "verify_files.sentences.json")
SENTENCE_LIST_PATH = os.path.join(ORIG_DIR, "list_of_sentences.txt")


def _load_sentence_list(path):
    """Parse 'NNN : sentence text.' -> {int_id: [normalized word, ...]}."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if not m:
                continue
            sid = int(m.group(1))
            toks = re.findall(r"[A-Za-z']+", m.group(2).lower())
            out[sid] = toks
    return out


def _expand_stim(stim):
    """'116_120' -> [116, 117, 118, 119, 120]."""
    try:
        a, b = stim.split("_")
        a, b = int(a), int(b)
    except (ValueError, AttributeError):
        return []
    if b < a:
        a, b = b, a
    return list(range(a, b + 1))


def _is_subseq(needle, haystack):
    """True iff needle appears as a contiguous slice of haystack."""
    if not needle:
        return True
    L = len(needle)
    for i in range(0, len(haystack) - L + 1):
        if haystack[i:i + L] == needle:
            return True
    return False


def _sentence_diagnostic(spk, wav_intersect_keys, sentences):
    """Per-speaker sentence-identity check.

    Returns dict with counts and structured records covering:
      - files only in EMA or only in MRI (which stimuli/sentences are missing)
      - matched files whose recovered word stream does not contain every
        expected sentence id from the official list (i.e., the file at that
        stimulus range carries different sentences than expected on some side)
    """
    ema_dir = TG_SOURCES["ema"](spk)
    mri_dir = TG_SOURCES["mri"](spk)
    ema_map = list_textgrids(ema_dir) or {}
    mri_map = list_textgrids(mri_dir) or {}

    ema_keys = set(ema_map)
    mri_keys = set(mri_map)
    matched = sorted(ema_keys & mri_keys)
    only_ema = sorted(ema_keys - mri_keys)
    only_mri = sorted(mri_keys - ema_keys)

    def _describe_missing(key):
        stem = os.path.splitext(key)[0]
        stim = _stimulus(stem)
        ids = _expand_stim(stim) if stim else []
        return {
            "spk": spk,
            "file": key,
            "stimulus": stim,
            "expected_ids": ids,
            "expected_sentences": [
                {"id": sid, "text": " ".join(sentences.get(sid, []))}
                for sid in ids
            ],
        }

    ema_only_records = [_describe_missing(k) for k in only_ema]
    mri_only_records = [_describe_missing(k) for k in only_mri]

    mismatch_records = []
    matched_diff = 0
    matched_same = 0
    for key in matched:
        stem = os.path.splitext(key)[0]
        stim = _stimulus(stem)
        ids = _expand_stim(stim) if stim else []
        if not ids:
            continue
        try:
            ema_tiers = parse_textgrid(ema_map[key])
            mri_tiers = parse_textgrid(mri_map[key])
        except Exception:
            continue
        ema_seq = _word_labels(ema_tiers.get("words", []))
        mri_seq = _word_labels(mri_tiers.get("words", []))

        ema_contains = [
            {"id": sid, "found": _is_subseq(sentences.get(sid, []), ema_seq)}
            for sid in ids
        ]
        mri_contains = [
            {"id": sid, "found": _is_subseq(sentences.get(sid, []), mri_seq)}
            for sid in ids
        ]

        ema_missing = [c["id"] for c in ema_contains if not c["found"]]
        mri_missing = [c["id"] for c in mri_contains if not c["found"]]

        if ema_missing or mri_missing:
            matched_diff += 1
            mismatch_records.append({
                "spk": spk,
                "file": key,
                "stimulus": stim,
                "in_wav_intersection": key in wav_intersect_keys,
                "expected": [
                    {"id": sid, "text": " ".join(sentences.get(sid, []))}
                    for sid in ids
                ],
                "ema": {
                    "sentence": " ".join(ema_seq),
                    "contains": ema_contains,
                    "missing_ids": ema_missing,
                },
                "mri": {
                    "sentence": " ".join(mri_seq),
                    "contains": mri_contains,
                    "missing_ids": mri_missing,
                },
            })
        else:
            matched_same += 1

    return {
        "counts": {
            "files_only_in_ema": len(only_ema),
            "files_only_in_mri": len(only_mri),
            "matched_files": len(matched),
            "matched_with_diff_sentences": matched_diff,
            "matched_with_same_sentences": matched_same,
        },
        "files_only_in_ema": ema_only_records,
        "files_only_in_mri": mri_only_records,
        "matched_with_diff_sentences": mismatch_records,
    }


def main():
    total_mismatches = 0
    wav_intersect = {}
    log = {"wav": {}, "textgrid": {}}

    print(f"{'spk':<5} {'orig':>6} {'meta':>6} {'nvidia':>7}   status")
    print("-" * 50)

    for spk in SPKS:
        sets = {name: list_wavs(getter(spk)) for name, getter in SOURCES.items()}

        missing_dirs = [n for n, s in sets.items() if s is None]
        counts = {n: (len(s) if s is not None else 0) for n, s in sets.items()}

        line = f"{spk:<5} {counts['orig']:>6} {counts['meta']:>6} {counts['nvidia']:>7}"

        if missing_dirs:
            print(f"{line}   MISSING DIR: {missing_dirs}")
            for n in missing_dirs:
                print(f"      -> {SOURCES[n](spk)}")
            total_mismatches += 1
            wav_intersect[spk] = set()
            log["wav"][spk] = {
                "status": "missing_dir",
                "counts": counts,
                "missing": {n: SOURCES[n](spk) for n in missing_dirs},
            }
            continue

        orig, meta, nvidia = sets["orig"], sets["meta"], sets["nvidia"]
        wav_intersect[spk] = orig & meta & nvidia
        if orig == meta == nvidia:
            print(f"{line}   OK")
            log["wav"][spk] = {"status": "ok", "counts": counts}
        else:
            print(f"{line}   MISMATCH")
            total_mismatches += 1
            pair_diffs = {}
            for a_name, b_name in [("orig", "meta"), ("orig", "nvidia"), ("meta", "nvidia")]:
                a, b = sets[a_name], sets[b_name]
                only_a = sorted(a - b)
                only_b = sorted(b - a)
                if only_a or only_b:
                    pair_diffs[f"{a_name}_vs_{b_name}"] = {
                        f"only_in_{a_name}": only_a,
                        f"only_in_{b_name}": only_b,
                    }
                if only_a:
                    print(f"      in {a_name} but not {b_name} ({len(only_a)}): {only_a[:5]}{' ...' if len(only_a) > 5 else ''}")
                if only_b:
                    print(f"      in {b_name} but not {a_name} ({len(only_b)}): {only_b[:5]}{' ...' if len(only_b) > 5 else ''}")
            log["wav"][spk] = {
                "status": "mismatch",
                "counts": counts,
                "pair_diffs": pair_diffs,
            }

    print("-" * 50)
    print(f"Speakers with issues: {total_mismatches}/{len(SPKS)}")

    # ---- TextGrid content check (EMA vs MRI), restricted to wav intersection ----
    # WER/PER computed with EMA as reference, MRI as hypothesis (Levenshtein).
    print()
    print("TextGrid content check (EMA ref vs MRI hyp, restricted to wav intersection):")
    header = (f"{'spk':<5} {'ema':>5} {'mri':>5} {'match':>6} {'skip':>5} "
              f"{'w_mm':>5} {'p_mm':>5} {'WER':>7} {'PER':>7}   status")
    print(header)
    print("-" * len(header))

    tg_issues = 0
    g_w_s = g_w_d = g_w_i = g_w_n = 0
    g_p_s = g_p_d = g_p_i = g_p_n = 0
    for spk in SPKS:
        allowed = wav_intersect.get(spk, set())
        res = compare_textgrids(spk, allowed_keys=allowed)
        ema_n = res["ema_count"]
        mri_n = res["mri_count"]
        matched = res["matched"]
        skipped = res["skipped"]
        w_mm = res["word_seq_mismatches"]
        p_mm = res["files_with_phone_mismatch"]
        wer = res["wer"]
        per = res["per"]
        missing = res["missing"]
        records = res["records"]

        g_w_s += wer["S"]; g_w_d += wer["D"]; g_w_i += wer["I"]; g_w_n += wer["N_ref"]
        g_p_s += per["S"]; g_p_d += per["D"]; g_p_i += per["I"]; g_p_n += per["N_ref"]

        wer_str = _pct(wer["errors"], wer["N_ref"])
        per_str = _pct(per["errors"], per["N_ref"])
        line = (f"{spk:<5} {ema_n:>5} {mri_n:>5} {matched:>6} {skipped:>5} "
                f"{w_mm:>5} {p_mm:>5} {wer_str:>7} {per_str:>7}")

        spk_log = {
            "ema_count": ema_n, "mri_count": mri_n,
            "matched": matched, "skipped_due_to_wav": skipped,
            "word_seq_mismatches": w_mm, "files_with_phone_mismatch": p_mm,
            "wer": wer, "per": per,
            "records": records,
        }
        if missing:
            print(f"{line}   MISSING DIR: {[m[0] for m in missing]}")
            for name, path in missing:
                print(f"      -> {name}: {path}")
            tg_issues += 1
            spk_log["status"] = "missing_dir"
            spk_log["missing"] = {n: p for n, p in missing}
        elif not records and w_mm == 0 and p_mm == 0:
            print(f"{line}   OK")
            spk_log["status"] = "ok"
        else:
            print(f"{line}   MISMATCH ({len(records)} records -> {LOG_PATH})")
            tg_issues += 1
            spk_log["status"] = "mismatch"

        log["textgrid"][spk] = spk_log

    print("-" * len(header))
    print(f"Speakers with textgrid issues: {tg_issues}/{len(SPKS)}")
    g_w_err = g_w_s + g_w_d + g_w_i
    g_p_err = g_p_s + g_p_d + g_p_i
    print(f"Overall WER: {_pct(g_w_err, g_w_n)}  "
          f"(S={g_w_s} D={g_w_d} I={g_w_i} N_ref={g_w_n})")
    print(f"Overall PER: {_pct(g_p_err, g_p_n)}  "
          f"(S={g_p_s} D={g_p_d} I={g_p_i} N_ref={g_p_n})")

    log["textgrid_totals"] = {
        "wer": {"S": g_w_s, "D": g_w_d, "I": g_w_i, "N_ref": g_w_n,
                "errors": g_w_err, "rate": (g_w_err / g_w_n) if g_w_n else None},
        "per": {"S": g_p_s, "D": g_p_d, "I": g_p_i, "N_ref": g_p_n,
                "errors": g_p_err, "rate": (g_p_err / g_p_n) if g_p_n else None},
    }

    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote detailed log to: {LOG_PATH}")

    # ---- Sentence-identity diagnostic (EMA vs MRI, by stimulus range) ----
    sentences = _load_sentence_list(SENTENCE_LIST_PATH)
    print()
    print(f"Sentence-identity check against {SENTENCE_LIST_PATH}")
    if not sentences:
        print("  [warn] sentence list empty or unreadable; skipping.")
    else:
        header = (f"{'spk':<5} {'only_ema':>9} {'only_mri':>9} "
                  f"{'matched':>8} {'diff':>5} {'same':>5}")
        print(header)
        print("-" * len(header))

        sent_log = {
            "sentence_list": SENTENCE_LIST_PATH,
            "totals": {
                "files_only_in_ema": 0,
                "files_only_in_mri": 0,
                "matched_files": 0,
                "matched_with_diff_sentences": 0,
                "matched_with_same_sentences": 0,
            },
            "per_speaker": {},
        }
        for spk in SPKS:
            allowed = wav_intersect.get(spk, set())
            res = _sentence_diagnostic(spk, allowed, sentences)
            c = res["counts"]
            for k, v in c.items():
                sent_log["totals"][k] += v
            sent_log["per_speaker"][spk] = res
            print(f"{spk:<5} {c['files_only_in_ema']:>9} "
                  f"{c['files_only_in_mri']:>9} {c['matched_files']:>8} "
                  f"{c['matched_with_diff_sentences']:>5} "
                  f"{c['matched_with_same_sentences']:>5}")

        with open(SENT_LOG_PATH, "w", encoding="utf-8") as fh:
            json.dump(sent_log, fh, indent=2, ensure_ascii=False)
        print(f"Wrote sentence-identity log to: {SENT_LOG_PATH}")

    return 1 if (total_mismatches or tg_issues) else 0


if __name__ == "__main__":
    sys.exit(main())
