#!/usr/bin/env python
"""Replicate MATLAB RMS-v1 truncation on MAIN _pre, self-verified against audio_trunc."""
import os, glob, re
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta

SRC   = "/project2/shrikann_35/sfoley/data/single_spk_corpus/audio_orig_full"
TRUNC = os.path.join(SRC, "audio_trunc")
OUT   = os.path.join(SRC, "main_trunc")
os.makedirs(OUT, exist_ok=True)

TS_FMT = "%Y-%m-%d,%H;%M;%S"


def mround(v):  # MATLAB round: half away from zero
    return int(np.floor(v + 0.5)) if v >= 0 else int(np.ceil(v - 0.5))


def movmean(x, k):  # MATLAB movmean(x,k): centered, shrinking windows at edges
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    nB = k // 2
    nF = k - 1 - nB
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(n)
    lo = np.maximum(i - nB, 0)
    hi = np.minimum(i + nF, n - 1)
    return (c[hi + 1] - c[lo]) / (hi - lo + 1)


def find_rms_drop1(signal, fs):  # returns 1-based dropInd, matching MATLAB
    thresh_ratio = 0.9
    w         = mround(0.05 * fs)
    start_cut = mround(0.5 * fs)                       # 1-based
    end_cut   = mround(min(7 * fs, 0.7 * len(signal)))
    end_ind   = len(signal) - end_cut                 # 1-based
    rms = np.sqrt(movmean(signal ** 2, w))
    rms_thresh = thresh_ratio * rms[start_cut - 1:end_ind].min()
    tail = rms[end_ind - 1:]
    below = np.where(tail < rms_thresh)[0]
    if below.size == 0:
        return None
    return end_ind + (int(below[0]) + 1)


def mono(x):
    return x[:, 0] if x.ndim > 1 else x


def resolve_dsp_ts(ts):  # MAIN names can be +/-1s off the DSP/audio_trunc name
    base = datetime.strptime(ts, TS_FMT)
    for off in (0, 1, -1):
        cand = (base + timedelta(seconds=off)).strftime(TS_FMT)
        if os.path.isfile(os.path.join(TRUNC, f"DSP_OUT_{cand}.wav")):
            return cand
    return None


ok = bad = skipped = 0
for main_path in sorted(glob.glob(os.path.join(SRC, "MAIN_*_pre.wav"))):
    if main_path.endswith("_pre_ref.wav"):
        continue
    ts = re.search(r"MAIN_(.*)_pre\.wav", os.path.basename(main_path)).group(1)

    dsp_ts = resolve_dsp_ts(ts)
    if dsp_ts is None:
        print("skip (no audio_trunc):", ts); skipped += 1; continue

    clip, cfs = sf.read(os.path.join(TRUNC, f"DSP_OUT_{dsp_ts}.wav"))
    dsp,  dfs = sf.read(os.path.join(SRC,   f"DSP_OUT_{dsp_ts}.wav"))
    main, mfs = sf.read(main_path)
    clip, dsp, main = mono(clip), mono(dsp), mono(main)
    assert cfs == dfs == mfs, f"{ts}: fs mismatch clip={cfs} dsp={dfs} main={mfs}"

    num = len(clip) - 1                  # num_audio_frames (else-branch length = num+1)
    end_idx = find_rms_drop1(main, mfs)  # anchor on MAIN _pre, exactly as MATLAB
    if end_idx is None:
        print(f"skip (no RMS drop found): {ts}"); skipped += 1; continue

    s0 = end_idx - num - 1               # 0-based start of MATLAB slice
    if s0 < 0:
        print(f"WARN {ts}: zero-pad branch (end_idx={end_idx} < num={num})"); bad += 1; continue

    # verify: rebuild DSP slice with this end_idx and compare to existing audio_trunc
    dsp_slice = dsp[s0:end_idx]          # length num+1
    if not (len(dsp_slice) == len(clip) and np.allclose(dsp_slice, clip, atol=1e-6)):
        print(f"MISMATCH {ts}: end_idx={end_idx} len_slice={len(dsp_slice)} len_clip={len(clip)}")
        bad += 1; continue

    main_trunc = main[s0:end_idx]
    sf.write(os.path.join(OUT, f"MAIN_{ts}_pre.wav"), main_trunc, mfs)
    print(f"{ts}  end_idx={end_idx}  front_trim={s0} ({s0/mfs:.3f}s)  wrote {len(main_trunc)}")
    ok += 1

print(f"\ndone: verified+wrote {ok}, mismatch/bad {bad}, no-trunc {skipped}")