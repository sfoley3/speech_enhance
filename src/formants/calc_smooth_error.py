"""
Token-level FastTrack smoothing error (`error` column) from fave-extract
`*_tracks.csv`, as a *reference-free* formant-track quality metric.

The `error` column is FastTrack's own mismatch between each vowel's raw formant
track and its DCT smooth -- one value per token. We read it, dedup to one row
per token, pool across files, and compare distributions. No EMA reference or DTW
needed (cf. calc_rmse.py / abx_utils.py).

Conditions (mirrors abx_compare_grouped.py)
-------------------------------------------
EMA baseline  : USC-TIMIT `orig_ema` (clean speech floor, 4 speakers).
LSS grid      : group in {raw, dsp} x family in {mri, meta, nvidia, pase}.
Layout:
    {timit_dir}/orig_ema/{spk}/*_tracks.csv
    {lss_dir}/raw/USC_LSS/{raw_mri, meta, nvidia, pase}/*_tracks.csv
    {lss_dir}/dsp/USC_LSS/{dsp_mri, meta, nvidia, pase}/*_tracks.csv
(the mri folder is named `{group}_mri`; other families use the bare name.)

Shared sentence inventory (as in abx_utils): LSS keeps usc_s1_1..37 (drops the
Rainbow Passage usc_s1_38); TIMIT keeps sentence ids <= 455.

Plot
----
    [EMA]  [MRI: raw|dsp]  [META: raw|dsp]  [NVIDIA: raw|dsp]  [PASE: raw|dsp]
     grey       orange blue       ...
EMA grey far-left; per family two boxes, raw (orange) + dsp (blue).

Statistics
----------
PRIMARY  : H0 = no difference in token error between EMA and each of the 8
           LSS boxes. Mann-Whitney U (unpaired -- EMA is a separate corpus/
           recording session, NOT the same audio), Holm-corrected over the 8.
SECONDARY: H0 = no difference between raw and dsp. Token-paired Wilcoxon
           signed-rank per family + a GLOBAL pooled test.

Usage
-----
    python formant_error_boxplot.py
    python formant_error_boxplot.py --lss-dir /project2/.../fave_results \
        --timit-dir /project2/.../fave_results/USC-TIMIT --out-dir /path/out
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scipy import stats  # noqa: E402

plt.rc('font', size=20)
plt.rc('legend', fontsize=15)  


# ============================== CONFIG ==============================

TIMIT_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/dsp/USC-TIMIT")
LSS_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results")
DEFAULT_OUT_DIR = Path("/scratch1/seanfole/speech_enhance/src/formants/error")

REFERENCE = "orig_ema"                       # USC-TIMIT clean EMA (baseline)
CORPUS = "USC_LSS"
GROUPS = ["raw_clean", "dsp"]
FAMILIES = ["mri", "meta", "nvidia", "pase"]
TIMIT_SPKS = ["F1", "F5", "M1", "M3"]

COND_LABELS = {
    "orig_ema": "EMA", "mri": "Orig", "meta": "Denoiser",
    "nvidia": "REUSE", "pase": "PASE",
}
COND_COLORS = {"orig_ema": "#bdbdbd", "raw_clean": "#dd8452", "dsp": "#4c72b0"}
GROUP_PLOT_ORDER = ["raw_clean", "dsp"]            # raw left, dsp right

ERROR_COL = "error"
PAIR_MIN = 8                                 # min matched tokens for paired test

# Shared-inventory caps (set to 0 to disable)
LSS_MAX_FILE = 37                            # drop usc_s1_38 (Rainbow Passage)
TIMIT_MAX_SENTENCE = 455                     # drop the *_456_460 range files

# Strip the modality token so EMA/MRI versions share a recording key
MODALITY_RE = re.compile(r"_(mri|ema)_", flags=re.IGNORECASE)
# ===================================================================


def folder_name(group: str, family: str) -> str:
    """LSS folder name: the mri family lives in `{group}_mri`, others bare."""
    return f"{group}_mri" if family == "mri" else family


def _normalize_recording(file_name: str) -> str:
    stem = Path(str(file_name)).stem.replace("_tracks", "")
    return MODALITY_RE.sub("_", stem, count=1).lower()


def _iter_tracks_csvs(cond_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (speaker, csv_path). Supports {cond}/{spk}/*.csv and {cond}/*.csv."""
    if not cond_dir.is_dir():
        print(f"[warn] missing dir: {cond_dir}", file=sys.stderr)
        return
    spk_dirs = [d for d in cond_dir.iterdir() if d.is_dir()]
    if spk_dirs:
        for d in sorted(spk_dirs):
            for p in sorted(d.glob("*_tracks.csv")):
                yield d.name.upper(), p
    else:
        for p in sorted(cond_dir.glob("*_tracks.csv")):
            m = re.match(r"[a-z]+_(?:[a-z]+_)?([a-zA-Z0-9]+)_", p.stem)
            yield (m.group(1).upper() if m else "?"), p


def load_token_error(cond_dir: Path) -> Dict[Tuple[str, str, str], float]:
    """Return {(speaker, recording_key, vowel_id) -> error}, one row per token."""
    out: Dict[Tuple[str, str, str], float] = {}
    needed = {"file_name", "id", ERROR_COL}
    for spk, csv_path in _iter_tracks_csvs(cond_dir):
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in needed)
        except Exception as e:
            print(f"[err] {csv_path}: {e}", file=sys.stderr)
            continue
        if needed - set(df.columns):
            print(f"[warn] {csv_path.name} missing {needed - set(df.columns)}; skip",
                  file=sys.stderr)
            continue
        df[ERROR_COL] = pd.to_numeric(df[ERROR_COL], errors="coerce")
        df = df.dropna(subset=[ERROR_COL])
        df["rec_key"] = df["file_name"].map(_normalize_recording)
        first = df.groupby(["rec_key", "id"], sort=False)[ERROR_COL].first()
        for (rec_key, vid), err in first.items():
            out[(spk, str(rec_key), str(vid))] = float(err)
    return out


# ---------- shared-inventory filters (operate on the token dict) ----------

_FILE_NUM_RE = re.compile(r"(\d+)\Z")
_STIM_RE = re.compile(r"(?<!\d)(\d+)_(\d+)\Z")


def filter_lss_files(d: Dict, max_file: int) -> Dict:
    """Keep LSS tokens whose trailing file number is <= max_file."""
    if max_file <= 0:
        return d
    out = {k: v for k, v in d.items()
           if (m := _FILE_NUM_RE.search(k[1])) and int(m.group(1)) <= max_file}
    print(f"[filter] LSS file<= {max_file}: kept {len(out)}/{len(d)} tokens")
    return out


def filter_timit_sentences(d: Dict, max_sentence: int) -> Dict:
    """Keep TIMIT tokens whose stimulus range max is <= max_sentence."""
    if max_sentence <= 0:
        return d

    def smax(rec: str):
        m = _STIM_RE.search(rec)
        if m:
            return max(int(m.group(1)), int(m.group(2)))
        m2 = _FILE_NUM_RE.search(rec)
        return int(m2.group(1)) if m2 else None

    out = {k: v for k, v in d.items()
           if (s := smax(k[1])) is not None and s <= max_sentence}
    print(f"[filter] TIMIT sentence<= {max_sentence}: kept {len(out)}/{len(d)} tokens")
    return out


# ---------------------------------- stats ----------------------------------


def _rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def unpaired_test(a: Dict, b: Dict) -> dict:
    """Independent-samples Mann-Whitney U (EMA vs a comparison)."""
    xa = np.fromiter(a.values(), float)
    xb = np.fromiter(b.values(), float)
    res = {"n_a": len(xa), "n_b": len(xb), "n_pair": 0, "test": "mannwhitney"}
    if xa.size == 0 or xb.size == 0:
        res.update(stat=np.nan, p=np.nan, median_diff=np.nan, effect=np.nan)
        return res
    U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    res.update(stat=float(U), p=float(p),
               median_diff=float(np.median(xa) - np.median(xb)),
               effect=float(2 * U / (xa.size * xb.size) - 1))
    return res


def paired_test(a: Dict, b: Dict) -> dict:
    """Token-paired Wilcoxon signed-rank on shared keys (raw vs dsp)."""
    keys = [k for k in a if k in b]
    res = {"n_a": len(a), "n_b": len(b), "n_pair": len(keys),
           "test": "wilcoxon_paired"}
    if len(keys) < PAIR_MIN:
        res.update(stat=np.nan, p=np.nan, median_diff=np.nan, effect=np.nan,
                   warn=f"only {len(keys)} matched tokens (<{PAIR_MIN})")
        return res
    xa = np.array([a[k] for k in keys])
    xb = np.array([b[k] for k in keys])
    try:
        W, p = stats.wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        W, p = np.nan, np.nan
    res.update(stat=float(W), p=float(p),
               median_diff=float(np.median(xa - xb)),
               effect=_rank_biserial_paired(xa, xb))
    return res


def holm(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    valid = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    if not valid.any():
        return adj
    pv = p[valid]
    order = np.argsort(pv)
    a = np.empty_like(pv)
    for rank, idx in enumerate(order):
        a[idx] = min(1.0, pv[idx] * (len(pv) - rank))
    a[order] = np.maximum.accumulate(a[order])
    adj[valid] = a
    return adj


def summarize(name: str, d: Dict) -> str:
    x = np.fromiter(d.values(), float)
    if x.size == 0:
        return f"  {name:>14s}: (no data)"
    return (f"  {name:>14s}: n={x.size:5d}  mean={x.mean():.4f}  "
            f"median={np.median(x):.4f}  sd={x.std(ddof=1):.4f}")


def run_stats(ema: Dict, comps: Dict[Tuple[str, str], Dict]
              ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    lines = ["=== token-level FastTrack `error` ===", "", "-- summary --",
             summarize("EMA", ema)]
    for fam in FAMILIES:
        for g in GROUPS:
            lines.append(summarize(f"{COND_LABELS[fam]} {g}", comps[(fam, g)]))

    # PRIMARY: EMA vs each of the 8 LSS boxes (unpaired MWU, Holm over family)
    lines += ["", "-- PRIMARY: EMA vs each comparison "
              "(Mann-Whitney, unpaired; H0: no difference) --"]
    ema_rows, fam_p = [], []
    for fam in FAMILIES:
        for g in GROUPS:
            r = unpaired_test(ema, comps[(fam, g)])
            r.update(family=fam, group=g)
            ema_rows.append(r)
            fam_p.append(r["p"])
    adj = holm(fam_p)
    for r, pa in zip(ema_rows, adj):
        r["p_holm"] = float(pa)
        sig = "*" if np.isfinite(pa) and pa < 0.05 else " "
        lines.append(
            f"  EMA vs {COND_LABELS[r['family']]+' '+r['group']:<12s}: "
            f"p={r['p']:.3g}  p_holm={pa:.3g}{sig}  "
            f"Δmed={r['median_diff']:+.4f}  r={r['effect']:+.2f}  "
            f"(n_ema={r['n_a']}, n_b={r['n_b']})")
    vs_ema = pd.DataFrame(ema_rows)[
        ["family", "group", "n_a", "n_b", "stat", "median_diff",
         "effect", "p", "p_holm"]].rename(columns={"n_a": "n_ema", "n_b": "n_cond"})

    # SECONDARY: raw vs dsp, token-paired, per family + GLOBAL pooled
    lines += ["", "-- SECONDARY: raw vs dsp "
              "(Wilcoxon, token-paired; H0: no difference) --"]
    rd_rows = []
    pooled_raw, pooled_dsp = {}, {}
    for fam in FAMILIES:
        ra, ds = comps[(fam, "raw_clean")], comps[(fam, "dsp")]
        for k, v in ra.items():
            pooled_raw[(fam, *k)] = v
        for k, v in ds.items():
            pooled_dsp[(fam, *k)] = v
        r = paired_test(ra, ds)               # raw vs dsp: +Δ => raw rougher
        r["family"] = fam
        rd_rows.append(r)
        warn = f"  [!] {r['warn']}" if r.get("warn") else ""
        lines.append(
            f"  {COND_LABELS[fam]:<8s} raw vs dsp: p={r['p']:.3g}  "
            f"Δmed={r['median_diff']:+.4f}  r={r['effect']:+.2f}  "
            f"(n_pair={r['n_pair']}){warn}")
    rg = paired_test(pooled_raw, pooled_dsp)
    rg["family"] = "GLOBAL"
    rd_rows.append(rg)
    sig = "*" if np.isfinite(rg["p"]) and rg["p"] < 0.05 else " "
    warn = f"  [!] {rg['warn']}" if rg.get("warn") else ""
    lines.append(
        f"  {'GLOBAL':<8s} raw vs dsp: p={rg['p']:.3g}{sig}  "
        f"Δmed={rg['median_diff']:+.4f}  r={rg['effect']:+.2f}  "
        f"(n_pair={rg['n_pair']}){warn}")
    raw_vs_dsp = pd.DataFrame(rd_rows)[
        ["family", "n_pair", "stat", "median_diff", "effect", "p"]]

    return "\n".join(lines), vs_ema, raw_vs_dsp


# --------------------------------- plotting --------------------------------


def make_boxplot(ema: Dict, comps: Dict[Tuple[str, str], Dict], out_pdf: Path):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    box_data, positions, colors = [], [], []
    centers, labels = [], []

    # EMA single grey box, far left
    ev = np.fromiter(ema.values(), float)
    if ev.size:
        box_data.append(ev); positions.append(0.0); colors.append(COND_COLORS["orig_ema"])
    centers.append(0.0); labels.append("EMA")

    w = 0.34
    x = 1.2
    gap = 1.4
    for fam in FAMILIES:
        sub = []
        for j, g in enumerate(GROUP_PLOT_ORDER):
            vals = np.fromiter(comps[(fam, g)].values(), float)
            pos = x + (j - 0.5) * (w + 0.04)
            sub.append(pos)
            if vals.size:
                box_data.append(vals); positions.append(pos)
                colors.append(COND_COLORS[g])
        centers.append(float(np.mean(sub))); labels.append(COND_LABELS[fam])
        x += gap

    bp = ax.boxplot(box_data, positions=positions, widths=w,
                    patch_artist=True, showfliers=False, notch=True,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_edgecolor("black")

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error")
    #ax.set_title("Formant-track quality: token-level FastTrack error\n"
    #             "(reference-free; lower = smoother track)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_xlim(-0.6, centers[-1] + 0.8)

    handles = [Patch(facecolor=COND_COLORS["raw_clean"], edgecolor="black", label="Raw"),
               Patch(facecolor=COND_COLORS["dsp"], edgecolor="black", label="DSP")]
    ax.legend(handles=handles, title="", loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[plot] wrote {out_pdf}")


# ----------------------------------- main ----------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--lss-dir", type=Path, default=LSS_RESULTS_DIR,
                    help="root containing raw/USC_LSS and dsp/USC_LSS")
    ap.add_argument("--timit-dir", type=Path, default=TIMIT_RESULTS_DIR,
                    help="USC-TIMIT root containing orig_ema/{spk}/*_tracks.csv")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-file", type=int, default=LSS_MAX_FILE,
                    help="keep LSS files usc_s1_1..N (drops Rainbow Passage). 0=off")
    ap.add_argument("--max-sentence", type=int, default=TIMIT_MAX_SENTENCE,
                    help="keep TIMIT sentence ids <= N. 0=off")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] lss_dir   = {args.lss_dir}")
    print(f"[cfg] timit_dir = {args.timit_dir}")
    print(f"[cfg] out_dir   = {args.out_dir}")

    # EMA baseline from USC-TIMIT orig_ema
    ema = load_token_error(args.timit_dir / REFERENCE)
    ema = filter_timit_sentences(ema, args.max_sentence)
    print(f"[load] EMA (orig_ema): {len(ema)} tokens")

    # LSS grid
    comps: Dict[Tuple[str, str], Dict] = {}
    for fam in FAMILIES:
        for g in GROUPS:
            cond_dir = args.lss_dir / g / CORPUS / folder_name(g, fam)
            d = load_token_error(cond_dir)
            d = filter_lss_files(d, args.max_file)
            comps[(fam, g)] = d
            print(f"[load] {g}/{folder_name(g, fam)}: {len(d)} tokens")

    if not ema and all(len(d) == 0 for d in comps.values()):
        sys.exit("[err] no token data loaded; check CONFIG dir names / layout")

    # long-form dump
    rows = [{"family": "EMA", "group": "", "speaker": k[0],
             "recording": k[1], "vowel_id": k[2], "error": e}
            for k, e in ema.items()]
    for (fam, g), d in comps.items():
        for k, e in d.items():
            rows.append({"family": COND_LABELS[fam], "group": g, "speaker": k[0],
                         "recording": k[1], "vowel_id": k[2], "error": e})
    pd.DataFrame(rows).to_csv(args.out_dir / "formant_error_long.csv", index=False)

    stats_txt, vs_ema, raw_vs_dsp = run_stats(ema, comps)
    print("\n" + stats_txt)
    (args.out_dir / "formant_error_stats.txt").write_text(stats_txt + "\n")
    vs_ema.to_csv(args.out_dir / "formant_error_vs_ema_tests.csv", index=False)
    raw_vs_dsp.to_csv(args.out_dir / "formant_error_raw_vs_dsp_tests.csv", index=False)

    make_boxplot(ema, comps, args.out_dir / "formant_error_boxplot.pdf")


if __name__ == "__main__":
    main()