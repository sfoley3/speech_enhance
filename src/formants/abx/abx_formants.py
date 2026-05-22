
# calc ABX scores for USC TIMIT - baseline
# calc ABX scores for USC LSS - compare to baseline: orig vs. meta vs. nvidia enhanced audio
# need to calc within spk for TIMIT - SPKS = ["F1", "F5", "M1", "M3"]; orig_ema only in fave_results; avg across spk
# return plot baseline in grey - EMA; then orig MRI, meta, nvidia in different colors; error bars 
# ABX = DTW + KL as distance based on F1_s-F3_s
# csv headers: F1,F2,F3,F1_s,F2_s,F3_s,B1,B2,B3,error,
# time,rel_time,prop_time,max_formant,n_formant,smooth_method,file_name,id,group,label,speaker_num,f0,
# intensity,optimized,word,stress,dur,pre_word,fol_word,pre_seg,fol_seg,abs_pre_seg,abs_fol_seg,context


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

TIMIT_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC-TIMIT")
LSS_RESULTS_DIR = Path("/project2/shrikann_35/sfoley/data/fave_results/USC_LSS")
DEFAULT_OUT_DIR = Path("/scratch1/seanfole/speech_enhance/src/formants/abx")

REFERENCE = "orig_ema" # USC TIMIT
COMPARATORS = ["orig_mri", "meta", "nvidia"] # USC LSS
CONDITIONS = [REFERENCE, *COMPARATORS]

FORMANTS = ["F1_s", "F2_s", "F3_s"]
FORMANT_LABELS = {"F1_s": "F1", "F2_s": "F2", "F3_s": "F3"}