
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