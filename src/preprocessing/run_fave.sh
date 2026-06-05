#!/bin/bash

# --- raw ---
fave-extract corpus /project2/shrikann_35/sfoley/data/fave_work/raw/USC_LSS/raw_mri --destination /project2/shrikann_35/sfoley/data/fave_results/raw/USC_LSS/raw_mri 
fave-extract corpus /project2/shrikann_35/sfoley/data/fave_work/raw/USC_LSS/meta --destination /project2/shrikann_35/sfoley/data/fave_results/raw/USC_LSS/meta 
fave-extract corpus /project2/shrikann_35/sfoley/data/fave_work/raw/USC_LSS/nvidia --destination /project2/shrikann_35/sfoley/data/fave_results/raw/USC_LSS/nvidia 
fave-extract corpus /project2/shrikann_35/sfoley/data/fave_work/raw/USC_LSS/pase --destination /project2/shrikann_35/sfoley/data/fave_results/raw/USC_LSS/pase 

# --- dsp (pase only) ---
fave-extract corpus /project2/shrikann_35/sfoley/data/fave_work/dsp/USC_LSS/pase --destination /project2/shrikann_35/sfoley/data/fave_results/dsp/USC_LSS/pase 