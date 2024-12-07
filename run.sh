#!/bin/bash

python scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+synface" \
    --with-swa \
    --with-nll-loss \
    --backbone hybrid_vit \
    --rampup-nll-losses
    
# --outdir model_files/current/run0/