#!/bin/bash

python -O scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+replicantface" \
    --with-swa \
    --with-nll-loss \
    --rampup-nll-losses \
    --backbone resnet18 \
    --no-pointhead \
    --outdir model_files/current/test
