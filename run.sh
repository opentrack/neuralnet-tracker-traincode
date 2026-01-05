#!/bin/bash

python scripts/train_poseestimator.py -c --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+replicantface+panoptic" \
    --with-swa \
    --backbone resnet18 \
    --with-nll \
    --roi-override extent_to_forehead \
    --outdir model_files/current/resnet18