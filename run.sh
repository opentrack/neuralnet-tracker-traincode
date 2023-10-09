#!/bin/bash

python scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+synface+lapa_megaface_lp+wflw_lp" --auglevel 2 \
    --save-plot train.pdf \
    --with-swa \
    --backbone mobilenetv1