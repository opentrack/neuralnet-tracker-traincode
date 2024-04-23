#!/bin/bash

#python -m cProfile -o run.prof 

python scripts/train_poseestimator.py --lr 1.e-3 --epochs 1500 --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+synface" \
    --save-plot train.pdf \
    --with-swa \
    --with-nll-loss \
    --backbone mobilenetv1

#--roi-override original
# +synface