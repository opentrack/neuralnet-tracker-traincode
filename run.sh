#!/bin/bash

# python scripts/train_poseestimator.py --lr 1.e-3 --epochs 500 --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+synface" \
#     --save-plot train.pdf \
#     --with-swa \
#     --with-nll-loss \
#     --roi-override original \
#     --no-onnx \
#     --backbone mobilenetv1 \
#     --outdir model_files/



#--rampup_nll_losses \

python scripts/train_poseestimator_lightning.py --ds "repro_300_wlp+lapa_megaface_lp+wflw_lp+synface" \
    --epochs 10 \
    --with-swa \
    --with-nll-loss \
    --rampup-nll-losses