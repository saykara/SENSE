#!/bin/bash

export PYTHONPATH=./

RES_DIR=C:/Users/Utkua/OneDrive/SENSE/SENSE/data/temp


python tools/train_ego_autoencoder.py pre-train \
   --dataset sceneflow \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel ${RES_DIR} \
   --workers 0 \
   --lr 0.001 \
   --lr-steps 70 \
   --lr-gamma 0.1 \
   --epochs 100 \
   --bn-type syncbn \
   --batch-size 2 \
   --corr-radius 4  \
   --disp-crop-imh 256 \
   --disp-crop-imw 512 \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --disp-loss-weight 0.25 \
   --print-freq 1