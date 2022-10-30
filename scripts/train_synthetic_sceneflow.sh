#!/bin/bash

export PYTHONPATH=./

RES_DIR=./output/sceneflow

python tools/train_joint_synthetic_sceneflow.py pre-train \
   --dataset flyingthings3d \
   --enc-arch psmnext \
   --dec-arch pwcdcnext \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdcnext \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel ${RES_DIR} \
   --workers 2 \
   --lr 0.001 \
   --lr-steps 70 \
   --lr-gamma 0.1 \
   --epochs 10 \
   --bn-type syncbn \
   --batch-size 1 \
   --corr-radius 4  \
   --disp-crop-imh 256 \
   --disp-crop-imw 512 \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --disp-loss-weight 0.25 \
   --print-freq 20 \
   --kernel-size 7 \
   --dec-kernel-size 7