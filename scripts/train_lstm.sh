#!/bin/bash

export PYTHONPATH=./

RES_DIR=./output/sceneflow
BASE_DIR=/content/dataset

python tools/train_lstm.py pre-train \
   --dataset mixed \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel ${RES_DIR} \
   --workers 0 \
   --lr 0.0001 \
   --lr-steps 20 \
   --lr-gamma 0.1 \
   --epochs 60 \
   --bn-type syncbn \
   --batch-size 36 \
   --corr-radius 4  \
   --disp-crop-imh 256 \
   --disp-crop-imw 512 \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --disp-loss-weight 0.25 \
   --print-freq 20 \
   --base-dir ${BASE_DIR}