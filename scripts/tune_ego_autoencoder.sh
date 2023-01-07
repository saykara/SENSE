#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model/EgoAutoencoderTuned
LOAD_DIR=/content/model_0040.pth
BASE_DIR=/content/dataset

python tools/train_ego_autoencoder.py finetune \
   --dataset kittimalaga \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --savemodel ${RES_DIR} \
   --loadmodel ${LOAD_DIR} \
   --resume ${LOAD_DIR} \
   --workers 2 \
   --lr 0.0001 \
   --lr-steps 70 \
   --lr-gamma 0.1 \
   --epochs 40 \
   --bn-type syncbn \
   --batch-size 3 \
   --print-freq 20 \
   --enc-arch psm \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --corr-radius 4 \
   --base-dir ${BASE_DIR}