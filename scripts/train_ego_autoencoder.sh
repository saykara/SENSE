#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model/EgoAutoencoder
BASE_DIR=/content/dataset

python tools/train_ego_autoencoder.py pre-train \
   --dataset sceneflow \
   --ego-enc next \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --savemodel ${RES_DIR} \
   --workers 2 \
   --lr 0.0001 \
   --lr-steps 10 \
   --lr-gamma 0.1 \
   --epochs 40 \
   --bn-type syncbn \
   --batch-size 48 \
   --print-freq 20 \
   --base-dir ${BASE_DIR}