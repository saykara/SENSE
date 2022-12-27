#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model/EgoAutoencoder

python tools/train_ego_autoencoder.py pre-train \
   --dataset sceneflow \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --savemodel ${RES_DIR} \
   --workers 2 \
   --lr 0.0001 \
   --lr-steps 70 \
   --lr-gamma 0.1 \
   --epochs 10 \
   --bn-type syncbn \
   --batch-size 48 \
   --print-freq 20