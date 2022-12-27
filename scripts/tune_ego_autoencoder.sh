#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model/EgoAutoencoderTuned
LOAD_DIR=/content/model.pth

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
   --epochs 80 \
   --bn-type syncbn \
   --batch-size 4 \
   --print-freq 20