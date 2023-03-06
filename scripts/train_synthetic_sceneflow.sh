#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model
BASE_DIR=/content

python tools/train_joint_synthetic_sceneflow.py pre-train \
   --dataset sceneflow \
   --enc-arch psmnext \
   --dec-arch pwcdc \
   --disp-refinement hourglass \
   --flow-dec-arch pwcdc \
   --flow-no-ppm \
   --flow-refinement none \
   --maxdisp 192 \
   --savemodel ${RES_DIR} \
   --loadmodel /content/model_0007.pth \
   --resume /content/model_0007.pth \
   --workers 2 \
   --lr 0.0002 \
   --lr-steps 70 \
   --lr-gamma 0.1 \
   --epochs 100 \
   --bn-type syncbn \
   --batch-size 8 \
   --corr-radius 4  \
   --disp-crop-imh 256 \
   --disp-crop-imw 512 \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --disp-loss-weight 0.25 \
   --print-freq 20 \
   --base-dir ${BASE_DIR} \
   --kernel-size 7 \
   --dec-kernel-size 7
   
