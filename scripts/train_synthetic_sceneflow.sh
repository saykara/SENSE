#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model # The directory where the models will be saved.
BASE_DIR=/content                           # Base directory for dataset
LOAD_DIR=/content/model_0040.pth            # The model that we continue training, put None at start.

### There are two main architectures for SENSE model [SENSE, SENSENeXt]
### enc-arch argument decides the architecture, [psm] for SENSE or [psmnext] for SENSENeXt

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
   --loadmodel ${LOAD_DIR} \
   --resume ${LOAD_DIR} \
   --workers 2 \
   --lr 0.0004 \
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
   
