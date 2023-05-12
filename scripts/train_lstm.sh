#!/bin/bash

export PYTHONPATH=./

RES_DIR=/content/drive/MyDrive/Thesis/Model/LSTM/next # The directory where the models will be saved.
BASE_DIR=/content/dataset                             # Base directory for dataset
LOAD_DIR=/content/model_0040.pth                      # The model that we continue training, put None at start.
ENC_DIR='/content/model_0009.pth'                     # The Encoder-Decoder model to load (Stage 2)
FLOW_DIR='/content/model_0068.pth'                    # The Optical Flow extractor model to load (Stage 1)


### There are two main architectures for ego-motion estimation model, 
### Stage 1 works as optical flow extractor [SENSE, SENSENeXt]
### Stage 2 works as optical flow feature extractor [Enc, EncNeXt]
### Stage1 network => enc-arch argument decides the architecture, [psm] for SENSE or [psmnext] for SENSENeXt
### Stage2 network => ego-enc argument decides the architecture, [next] for nextified encoder else base encoder

python tools/train_lstm.py pre-train \
   --dataset kitti_vo \
   --enc-arch psmnext \
   --ego-enc next \
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
   --lr 0.0001 \
   --lr-steps 20 \
   --lr-gamma 0.1 \
   --epochs 60 \
   --bn-type syncbn \
   --batch-size 20 \
   --corr-radius 4  \
   --disp-crop-imh 256 \
   --disp-crop-imw 512 \
   --flow-crop-imh 384 \
   --flow-crop-imw 640 \
   --disp-loss-weight 0.25 \
   --print-freq 20 \
   --base-dir ${BASE_DIR} \
   --kernel-size 7 \
   --flow-model ${ENC_DIR} \
   --enc-model ${FLOW_DIR}