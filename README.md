# SENSE: a Shared Encoder for Scene Flow Estimation
PyTorch implementation of our ICCV 2019 Oral paper [SENSE: A Shared Encoder for Scene-flow Estimation](https://arxiv.org/pdf/1910.12361.pdf).

<p align="center">
  <img src="sense.png" width="500" />
</p>

## License

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Requirements
* Python (tested with Python3.8.16)
* PyTorch (tested with 11.6.0)
* SynchronizedBatchNorm (borrowed from https://github.com/CSAILVision/semantic-segmentation-pytorch)
* tensorboardX
* tqdm
* OpenCV
* scipy
* numpy

## Preparing Dataset

Because our model is versatile, it uses a lot of large data. Therefore, approximately **1.1 TB** of storage space is required for datesets. We gathered up all dataset archives in one google drive, you can access it with following link: https://drive.google.com/drive/folders/1yHMV9gm4o3anGITy1YV8HZwHRUCn4kUF?usp=sharing.  

```
  pip install gdown
```
### FlyingThings3D_subset download scripts:
```
  gdown https://drive.google.com/file/d/1DCJVazDCDDPFFE2SboybKhKUVvTCnJsD/view?usp=share_link
  gdown https://drive.google.com/file/d/1-2Z8gCOjp4BaLZdTQwKTRB-sHEevox1W/view?usp=share_link
  gdown https://drive.google.com/file/d/17oSTItHBy6HOBKnhTyKFN5i2Dh4Hmu8f/view?usp=share_link
  gdown https://drive.google.com/file/d/1-56X1eJMJxFlszTxVQUeaOfzmKU6T7N1/view?usp=share_link
  gdown https://drive.google.com/file/d/1-w5oYYvnPHf2dbKrVBrHeuJf0GiLevPa/view?usp=share_link
```
### Monkaa download scripts:
```
  gdown https://drive.google.com/file/d/1-QfVPh6qDXfJeDfKHJ_j_wco5bZTthEn/view?usp=share_link
  gdown https://drive.google.com/file/d/1-VmZ8gM87giEKPYy2Se75e4FlDT_ZS7z/view?usp=share_link
  gdown https://drive.google.com/file/d/1-s2FRMyZf-YicIbqDGuuYxMS-xQdLZj1/view?usp=share_link
```
### Driving download scripts:
```
  gdown https://drive.google.com/file/d/1-3QEO0ZVxBgEF5tSoB2fPpRIpXdb2zOY/view?usp=share_link
  gdown https://drive.google.com/file/d/1-JcyuA5vVq7GZdJx7_bs3ynWwjkQ41vf/view?usp=share_link
  gdown https://drive.google.com/file/d/1-D18UuJOmP83gYzUIyyNHmH1hzdcjQ9N/view?usp=share_link
```
### MPI_Sintel download scripts:
```
  gdown https://drive.google.com/file/d/102dFyM1iV42t6BJQxSipT9d-R43Sltv7/view?usp=share_link
  gdown https://drive.google.com/file/d/10BSBCbzyWzid0Jo61qJ9ak8En7dSf7rc/view?usp=share_link
```
### KITTI_VO download scripts:
```
  gdown https://drive.google.com/file/d/1e08fgGRHjcm5P4j0L6LRuCkMz_PJbQJ4/view?usp=share_link
  gdown https://drive.google.com/file/d/1-3JU2P6OSaudYCTR4Cy5c4utAJXF9KUd/view?usp=share_link
```
### Malaga download scripts:
```
  gdown https://drive.google.com/file/d/16Us_VYkZxedJ1nQFsegY4Py8m0b1GtjT/view?usp=share_link
```

## File Structure
Our expected dataset structure as follows:

```
<Base_dir>
|-- SceneFlow
|   |-- Driving
|   |   |-- disparity
|   |   |-- frames_cleanpass
|   |   |-- optical_flow
|   |-- FlyingThings3D_subset
|   |   |-- train
|   |   |   |-- disparity
|   |   |   |-- disparity_occlusions
|   |   |   |-- flow
|   |   |   |-- flow_occlusions
|   |   |   |-- image_clean
|   |   |-- val
|   |   |   |-- disparity
|   |   |   |-- disparity_occlusions
|   |   |   |-- flow
|   |   |   |-- flow_occlusions
|   |   |   |-- image_clean
|   |-- Monkaa
|   |   |-- disparity
|   |   |-- frames_cleanpass
|   |   |-- optical_flow
|-- MPI_Sintel
|   |-- training
|   |-- test
|   |-- stereo
|   |   |-- training
|-- kitti_vo
|   |-- dataset
|   |   |-- poses
|   |   |-- sequences
|   |   |   |-- 00
|   |   |   |-- 01
|   |   |   |-- ..
|   |   |   |-- 22
|-- malaga
|   |-- malaga-urban-dataset-extract-01
|   |-- malaga-urban-dataset-extract-02
|   |-- ..
|   |-- malaga-urban-dataset-extract-15
```

## Pre-Installations 

```
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu112
  pip install timm
  pip install scikit-image
  pip install opencv-python
  pip install sklearn
  pip install joblib
  pip install matplotlib
  pip install ninja
```

To compile the correlation package, run `sh scripts/install.sh`. 

## Training

All hyper-parameters can be configured by training script arguments, you can access them under the **scripts** directory. The most important part is providing the base directory of datasets. Open *scripts/train_synthetic_sceneflow.sh* file and change the BASE_DIR variable according to <Base_dir> name you set before. 

After everthing is configured, run `sh scripts/train_synthetic_sceneflow.sh` command to start training. It will take a little while to cache the dataset when running it the first time. 


The log message structure of losses as follows; 

'Train',  global step, epoch, batch id, total batch, overall loss, 
		flow loss, flow occ loss, disp loss, disp occ loss, batch process time, 
        elapsed time from beginning, learning rate


'Test', global step, epoch, total flow loss, total flow occ loss,
		total disp loss, total_test_err_pct, total disp occ loss,
			  flow loss, flow occ loss, disp loss, disp occ loss, elapsed time, learning rate