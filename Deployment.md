# README
*Deployment of HOI model HOTR on server and NUC.*

## Introduction 
### HOI
- Definition: Human Object Interaction detection. 
### HOTR
- HOTR: End-to-End Human-Object Interaction Detection with Transformers
- Reference: [Arxiv](https://arxiv.org/abs/2104.13682 'HOTR: End-to-End Human-Object Interaction Detection with Transformers') & [Github](https://github.com/kakaobrain/HOTR 'https://github.com/kakaobrain/HOTR') - CVPR 2021
- Features:  
    * One stage method. 
    * Stable and accurate output. 
    * Easy deployment. 
### Deployment 
This folder contains an additionnal method to use this algorithm on video inputs. The tests were completed on AI server and INTEL NUC devices. This is a description file of how to use this method. 

## Steps
Before the start, make sure all the preparations are done:  
- An input video under `input/`
- An empty folder to store the output `output/temp/`
- Several files: 
    * `checkpoints/hico_det/hico_q16.pth`
    * `data/hico_20160224_det/list_action.txt`
    * `data/hico_20160224_det/corre_hico.npy`

_Note that these names can be changed but the input parameters should be changed at the same time. The checkpoint file can be downloaded from [kakaobrain](https://arena.kakaocdn.net/brainrepo/hotr/hico_q16.pth 'COCO detector for HICO-DET dataset')._

Create virtual environment
```sh
conda create -n hotr python=3.8
conda activate hotr
```

Environment setup
```sh
conda install pytorch torchvision
conda install cython scipy
pip install pycocotools
pip install opencv-python
pip install wandb
```
_The versions during tests are:  
NUC: torch = `2.0.1`, torchvision = `0.15.2a0`  
AI server: torch = `1.10.0`, torchvision = `0.11.0`  
This algorithm should also be able to work on other torch versions._

Run with video input
```sh
python predict_video.py --HOIDet --share_enc --pretrained_dec --num_hoi_queries 16 --object_threshold 0 --temperature 0.2 --no_aux_loss --eval --resume checkpoints/hico_det/hico_q16.pth --dataset_file hico-det --action_list_file data/hico_20160224_det/list_action.txt --correct_path data/hico_20160224_det/corre_hico.npy --img_dir ./input --outpath output/temp/
```

Additional parameters:  
- conf_thres 
    >Confidence threshold, more 'uncertain' detections will be shown in the output if this boundary is set lower. Default is 0.33. Can be seta as `--conf_thres 0.2` to deal with dim videos which will lead to less dectections normally. 
- device
    > Use this argument when trying to define the device used by torch. Default is `cuda`, need to be set as `--device cpu` when deploying on a cpu only device *(i.e. INTEL NUC)*. 