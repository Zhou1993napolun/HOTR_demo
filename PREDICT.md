## 1. Environmental Setup
```bash
$ conda create -n kakaobrain python=3.7
$ conda install -c pytorch pytorch torchvision # PyTorch 1.7.1, torchvision 0.8.2, CUDA=11.0
$ conda install cython scipy
$ pip install pycocotools
$ pip install opencv-python
$ pip install wandb
```

## 2. Data preparation
If you only want to make predictions, there is no need to download the dataset, since all files needed are already in `data`.

If you want to train or evaluate the model, ou can refer to `README.md` for downloading the dataset.

## 3. Pretrained model
### HICO-DET dataset
| Epoch | detector | # queries |  Default(Full)  |  Rare  | Non-Rare | Checkpoint   |
|:-----:|:--------:|:---------:|:---------------:|:------:|:--------:|:------------:|
|  100  |    COCO  |     16    |      23.76      |  22.34 |   24.19  | [download](https://arena.kakaocdn.net/brainrepo/hotr/hico_q16.pth)  |
|  100  | HICO-DET |     16    |      25.73      |  21.85 |   26.89  | [download](https://arena.kakaocdn.net/brainrepo/hotr/hico_ft_q16.pth) |

Download the model and store it at `checkpoints/hico_det/`.
## 4. Inference
```bash
make hico_single_predict
```
The path of the image to inference is passed by ```--img_dir``` argument, which can be changed in `Makefile`.