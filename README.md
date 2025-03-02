# VIMOC-Net
Depth-Assisted Network for Indiscernible Marine Object Counting with Flow-Guided Adaptive Motion-Differentiated Feature Encoding

<div>
  <img src="./assets/images.gif" width="30%" alt="teaser" align=center style="float: left;" />
  <img src="./assets/gt.gif" width="30%" alt="teaser" align=center style="float: right;" />
  <img src="./assets/depth.gif" width="30%" alt="teaser" align=center style="float: right;" />
</div>

## Setup
Pytorch 1.10.2

Python 3.8.19

Optical flow estimates for this project are based on RAFT, which you can find [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)

## Dataset preparation 
You need to prepare datasets for following training and testing activities. [Datasets](https://drive.google.com/file/d/12vmbS7p7gIDv2-T-pJUgGJRPLi5lvh_E/view?usp=sharing)


## Pre-train models
You can download the model weights we provided [here](https://drive.google.com/file/d/1H8N6d3ugaGdzsQ1rF0Dmd-1wzgzGuVzT/view?usp=drive_link)

## Test

We recommend putting **raft-things.pth** in the *./RAFT* folder and **model_best.pth.tar** in the main folder. You can test VIMOC-Net with:

```shell
python test.py --raft_model /path/to/RAFT/raft-things
```
