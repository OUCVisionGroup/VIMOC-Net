# VIMOC-Net
Depth-Assisted Network for Indiscernible Marine Object Counting with Adaptive Motion-Differentiated Feature Encoding

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
The dataset comprises 50 high-deﬁnition ﬁsh videos, with each 10th frame annotated, resulting in approximately 40, 800 annotated points in total.

<div>
<img src="./assets/Dataset.png" width="90%" alt="teaser" align=center/>  
</div>

We classify them according to the target numbers and the motion rates! This classiﬁcation framework provides robust support for evaluating the model’s performance in 
indiscernible object counting across varying densities and varying motion rates.

<div>
<img src="./assets/Number.png" width="45%" alt="teaser" align=center style="float: left;"/>  
<img src="./assets/Rate.png" width="45%" alt="teaser" align=center style="float: right;"/>
</div>

The entire dataset will be provided after the paper is received! Now we present a portion of the test dataset.[Datasets](https://drive.google.com/file/d/1RoP3pD3Y-FoYOTkklnNrTIdK2QSctRKu/view?usp=drive_link)


## Pre-train models
You can download the model weights we provided [here](https://drive.google.com/file/d/1H8N6d3ugaGdzsQ1rF0Dmd-1wzgzGuVzT/view?usp=drive_link)

## Test

We recommend putting **raft-things.pth** in the *./RAFT* folder and **model_best.pth.tar** in the main folder. You can test VIMOC-Net with:

```shell
python test.py --raft_model /path/to/RAFT/raft-things
```

## Train

The training code will be available after the paper is received!
