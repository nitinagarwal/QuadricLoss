# Emebedding 3D models with Quadric Loss 
### BMVC 2019 [[Paper]](https://arxiv.org/abs/1907.10250)[[Project Page]](https://www.ics.uci.edu/~agarwal/quadLoss/index.html)

We propose a new point-to-surface based loss function named Quadric Loss, which minimizes the quadric error between the reconstructed points and the input surface. Unlike Chamfers or L2 which are spherical losses (equidistant points have equal error), Quadric loss is a ellipsoidal loss, which penalizes displacements of points more in the normal direction thereby preserving sharp features and edges in the output reconstruction.

## Getting Started

This implementation uses Pytorch version 1.0. 

### Installation

```
## Clone the repository
git clone https://github.com/nitinagarwal/QuadricLoss
cd QuadricLoss

## create a virtual environment and install the required packages
virtualenv .quadricLoss
source .quadricLoss/bin/activate
pip install -r requirements.txt

## You are Done
```

The code has been tested with pytorch 1.0.0 and python 2.7.16

### Data

Note we use 5000 meshes from the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) and process them to obtain approximately 8064 meshes each containing 2500 vertices (for details on processing please refer to the paper). When using this dataset make sure to respect the ABC dataset license. 

Download the provided dataset. 
```
bash ./scripts/download_data.sh
```

### Training with Various Loss Functions

Launch the visdom server
```
python -m visdom.server -p 8888
```

Train with Quadric loss
```
export CUDA_VISIBLE_DEVICES=0  #whichever gpu you want to use
bash ./scripts/train.sh
```

Monitor your training on [http://localhost:8888/](http://localhost:8888/)

To train with other (or a combination of) loss functions add appropriate weights in ./scripts/train.sh.


### Recontructing Meshes using Trained Models

Change the path for the pretrained model and the output folder in ./scripts/inference.sh.

```
bash ./scripts/inference.sh
```

The reconstructed point clouds and the meshes can be visualized using [MeshLab](http://www.meshlab.net/)


## Citation
If you find this code useful, please consider citing our paper

```
@inproceedings{agarwal2019quadric,
    title={Learning Embedding of 3D models with Quadric Loss},
    author={Agarwal, Nitin and Yoon, Sung-eui and Gopi, M},
    booktitle={BMVC},
    year={2019}}
```

## License
Our code is released under MIT license (see License file for details)

## Contact
Please contact [Nitin Agarwal](https://www.ics.uci.edu/~agarwal/) if you have questions or comments
