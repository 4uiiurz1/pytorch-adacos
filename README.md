# PyTorch implementation of AdaCos, ArcFace, CosFace, and SphereFace
This repository contains code for **AdaCos**, **ArcFace**, **CosFace**, and **SphereFace**  based on [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/abs/1905.00292) implemented in PyTorch.

## TODO
- [ ] Train on WebFace and test on LFW

## Requirements
- Python 3.6
- PyTorch 1.0

## Training
### MNIST
```
python mnist_train.py --metric adacos
```
