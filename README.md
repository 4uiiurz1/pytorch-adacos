# PyTorch implementation of AdaCos, ArcFace, CosFace, and SphereFace
This repository contains code for **AdaCos**, **ArcFace**, **CosFace**, and **SphereFace**  based on [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/abs/1905.00292) implemented in PyTorch.

## TODO
- [x] Omniglot
- [x] Results of Omniglot experiments
- [ ] Train on WebFace and test on LFW


## Requirements
- Python 3.6
- PyTorch 1.0

## Training
### MNIST
```
python mnist_train.py --metric adacos
```
### Omniglot
```
cd omniglot
. download.sh
cd ..
python omniglot_train.py --metric adacos
```

## Results
### Omniglot
| Method                  |   acc@1   |   acc@5   |
|:------------------------|:---------:|:---------:|
| SphereFace              |   89.66   |   98.48   |
| CosFace                 |   89.68   |   98.23   |
| ArcFace                 |   89.54   |   98.48   |
| AdaCos                  | **90.06** | **98.55** |
