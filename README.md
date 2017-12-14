# S3Pool
S3Pool: Pooling with Stochastic Spatial Sampling in PyTorch.

I tried to reproduce the original paper ( https://arxiv.org/abs/1611.05138 ) in PyTorch using a small version of VGG (6 layers) with 2 S3Pool blocks and the original ResNet topology described in the paper.

Also, using ResNet, errors obtained were 6.9%, 7.0% and 7.1% and paper reports 7.09%. Data augmentation with horizontal flips and crops was employed.

I borrowed utils.py from one of all github sites that use this excellent file.
