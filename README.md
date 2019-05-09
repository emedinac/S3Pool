## S3Pool
I developed this code during my master studies (approximately January/March 2017)

# Usage
* The next line is used to run this code, the parameters configured are: lr=0.1, learning schedule apply a divison by 10 at 40,80,110,140 for an overall of 150 epochs. Models are changed manually inside the code.
```run main.py --lr=0.1 --lrsch=40,80,110,140 --epoch=150```




- Rev 2.0

* This code works now on Pytorch 1.0.1. But organization should be improved.
* Usage line is included.


- Rev 1.0

* S3Pool: Pooling with Stochastic Spatial Sampling in PyTorch.


* I tried to follow the original paper ( https://arxiv.org/abs/1611.05138 ) in PyTorch using the original ResNet (20 layers) topology described in the paper and a small version of VGG (6 layers) with 2 S3Pool blocks.

* my implementation of 6-layers VGG reports 7.97% of error. Moreover, using ResNet, errors obtained were 6.9%, 7.0% and 7.1% and paper reports 7.09%. Data augmentation with horizontal flips and crops was employed.

* I borrowed utils.py from one of all github sites that use this excellent file.
