# ResNet for mxnet

## Usage guide

#### 1. Prepare dataset according to mxnet documentation
For both ilsvrc12 and cifar10 dataset, there exists several scipts to make the required `.rec` file for mxnet to use, please consult the `example` folder of the [mxnet repo](https://github.com/shuokay/mxnet/tree/master/example) as well as the [documentation](mxnet.rtfd.org) to make or download these files and put them in to `data_ilsvrc12` and `data_cifar10` folder respectively. See the `*.sh` scripts in this folder to gain a quick insight of how to use these files.

#### 2. Train the model
##### Small network for cifar10
According to the paper, several data augmentation techniques is used the same as [caffe](../caffe) in this repo.  
With padded input and `n = 9` and total batch size 256 for two gpus, the model can achieve `86%` accuracy, which is quite far from their stated accuracy of `94%`. Help appreciated.

##### Large network for ilsvrc12 (ImageNet)
The smallest variation (50 layers) of the large network consumes the GPU memeory quite a lot. When trained on 4 gpus, 5G memory is consumed on each GPU when a total batch size of 128 is set. The training speed for this setting is around 80 images/sec, which means if we are to train this for 45 epoches (the case of alexnet), maybe 10 days are required to get the final result.  
As for the largest the network (154 layers), at a batch size of 8 on one GPU, the memory consumption is 3.8G, and training speed is 10 images/sec.

