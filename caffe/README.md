# Caffe ResNet `.prototxt` file generator
Tested using Python 2.7

#### How to
1. According to your Linux distribution, install Protocol Buffer package which includes `protoc`, whose
version number should be `2.x`. `3.x` release not tested.
2. install Protocol Buffer Python package (if not shipped with `protoc`) by `pip install protobuf`.
3. Generate `caffe_pb2.py` file by `protoc --python_out=./ --proto_path=../../caffe/src/caffe/proto/ ../../caffe/src/caffe/proto/caffe.proto`, check if file is successfully generated, modify each file path accordingly.
4. Run the generator by `python resnet_caffe_gen.py`, and several network definition files should be generated.
5. Check the provided `solver_cifar.prototxt` in this directory. Modify the network definition file path in this file if necessary. `mkdir snapshot`, accordingly.
6. Write a script to call Caffe:
```
/path/to/caffe/binary
```

#### Current result
`20 layer cifar10`: 86%.  
Almost 10 hours in training one K40 GPU with ECC off (one of the GPU core of several K80 cards).
Even slower when trained with multiple GPUs using official Caffe repo.
Unfortunately, the Nvidia fork of Caffe scales to multiple GPUs in a reasonable sense, however the fork is behind upstream a lot and thus it doesn't provide a Batch Normalization layer. One is encouraged to try it out.


`54 layer cifar10`: 89%.  
The same hardware for nearly 16 hours.

#### Known Issues
1. `TODO:` generate network for `ilsvrc 12`, currently it only generates files for `cifar10` dataset, according to the descriptions provided in the paper.
2. This network **hasn't** yet achieve the stipulated accuracy in the paper. But several techniques were used:
  + Padded training image with `4 pixel` each side.
  + Used the Batch Normalization layer.
  + The standard deviation for initialization Gaussian distribution is set to `sqrt(2/(n*n*c))`, where `n` is the kernel size and `c` is the input channel size.
  + **If the learning rate is set to `0.1` as the paper stated, the network will overfit.** The only successfully trained network is using learning rate of `0.01`, even `0.05` would not work.
3. There is 3 versions of the shortcut link according to the paper, only option `B` is provided here. Since, option `A` is very ambiguous, no one in my group figured out how to perform the zero padding appropriately.
  + `A:` zero Padded
  + `B:` projection only on feature map size change
  + `C:` always project, no matter the size of input/output feature map size
