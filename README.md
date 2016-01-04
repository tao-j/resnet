# ResNet examples
This repository contains several implementations of RenNet proposed by Microsoft Research Asia.

## Caffe
please see [caffe](caffe) folder for details
## Mxnet (recommended)
**updated mxnet** implementation on both cifar10 and ilsvrc12 dataset, which is much faster than the caffe version. See the [mxnet](/mxnet) folder for details.

##### Fun facts
For my best knowledge, the official version of the code might not be released because of their protocols. Even if it is going to be released, it will take a significant time for them to go through procedures.

It is said that the ResNet team used a very old version of Caffe which they forked quite a long while ago, accounting their huge code base change, their tool might be a completely different beast merely bears the name Caffe.
One anecdote about the team stated that their supervisor is not wiling to provide more funding for more GPU cards, that they finally end up buying high end game card instead. Also, they burned up several cards at the early stage of their project due to negligence. It is most probably that they don't even use server solutions but run on several office computers.
