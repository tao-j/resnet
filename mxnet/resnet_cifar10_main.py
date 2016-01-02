import mxnet as mx
import argparse
import os, sys
import resnet_cifar10_train as train_model

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='resnet',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='./',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='2,3',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=50000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=100,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load/save')
parser.add_argument('--num-epochs', type=int, default=300,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module(args.network + '_cifar10_net').get_symbol(10)

# data
def get_iterator(args, kv):
    data_shape = (3, 32, 32)
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_img    = args.data_dir + "mean_pad.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        mean_img    = args.data_dir + "mean_no_pad.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
