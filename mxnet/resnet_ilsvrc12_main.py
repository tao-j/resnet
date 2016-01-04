import mxnet as mx
import argparse
import logging

parser = argparse.ArgumentParser(description='train an image classifer on ilsvrc12')
parser.add_argument('--network', type=str, default='resnet',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='./',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='0,1,2,3',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=64,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=40,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load/save')
parser.add_argument('--num-epochs', type=int, default=120,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local_allreduce_device',
                    help='the kvstore type')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module(args.network + '_ilsvrc12_net').get_symbol(1000, 0)

# data
def get_iterator(args, kv):
    data_shape = (3, 224, 224)
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "val.rec",
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def train_model(args, network, data_loader):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # load model?
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model?
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.0001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 50),
        epoch_end_callback = checkpoint)


# check the network graph
# g = mx.visualization.plot_network(net)
# g.format = 'png'
# g.render()

# train
train_model(args, net, get_iterator)
