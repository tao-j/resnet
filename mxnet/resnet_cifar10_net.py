import mxnet as mx

def Conv_BN_ReLU(data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, workspace=512, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def Conv_BN(data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, workspace=512, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    return bn

def Conv(data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, workspace=512, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    return conv

def Neck(data, num_filter, layer_idx, project=False):

    # first layer
    layer_idx += 1
    if project:
        proj = mx.symbol.Convolution(data=data, workspace=512, num_filter=num_filter,
                                     kernel=(1, 1), stride=(2, 2), pad=(0, 0), name='proj_%d' %layer_idx)
        block1_stride = (2, 2)
    else:
        proj = data
        block1_stride = (1, 1)

    block1 = Conv_BN_ReLU(data, num_filter, kernel=(3, 3), stride=block1_stride, pad=(1, 1), name=str(layer_idx))

    # second layer
    layer_idx += 1
    block2 = Conv(block1, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=str(layer_idx))
    # block2 = Conv_BN(block1, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=str(layer_idx))

    esum = mx.symbol.ElementWiseSum(proj, block2)
    bn = mx.symbol.BatchNorm(data=esum, name='bn_%d' %layer_idx)
    relu = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%d' %layer_idx)
    # relu = mx.symbol.Activation(data=esum, act_type='relu', name='relu_%d' %layer_idx)

    return layer_idx, relu


def get_symbol(num_classes=10, n_const=9):

    data = mx.symbol.Variable(name='data')

    layer_idx = 0
    num_filter = 16 # 32, 64
    neck = Conv_BN_ReLU(data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=str(layer_idx))

    layer_sizes = [num_filter, 32, 64]
    project_enablers = [False, True, True]

    for num_filter, project_enable in zip(layer_sizes, project_enablers):
        for n in range(n_const):
            if n == 0 and project_enable:
                project = True
            else:
                project = False

            layer_idx, neck = Neck(data=neck, num_filter=num_filter, layer_idx=layer_idx, project=project)

    layer_idx += 1
    avg = mx.symbol.Pooling(data=neck, kernel=(2, 2), stride=(1, 1), name='global_pool', pool_type='avg')
    flatten = mx.sym.Flatten(data=avg, name="flatten")
    fc0 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc0')
    softmax = mx.symbol.SoftmaxOutput(data=fc0, name='softmax')

    print(layer_idx+1)
    return softmax
