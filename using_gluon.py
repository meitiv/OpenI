#!/usr/bin/env python3

import mxnet as mx
from mxnet import image, nd, autograd, gluon
import numpy as np

# context
ctx = mx.cpu()

# image transformer
def trans(img, label):
    img = image.imresize(img, 224, 224)
    img = nd.transpose(img, (2,0,1))
    img = img.astype(np.float32)
    return img, label

batch_size = 32
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset('OpenI', transform = trans),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset('OpenI', transform = trans),
    batch_size=batch_size, shuffle=False, last_batch='discard')

# build the CNN
net = gluon.nn.Sequential()
with net.name_scope():
    # first layer
    net.add(gluon.nn.Conv2D(channels = 32,
                            kernel_size = 5,
                            strides = (3,3),
                            activation = 'relu'))
    net.add(gluon.nn.MaxPool2D(pool_size = 2,
                               strides = 2))
    # second layer
    net.add(gluon.nn.Conv2D(channels = 32,
                            kernel_size = 5,
                            activation = 'relu'))
    net.add(gluon.nn.MaxPool2D(pool_size = 2,
                               strides = 2))
    # third layer
    net.add(gluon.nn.Conv2D(channels = 64,
                            kernel_size = 3,
                            activation = 'relu'))
    net.add(gluon.nn.MaxPool2D(pool_size = 2,
                               strides = 2))
    # flatten and apply fullly connected layers
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dropout(0.5))
    net.add(gluon.nn.Dense(1))

# init trainer
net.collect_params().initialize(mx.init.Xavier(magnitude = 2.24),
                                ctx = ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': .05})
# rename for convenience
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# accuracy computation
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        
    return acc.get()[1]

# training loop
epochs = 5
smoothing_constant = 0.01
for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        curr_loss = nd.mean(loss).asscalar()
        if i == 0 and e == 0:
            moving_loss = curr_loss
        else:
            moving_loss = (1 - smoothing_constant)*moving_loss +\
                          smoothing_constant*curr_loss
    # accuracy
    train_accuracy = evaluate_accuracy(train_data, net)
    test_accuracy = evaluate_accuracy(test_data, net)
    print ('Train accuracy:{}, test accuracy:{}, moving loss:{}'.\
           format(train_accuracy, test_accuracy, moving_loss))
                   
