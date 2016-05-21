import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros
import numpy as np
from latte import *
from latte.solvers import sgd_update
import random

# from https://raw.githubusercontent.com/amitgroup/amitgroup/master/amitgroup/io/mnist.py
def load_mnist(dataset="training", digits=None, path=None, asbytes=False, selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST`` to point to the folder where you
    put the data, so that you don't have to select path. On a Linux+bash setup,
    this is done by adding the following to your ``.bashrc``::

        export MNIST=/path/to/mnist

    Parameters
    ----------
    dataset : str 
        Either "training" or "testing", depending on which dataset you want to
        load. 
    digits : list 
        Integer list of digits to load. The entire database is loaded if set to
        until--but not including--the twentieth.
    return_labels : bool
        Specify whether or not labels should be returned. This is also a speed
        performance if digits are not specified, since then the labels file
        does not need to be read at all.
    return_indicies : bool
        Specify whether or not to return the MNIST indices that were fetched.
        This is valuable only if digits is specified, because in that case it
        can be valuable to know how far
        in the database it reached.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. If neither labels nor inices are returned, then this is returned directly, and not inside a 1-sized tuple.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned only if ``return_labels`` is `True`, which is default.
    indices : ndarray
        The indices in the database that were returned.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training') # doctest: +SKIP

    Load 100 sevens from the testing set:    

    >>> sevens = ag.io.load_mnist('testing', digits=[7], selection=slice(0, 100), return_labels=False) # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    if path is None:
        try:
            path = os.environ['MNIST']
        except KeyError:
            raise ValueError("Unspecified path requires environment variable $MNIST to be set")

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    # We can skip the labels file only if digits aren't specified and labels aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection] 
    N = len(indices)

    images = zeros((N, rows, cols), dtype=np.uint8)

    if return_labels:
        labels = zeros((N), dtype=int8)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images.astype(np.float32),)
    if return_labels:
        ret += (labels.astype(np.float32),)
    if return_indices:
        ret += (indices,)
    if len(ret) == 1:
        return ret[0] # Don't return a tuple of one
    else:
        return ret

train_data, train_label = load_mnist(dataset="training", path="./data")
test_data, test_label   = load_mnist(dataset="testing", path="./data")

num_train = train_data.shape[0]
train_data = np.pad(train_data.reshape(num_train, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
train_label = train_label.reshape(num_train, 1)

num_test = test_data.shape[0]
test_data = np.pad(test_data.reshape(num_test, 1, 28, 28), [(0, 0), (0, 7), (0, 0), (0, 0)], mode='constant')
test_label = test_label.reshape(num_test, 1)

net = Net(100)

data     = MemoryDataLayer(net, train_data[0].shape)
label    = MemoryDataLayer(net, train_label[0].shape)

_, conv1 = ConvLayer(net, data, num_filters=24, kernel=5, stride=1, pad=2)
relu1    = ReLULayer(net, conv1)
pool1    = MaxPoolingLayer(net, relu1, kernel=2, stride=2, pad=0)

_, conv2 = ConvLayer(net, pool1, num_filters=56, kernel=5, stride=1, pad=2)
relu2    = ReLULayer(net, conv2)
pool2    = MaxPoolingLayer(net, relu2, kernel=2, stride=2, pad=0)

_, conv3 = ConvLayer(net, pool2, num_filters=56, kernel=3, stride=1, pad=1)
relu3    = ReLULayer(net, conv3)
pool3    = MaxPoolingLayer(net, relu3, kernel=2, stride=2, pad=0)

_, fc4   = FullyConnectedLayer(net, pool3, 512)
relu4    = ReLULayer(net, fc4)
_, fc5   = FullyConnectedLayer(net, relu4, 512)
relu5    = ReLULayer(net, fc5)
_, fc6   = FullyConnectedLayer(net, relu5, 16)

loss     = SoftmaxLossLayer(net, fc6, label)
acc      = AccuracyLayer(net, fc6, label)

net.compile()

params = []
for name in net.buffers.keys():
    if name.endswith("weights") and "grad_" not in name:
        ensemble_name = name[:-len("weights")]
        grad = net.buffers[ensemble_name + "grad_weights"]
        params.append((net.buffers[name],
                       grad, 
                       np.zeros_like(net.buffers[name])))
    elif name.endswith("bias") and "grad_" not in name:
        ensemble_name = name[:-len("bias")]
        grad = net.buffers[ensemble_name + "grad_bias"]
        params.append((net.buffers[name],
                       grad, 
                       np.zeros_like(net.buffers[name])))

# net.buffers[fc6.name + "bias"][11:] = 0.0
# net.buffers[_.name + "weights"][11:] = 0.0

base_lr = .01
gamma = .0001
power = .75

train_batches = [i for i in range(0, num_train, 100)]

for epoch in range(10):
    random.shuffle(train_batches)
    print("Epoch {} - Training...".format(epoch))
    for i, n in enumerate(train_batches):
        data.set_value(train_data[n:n+100])
        label.set_value(train_label[n:n+100])
        net.forward()
        if i % 100 == 0:
            print("Epoch {}, Train Iteration {} - Loss = {}".format(epoch, i, net.loss))
        net.backward()
        lr = base_lr * (1 + gamma * i)**power
        mom = .9
        for param in params:
            # expected = param[0] - (param[2] * mom + np.sum(param[1], axis=0) * lr)
            sgd_update(param[0], param[1], param[2], lr, mom)
        net.clear_values()
        net.clear_grad()
        net.loss = 0.0

    print("Epoch {} - Testing...".format(epoch))
    acc = 0
    for i, n in enumerate(range(0, num_test, 100)):
        data.set_value(test_data[n:n+100])
        label.set_value(test_label[n:n+100])
        net.test()
        acc += net.accuracy
        net.clear_values()
    acc /= (num_test / 100)
    acc *= 100
    print("Epoch {} - Validation accuracy = {:.3f}%".format(epoch, acc))
