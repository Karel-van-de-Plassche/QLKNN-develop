"""A simple MNIST classifier which displays summaries in TensorBoard.
 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed
from collections import OrderedDict
from itertools import product, chain
import collections
import json
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None


#Dataset = collections.namedtuple('Dataset', ['data', 'target'])
#Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
class Dataset():
    def __init__(self, features, target):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._features = features
        self._target = target
        assert self._features.shape[0] == self._target.shape[0]
        self._num_examples = features.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_examples(self):
        return self._num_examples
    
    def next_batch(self, batch_size, shuffle=True):
        starttime = time.time()
        start = self._index_in_epoch
        if batch_size == -1:
            batch_size = self._num_examples
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self._features.iloc[perm]
                self._target = self._target.iloc[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, 'Batch size asked bigger than number of samples'
        end = self._index_in_epoch
        batch = (self._features.iloc[start:end], self._target.iloc[start:end])
        return batch

    def to_hdf(self, file, key):
        with pd.HDFStore(file) as store:
            store.put(key + '/features', self._features)
            store.put(key + '/target', self._target)

    @classmethod
    def read_hdf(cls, file, key):
        with pd.HDFStore(file) as store:
            dataset = Dataset(store.get(key + '/features'),
                              store.get(key + '/target'))
        return dataset

    def astype(self, dtype):
        self._features = self._features.astype(dtype)
        self._target = self._target.astype(dtype)
        return self


class Datasets():
    _fields = ['train', 'validation', 'test']
    def __init__(self, **kwargs):
        for name in self._fields:
            setattr(self, name, kwargs.pop(name))
        assert(~bool(kwargs))

    def to_hdf(self, file):
        for name in self._fields:
            getattr(self, name).to_hdf(file, name)

    @classmethod
    def read_hdf(cls, file):
        datasets = {}
        for name in cls._fields:
            datasets[name] = Dataset.read_hdf(file, name)
        return Datasets(**datasets)

    def astype(self, dtype):
        for name in self._fields:
            setattr(self, name, getattr(self, name).astype(dtype))
        return self

def convert_panda(panda, frac_validation, frac_test, features_names, target_name, shuffle=True):
    total_size = panda.shape[0]
    # Dataset might be ordered. Shuffle to be sure
    if shuffle:
        panda = shuffle_panda(panda)
    validation_size = int(frac_validation * total_size)
    test_size = int(frac_test * total_size)
    train_size = total_size - validation_size - test_size

    datasets = []
    for slice_ in [panda.iloc[:train_size],
                   panda.iloc[train_size:train_size+validation_size],
                   panda.iloc[train_size+validation_size:]]:
        datasets.append(Dataset(slice_[features_names], slice_[target_name].to_frame()))

    return Datasets(train=datasets[0], validation=datasets[1], test=datasets[2])


def qualikiz_sigmoid(x, name=""):
    return tf.divide(2., tf.add(1., tf.exp(tf.multiply(-2., x)))) - 1.

def split_panda(panda, frac=0.1):
    panda1 = panda.sample(frac=frac)
    panda2_i = panda.index ^ panda1.index
    panda2 = panda.loc[panda2_i]
    return (panda1, panda2)

def generate_input_fn(panda):
    x = panda[[dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']].copy()
    y = panda['efeETG_GB'].copy()
    return pandas_io.pandas_input_fn(x, y, batch_size=8, num_epochs=1)

def shuffle_panda(panda):
    return panda.iloc[np.random.permutation(np.arange(len(panda)))]

def normalize(array):
    norm = np.linalg.norm(array[~np.isnan(array)])
    return norm

def fake_func(input):
    return np.sum(np.square(input + 0.0 * (np.random.rand(len(input)) - 0.5))) / len(input)

def fake_panda(scan_dims, train_dim):
    fakepanda = pd.DataFrame(columns = scan_dims + [train_dim])
    fakedims = {}
    npoint = 12
    fakedims = {scan_dim: np.linspace(0, 1, npoint) for scan_dim in scan_dims}
    for scan_dim in scan_dims:
        fakepanda[scan_dim] = np.full(npoint**len(scan_dims), np.nan) 
    fakepanda[train_dim] = np.full_like(fakepanda[scan_dims[0]], np.nan)
    for ii, dims in enumerate(product(*fakedims.values())):
        fakepanda.iloc[ii] = list(dims) + [fake_func(dims)]
    return fakepanda

def model_to_json(name, scale_factor, scale_bias):
    dict_ = {x.name: x.eval().tolist() for x in tf.trainable_variables()}
    dict_['scale_factor'] =  scale_factor.to_dict()
    dict_['scale_bias'] =  scale_bias.to_dict()
    with open(name, 'w') as file_:
        json.dump(dict_, file_, sort_keys=True, indent=4, separators=(',', ': '))

def timediff(start, event):
    print(event + ' reached after ' + str(time.time() - start) + 's')

def train():
    # Import data
    shuffle = True
    start = time.time()
    if os.path.exists('filtered.h5'):
        panda = pd.read_hdf('filtered.h5')
        timediff(start, 'Dataset loaded')
        scan_dims = panda.columns[:-1]
        train_dim = panda.columns[-1]
    else:
        try:
            os.remove('splitted.h5')
        except:
            pass
        panda = pd.read_hdf('efe_GB.float16.h5')
        timediff(start, 'Dataset loaded')
        scan_dims = panda.columns[:-1]
        train_dim = panda.columns[-1]
        panda = panda[panda[train_dim] > 0]
        panda = panda[panda[train_dim] < 60]
        panda = panda[np.isclose(panda['An'], 2)]
        panda = panda[np.isclose(panda['x'], 3*.15, rtol=1e-2)]
        panda = panda[np.isclose(panda['Ti_Te'], 1)]
        panda = panda[np.isclose(panda['Nustar'], 1e-2, rtol=1e-2)]
        panda = panda[np.isclose(panda['Zeffx'], 1)]
        timediff(start, 'Dataset filtered')
        panda.to_hdf('filtered.h5', 'filtered', format='t')
        timediff(start, 'Filtered saved')

    if os.path.exists('splitted.h5'):
        datasets = Datasets.read_hdf('splitted.h5')
    else:
        datasets = convert_panda(panda, 0.1, 0.1, scan_dims, train_dim, shuffle=shuffle)
        datasets.to_hdf('splitted.h5')
    datasets.astype('float64')
    timediff(start, 'Dataset split')

    # Create a multilayer model.
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(datasets.train._target.dtypes.iloc[0], [None, len(scan_dims)], name='x-input')
        y_ = tf.placeholder(x.dtype, [None, 1], name='y-input')

    #with tf.name_scope('input_normalize'):
    #    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #    tf.summary.image('input', image_shaped_input, 10)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape, **kwargs):
        """Create a weight variable with appropriate initialization."""
        #initial = tf.truncated_normal(shape, stddev=0.1)
        initial = tf.random_normal(shape, **kwargs)
        return tf.Variable(initial)

    def bias_variable(shape, **kwargs):
        """Create a bias variable with appropriate initialization."""
        #initial = tf.constant(0.1, shape=shape)
        initial = tf.random_normal(shape, **kwargs)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid, dtype=tf.float32):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim], dtype=dtype)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim], dtype=dtype)
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    nodes1 = 40
    nodes2 = 40

    scale_factor = 1 / (panda.min() + panda.max())
    scale_bias = -panda.min() * scale_factor
    in_factor = tf.constant(scale_factor[scan_dims].values, dtype=x.dtype)
    in_bias = tf.constant(scale_bias[scan_dims].values, dtype=x.dtype)

    x_scaled = in_factor * x + in_bias
    timediff(start, 'Scaling defined')
    hidden1 = nn_layer(x_scaled, len(scan_dims), nodes1, 'layer1', dtype=x.dtype)
    hidden2 = nn_layer(hidden1, nodes1, nodes2, 'layer2', dtype=x.dtype)

    #with tf.name_scope('dropout'):
    #    keep_prob = tf.placeholder(tf.float32)
    #    tf.summary.scalar('dropout_keep_probability', keep_prob)
    #    dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    out_factor = tf.constant(scale_factor[train_dim], dtype=x.dtype)
    out_bias = tf.constant(scale_bias[train_dim], dtype=x.dtype)
    y_scaled = nn_layer(hidden2, nodes2, 1, 'layer3', dtype=x.dtype)
    y = (y_scaled - out_bias) / out_factor

    #with tf.name_scope('cross_entropy'):
    #    # The raw formulation of cross-entropy,
    #    #
    #    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #    #                               reduction_indices=[1]))
    #    #
    #    # can be numerically unstable.
    #    #
    #    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    #    # raw outputs of the nn_layer above, and then average across
    #    # the batch.
    #    diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, targets=y)
    #    with tf.name_scope('total'):
    #        cross_entropy = tf.reduce_mean(diff)
    #tf.summary.scalar('cross_entropy', cross_entropy)
    timediff(start, 'NN defined')

    with tf.name_scope('MSE'):
        #with tf.name_scope('correct_prediction'):
        #    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('loss'):
            mse = tf.to_double(tf.reduce_mean(tf.square(tf.subtract(y_, y))))
            #l2_norm = tf.reduce_sum(tf.square())
            l2_norm = tf.to_double(tf.add_n([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
            #mse = tf.losses.mean_squared_error(y_, y)
            #if x.dtype == tf.float32:
            l2_loss = 1 * tf.divide(l2_norm, tf.to_double(tf.size(y)))
            #else:
                #l2_loss = 1 * tf.divide(l2_norm, tf.to_float(tf.size(y))
            loss = mse + l2_loss
            #loss = mse
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('l2_norm', l2_norm)
    tf.summary.scalar('l2_loss', l2_loss)
    tf.summary.scalar('MSE', mse)

    optimizer = None
    with tf.name_scope('train'):
        #train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        #        loss)
        #train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, 0.60).minimize(loss)
        #train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        #             loss)
        #train_step = tf.contrib.opt.python.training.external_optimizer.ScipyOptimizerInterface(loss, options={'maxiter': 100}).minimize()

        #optimizer = tf.contrib.opt.python.training.external_optimizer.ScipyOptimizerInterface(loss, options={'maxiter': 10000, 'pgtol': 1e2, 'eps': 1e-2, 'factr': 10000})
        optimizer = tf.contrib.opt.python.training.external_optimizer.ScipyOptimizerInterface(loss, options={'maxiter': 1000})
        #tf.logging.set_verbosity(tf.logging.INFO)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    tf.global_variables_initializer().run()
    timediff(start, 'Variables initialized')

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set loss, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def gen_feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = datasets.train.next_batch(1000)
            k = FLAGS.dropout
        else:
            xs, ys = datasets.test.next_batch(datasets.test.num_examples, shuffle=False)
            k = 1.0
        return {x: xs, y_: ys}

    saver = tf.train.Saver()
    epoch = 0
    timediff(start, 'Starting loss calculation')
    summary, lo = sess.run([merged, loss], feed_dict=gen_feed_dict(False))
    timediff(start, 'Algorithm started')
    print('Loss at epoch %s: %s' % (epoch, lo))
    for i in range(FLAGS.max_steps):
        if optimizer:
            feed_dict = gen_feed_dict(True)
            optimizer.minimize(sess, feed_dict=feed_dict)
            ce = loss.eval(feed_dict=feed_dict)
            summary = merged.eval(feed_dict=feed_dict)
        else:
            feed_dict = gen_feed_dict(True)
            ce, summary, _ = sess.run([loss, merged, train_step], feed_dict=feed_dict)
        print(ce)
        train_writer.add_summary(summary, i)

        if datasets.train.epochs_completed > epoch:
            num_fits = 0
            epoch = datasets.train.epochs_completed
            #print(dataset.train._index_in_epoch)
            feed_dict = gen_feed_dict(False)
            summary, lo = sess.run([merged, loss], feed_dict=feed_dict)
            test_writer.add_summary(summary, i)
            saver.save(sess, './model.ckpt')
            print('Loss at epoch %s: %s' % (epoch, lo))
            #print(dataset.train._index_in_epoch)
        #if i % 10 == 0:    # Record summaries and test-set loss
        #    summary, acc = sess.run([merged, loss], feed_dict=feed_dict(False))
        #    test_writer.add_summary(summary, i)
        #    print('Accuracy at step %s: %s' % (i, acc))
        #else:    # Record train set summaries, and train
        #    if i % 100 == 99:    # Record execution stats
        #        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #        run_metadata = tf.RunMetadata()
        #        summary, _ = sess.run([merged, train_step],
        #                              feed_dict=feed_dict(True),
        #                              options=run_options,
        #                              run_metadata=run_metadata)
        #        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        #        train_writer.add_summary(summary, i)
        #        print('Adding run metadata for', i)
        #    else:    # Record a summary
    train_writer.close()
    test_writer.close()
    
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    #xs, ys = dataset.test.next_batch(dataset.test.num_examples)
    ests = y.eval({x: xs, y_: ys})
    line_x = np.linspace(float(ys.min()), float(ys.max()))
    print("Validation RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))
    plt.plot(line_x, line_x)
    plt.scatter(ys, ests)
    plt.show()

    xs, ys = datasets.test.next_batch(-1)
    print("Test RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))
    xs, ys = datasets.train.next_batch(-1)
    print("Train RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))
    model_to_json('nn.json', scale_factor.astype('float64'), scale_bias.astype('float64'))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                                            default=True,
                                            help='If true, uses fake data for unit testing.')
    #parser.add_argument('--max_steps', type=int, default=100000,
    parser.add_argument('--max_steps', type=int, default=1000,
                                            help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=10.,
                                            help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                                            help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/test/input_data',
                                            help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/test/logs/test_with_summaries',
                                            help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
