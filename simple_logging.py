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

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed
from collections import OrderedDict
from itertools import product, chain
import collections

FLAGS = None


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
class DataSet():
    def __init__(self, features, target):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._features = features
        self._target = target.to_frame()
        assert self._features.shape[0] == self._target.shape[0]
        self._num_examples = features.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features.iloc[perm]
            self._target = self._target.iloc[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features.iloc[start:end], self._target.iloc[start:end]

def convert_panda(panda, frac_validation, frac_test, features_names, target_name):
    total_size = panda.shape[0]
    # Dataset might be ordered. Shuffle to be sure
    panda = shuffle_panda(panda)
    validation_size = int(frac_validation * total_size)
    test_size = int(frac_test * total_size)
    train_size = total_size - validation_size - test_size

    datasets = []
    for slice_ in [panda.iloc[:train_size],
                   panda.iloc[train_size:train_size+validation_size],
                   panda.iloc[train_size+validation_size:]]:
        datasets.append(DataSet(slice_[features_names], slice_[target_name]))

    return Datasets(train=datasets[0], validation=datasets[1], test=datasets[2])



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
    npoint = 60
    fakedims = {scan_dim: np.linspace(0, 1, npoint) for scan_dim in scan_dims}
    for scan_dim in scan_dims:
        fakepanda[scan_dim] = np.full(npoint**len(scan_dims), np.nan) 
    fakepanda[train_dim] = np.full_like(fakepanda[scan_dims[0]], np.nan)
    for ii, dims in enumerate(product(*fakedims.values())):
        fakepanda.iloc[ii] = list(dims) + [fake_func(dims)]
    return fakepanda

def train():
    # Import data
    ds = xr.open_dataset('/mnt/hdd/4D.nc')
    #ds = ds.sel(smag=0, Ti_Te=1, method='nearest')
    scan_dims = [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']
    train_dim = 'efe_GB'
    df = ds.drop([coord for coord in ds.coords if coord not in ds.dims]).drop('kthetarhos').to_dataframe()
    scan_dims = scan_dims[:2]
    panda = fake_panda(scan_dims, train_dim)
    panda_test, panda_train = split_panda(panda, frac=0.2)
    dataset = convert_panda(panda, 0.1, 0.1, scan_dims, train_dim)
    embed()

    sess = tf.InteractiveSession()
    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, len(scan_dims)], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

    #with tf.name_scope('input_reshape'):
    #    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #    tf.summary.image('input', image_shaped_input, 10)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
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

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, len(scan_dims), 20, 'layer1')
    hidden2 = nn_layer(hidden1, 20, 20, 'layer2')

    #with tf.name_scope('dropout'):
    #    keep_prob = tf.placeholder(tf.float32)
    #    tf.summary.scalar('dropout_keep_probability', keep_prob)
    #    dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(hidden2, 20, 1, 'layer3', act=tf.sigmoid)

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, targets=y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

    with tf.name_scope('accuracy'):
        #with tf.name_scope('correct_prediction'):
        #    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = dataset.train.next_batch(10)
            k = FLAGS.dropout
        else:
            xs = panda_test[scan_dims]
            ys = np.atleast_2d(panda_test[train_dim]).T
            k = 1.0
        return {x: xs, y_: ys}

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:    # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:    # Record train set summaries, and train
            if i % 100 == 99:    # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:    # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


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
    parser.add_argument('--max_steps', type=int, default=1000,
                                            help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                                            help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                                            help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/test/input_data',
                                            help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/test/logs/test_with_summaries',
                                            help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
