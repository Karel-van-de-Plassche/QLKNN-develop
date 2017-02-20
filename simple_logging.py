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
import io

import tensorflow as tf
from tensorflow.contrib import opt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed
from collections import OrderedDict
from itertools import product, chain
import collections
import json
from run_model import QuaLiKizNDNN

FLAGS = None

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

def error_scatter(target, estimate):
    plt.figure()
    plt.scatter(target, estimate)
    line_x = np.linspace(float(target.min()), float(target.max()))
    plt.plot(line_x, line_x)
    #plt.scatter(real, estimate)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def slice_plotter(features, target, estimate):
    plt.figure()
    plt.scatter(features, target)
    plt.plot(features, estimate)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

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
    return tf.divide(tf.constant(2, x.dtype),
                     tf.add(tf.constant(1, x.dtype),
                            tf.exp(tf.multiply(tf.constant(-2, x.dtype), x)))) - tf.constant(1, x.dtype)

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

def model_to_json(name, feature_names, target_names, train_set, scale_factor, scale_bias):
    dict_ = {x.name: x.eval().tolist() for x in tf.trainable_variables()}
    dict_['prescale_factor'] =  scale_factor.to_dict()
    dict_['prescale_bias'] =  scale_bias.to_dict()
    dict_['feature_min'] = dict(train_set._features.min())
    dict_['feature_max'] = dict(train_set._features.max())
    dict_['feature_names'] = feature_names
    dict_['target_names'] = target_names

    with open(name, 'w') as file_:
        json.dump(dict_, file_, sort_keys=True, indent=4, separators=(',', ': '))

def timediff(start, event):
    print(event + ' reached after ' + str(time.time() - start) + 's')

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

def nn_layer(input_tensor, output_dim, layer_name, act=qualikiz_sigmoid, dtype=tf.float32):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    input_dim = input_tensor.get_shape().as_list()[1]
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
        #panda = panda[np.isclose(panda['An'], 2)]
        #panda = panda[np.isclose(panda['x'], 3*.15, rtol=1e-2)]
        #panda = panda[np.isclose(panda['Ti_Te'], 1)]
        #panda = panda[np.isclose(panda['Nustar'], 1e-2, rtol=1e-2)]
        #panda = panda[np.isclose(panda['Zeffx'], 1)]
        timediff(start, 'Dataset filtered')
        panda.to_hdf('filtered.h5', 'filtered', format='t')
        timediff(start, 'Filtered saved')

    if os.path.exists('splitted.h5'):
        datasets = Datasets.read_hdf('splitted.h5')
    else:
        datasets = convert_panda(panda, 0.1, 0.1, scan_dims, train_dim, shuffle=shuffle)
        datasets.to_hdf('splitted.h5')

    # Convert back to float64 for tensorflow compatibility
    datasets.astype('float64')
    timediff(start, 'Dataset split')

    # Get a slice of the data to visualize convergence
    slice_ =  panda[np.isclose(panda['qx'], 1.5, rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['smag'], .7, rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['Ti_Te'], 1, rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['Ate'], slice_['Ati'], rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['An'], 2)]
    slice_ = slice_[np.isclose(slice_['x'], 3*.15, rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['Ti_Te'], 1)]
    slice_ = slice_[np.isclose(slice_['Nustar'], 1e-2, rtol=1e-2)]
    slice_ = slice_[np.isclose(slice_['Zeffx'], 1)]

    # Start tensorflow session
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(datasets.train._target.dtypes.iloc[0], [None, len(scan_dims)], name='x-input')
        y_ = tf.placeholder(x.dtype, [None, 1], name='y-input')

    nodes1 = 30
    nodes2 = 30
    nodes3 = 30

    # Scale all input between -1 and 1
    scale_factor = 1 / (panda.min() + panda.max())
    scale_bias = -panda.min() * scale_factor
    in_factor = tf.constant(scale_factor[scan_dims].values, dtype=x.dtype)
    in_bias = tf.constant(scale_bias[scan_dims].values, dtype=x.dtype)

    x_scaled = in_factor * x + in_bias
    timediff(start, 'Scaling defined')

    # Define neural network
    hidden1 = nn_layer(x_scaled, nodes1, 'layer1', dtype=x.dtype)
    hidden2 = nn_layer(hidden1, nodes2, 'layer2', dtype=x.dtype)
    hidden3 = nn_layer(hidden2, nodes3, 'layer3', dtype=x.dtype)

    #with tf.name_scope('dropout'):
    #    keep_prob = tf.placeholder(tf.float32)
    #    tf.summary.scalar('dropout_keep_probability', keep_prob)
    #    dropped = tf.nn.dropout(hidden1, keep_prob)

    # Scale all output between -1 and 1
    out_factor = tf.constant(scale_factor[train_dim], dtype=x.dtype)
    out_bias = tf.constant(scale_bias[train_dim], dtype=x.dtype)
    y_scaled = nn_layer(hidden3, 1, 'layer4', dtype=x.dtype)
    y = (y_scaled - out_bias) / out_factor

    timediff(start, 'NN defined')

    # Define loss functions
    with tf.name_scope('Loss'):
        with tf.name_scope('mse'):
            mse = tf.to_double(tf.reduce_mean(tf.square(tf.subtract(y_, y))))
            tf.summary.scalar('MSE', mse)
        with tf.name_scope('l2'):
            l2_scale = tf.Variable(.7, dtype=x.dtype, trainable=False)
            #l2_norm = tf.reduce_sum(tf.square())
            l2_norm = tf.to_double(tf.add_n([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
            #mse = tf.losses.mean_squared_error(y_, y)
            l2_loss = l2_scale * tf.divide(l2_norm, tf.to_double(tf.size(y)))
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('l2_scale', l2_scale)
            tf.summary.scalar('l2_loss', l2_loss)
        loss = mse + l2_loss
        tf.summary.scalar('loss', loss)

    optimizer = None
    # Define fitting algorithm. Kept old algorithms commented out.
    with tf.name_scope('train'):
        #train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
        #train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, 0.60).minimize(loss)
        #train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
        #optimizer = opt.ScipyOptimizerInterface(loss, options={'maxiter': 10000, 'pgtol': 1e2, 'eps': 1e-2, 'factr': 10000})
        optimizer = opt.ScipyOptimizerInterface(loss, options={'maxiter': 1000})
        #tf.logging.set_verbosity(tf.logging.INFO)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    # Define scatter plots
    error_scatter_buf_ph = tf.placeholder(tf.string)
    error_scatter_image = tf.image.decode_png(error_scatter_buf_ph, channels=4)
    error_scatter_image = tf.expand_dims(error_scatter_image, 0)
    error_scatter_summaries = []
    # Define slice plots
    slice_buf_ph = tf.placeholder(tf.string)
    slice_image = tf.image.decode_png(slice_buf_ph, channels=4)
    slice_image = tf.expand_dims(slice_image, 0)
    slice_summaries = []
    # Add images to tensorboard
    num_image = 0
    max_images = 8
    for ii in range(max_images):
        error_scatter_summaries.append(tf.summary.image('error_scatter_' + str(ii) , error_scatter_image, max_outputs=1))
        slice_summaries.append(tf.summary.image('slice_' + str(ii) , slice_image, max_outputs=1))

    # Initialze writers and variables
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
    tf.global_variables_initializer().run()
    timediff(start, 'Variables initialized')

    epoch = 0

    timediff(start, 'Starting loss calculation')
    batch_size = 1000
    xs, ys = datasets.train.next_batch(batch_size)
    feed_dict = {x: xs, y_: ys}
    summary, lo = sess.run([merged, loss], feed_dict=feed_dict)
    timediff(start, 'Algorithm started')
    print('Loss at epoch %s: %s' % (epoch, lo))

    # Define variables for early stopping
    not_improved = 0
    best_mse = np.inf
    early_stop = 5
    best_mse_checkpoint = None
    saver = tf.train.Saver(max_to_keep=early_stop)
    try:
        for i in range(FLAGS.max_steps):
            xs, ys = datasets.train.next_batch(batch_size)
            feed_dict = {x: xs, y_: ys}
            if optimizer:
                optimizer.minimize(sess, feed_dict=feed_dict)
                ce = loss.eval(feed_dict=feed_dict)
                summary = merged.eval(feed_dict=feed_dict)
            else:
                ce, summary, _ = sess.run([loss, merged, train_step], feed_dict=feed_dict)
            print(ce)
            train_writer.add_summary(summary, i)

            # Write figures, summaries and check early stopping each epoch
            if datasets.train.epochs_completed > epoch:
                epoch = datasets.train.epochs_completed
                xs, ys = datasets.test.next_batch(-1, shuffle=False)
                feed_dict = {x: xs, y_: ys}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, lo, meanse = sess.run([merged, loss, mse], feed_dict=feed_dict, options=run_options,
                                               run_metadata=run_metadata)
                test_writer.add_summary(summary, i)
                test_writer.add_run_metadata(run_metadata, 'step%d' % i)

                save_path = saver.save(sess, './model.ckpt', global_step=i)

                # Early stepping, check if MSE is better
                if meanse < best_mse:
                    best_mse = meanse
                    not_improved = 0
                else:
                    not_improved += 1
                # Write image summaries
                xs, ys = datasets.validation.next_batch(-1, shuffle=False)
                ests = y.eval({x: xs, y_: ys})
                feed_dict = {error_scatter_buf_ph: error_scatter(ys, ests).getvalue()}
                summary = sess.run(error_scatter_summaries[num_image], feed_dict=feed_dict)
                test_writer.add_summary(summary, i)

                # Write checkpoint of NN
                model_to_json('nn_checkpoint.json', scan_dims.values.tolist(), [train_dim], datasets.train, scale_factor.astype('float64'), scale_bias.astype('float64'))
                # Use checkpoint of NN to plot slice
                nn = QuaLiKizNDNN.from_json('nn_checkpoint.json')
                fluxes = nn.get_output(**slice_)
                feed_dict = {slice_buf_ph: slice_plotter(slice_['Ate'], slice_[train_dim], fluxes).getvalue()}
                summary = sess.run(slice_summaries[num_image], feed_dict=feed_dict)
                test_writer.add_summary(summary, i)

                num_image += 1
                if num_image % max_images == 0:
                    num_image = 0
                print('Loss at epoch %s: %s' % (epoch, lo))
                # If not improved in 'early_stop' epoch, stop
                if not_improved >= early_stop:
                    print('Not improved for %s epochs, stopping..' % (early_stop))
                    saver.restore(sess, saver.last_checkpoints[0])
                    model_to_json('nn.json', scan_dims.values.tolist(), [train_dim], datasets.train, scale_factor.astype('float64'), scale_bias.astype('float64'))
                    break
    except KeyboardInterrupt:
        print('Stopping')

    train_writer.close()
    test_writer.close()
    
    # Finally, check against validation set
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    ests = y.eval({x: xs, y_: ys})
    line_x = np.linspace(float(ys.min()), float(ys.max()))
    print("Validation RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))
    plt.plot(line_x, line_x)
    plt.scatter(ys, ests)

    # And to be sure, test against test and train set
    xs, ys = datasets.test.next_batch(-1, shuffle=False)
    print("Test RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))
    xs, ys = datasets.train.next_batch(-1, shuffle=False)
    print("Train RMS error: " + str(np.sqrt(mse.eval({x: xs, y_: ys}))))

    plt.show()


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
    parser.add_argument('--max_steps', type=int, default=sys.maxsize,
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
