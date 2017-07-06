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
from shutil import copyfile
import subprocess

import tensorflow as tf
from tensorflow.contrib import opt
from tensorflow.python.client import timeline


import numpy as np
import pandas as pd
from IPython import embed
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
        start = self._index_in_epoch
        if batch_size == -1:
            batch_size = self._num_examples
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            # TODO: Use panda_shuffle function
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self._features.iloc[perm]
                self._target = self._target.iloc[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, \
                'Batch size asked bigger than number of samples'
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
        assert ~bool(kwargs)

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

def convert_panda(panda, frac_validation, frac_test, features_names,
                  target_name):
    total_size = panda.shape[0]
    # Dataset might be ordered. Shuffle to be sure
    panda = shuffle_panda(panda)
    validation_size = int(frac_validation * total_size)
    test_size = int(frac_test * total_size)
    train_size = total_size - validation_size - test_size

    datasets = []
    for slice_ in [panda.iloc[:train_size],
                   panda.iloc[train_size:train_size + validation_size],
                   panda.iloc[train_size + validation_size:]]:
        datasets.append(Dataset(slice_[features_names],
                                slice_[target_name].to_frame()))

    return Datasets(train=datasets[0],
                    validation=datasets[1],
                    test=datasets[2])

def split_panda(panda, frac=0.1):
    panda1 = panda.sample(frac=frac)
    panda2_i = panda.index ^ panda1.index
    panda2 = panda.loc[panda2_i]
    return (panda1, panda2)


def shuffle_panda(panda):
    return panda.iloc[np.random.permutation(np.arange(len(panda)))]


def model_to_json(name, feature_names, target_names,
                  train_set, scale_factor, scale_bias, l2_scale):
    dict_ = {x.name: x.eval().tolist() for x in tf.trainable_variables()}
    dict_['prescale_factor'] = scale_factor.to_dict()
    dict_['prescale_bias'] = scale_bias.to_dict()
    dict_['feature_min'] = dict(train_set._features.min())
    dict_['feature_max'] = dict(train_set._features.max())
    dict_['feature_names'] = feature_names
    dict_['target_names'] = target_names
    dict_['target_min'] = dict(train_set._target.min())
    dict_['target_max'] = dict(train_set._target.max())

    sp_result = subprocess.run('git rev-parse HEAD',
                               stdout=subprocess.PIPE,
                               shell=True,
                               check=True)
    nn_version = sp_result.stdout.decode('UTF-8').strip()
    metadata = {
        'nn_develop_version': nn_version,
        'activation': 'tanh(x)',
        'c_L2': l2_scale.eval()
    }
    dict_['_metadata'] = metadata

    with open(name, 'w') as file_:
        json.dump(dict_, file_, sort_keys=True,
                  indent=4, separators=(',', ': '))


def timediff(start, event):
    print('{:35} {:5.0f}s'.format(event + ' after', time.time() - start))


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
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, output_dim, layer_name, act=tf.tanh,
             dtype=tf.float32, debug=False):
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
            if debug:
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], dtype=dtype)
            if debug:
                variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            if debug:
                tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        if debug:
            tf.summary.histogram('activations', activations)
        return activations


def normab(panda, a, b):
    return ((b - a) / (panda.max() - panda.min()),
            (b - a) * panda.min() / (panda.max() - panda.min()) + a)


def train():
    # Import data
    start = time.time()
    # Get train dimension from path name
    train_dim = os.path.basename(os.getcwd())
    train_dim = 'efe_GB'
    # Use pre-existing filtered dataset, or extract from big dataset
    if os.path.exists('filtered.h5'):
        panda = pd.read_hdf('filtered.h5')
    else:
        store = pd.HDFStore('filtered_everything_nions0.h5', 'r')
        panda = store.select(train_dim)
        input = store.select('input')
        try:
            del input['nions']  # Delete leftover artifact from dataset split
        except KeyError:
            pass
        input = input.loc[panda.index]
        panda = pd.concat([input, panda], axis=1)
        timediff(start, 'Dataset loaded')
    timediff(start, 'Dataset filtered')

    # Use pre-existing splitted dataset, or split in train, validation and test
    train_dim = panda.columns[-1]
    scan_dims = panda.columns[:-1]
    if os.path.exists('splitted.h5'):
        datasets = Datasets.read_hdf('splitted.h5')
    else:
        datasets = convert_panda(panda, 0.1, 0.1, scan_dims, train_dim)
        datasets.to_hdf('splitted.h5')
    # Convert back to float64 for tensorflow compatibility
    datasets.astype('float64')
    timediff(start, 'Dataset split')

    """
    # Get a (random) slice of the data to visualize convergence
    # TODO: Make general
    slice_dict = {
        'qx': 1.5,
        'smag': .7,
    #    'Ti_Te': 1,
        'An': 2,
    #    'x': 3 * .15,
    #    'Nustar': 1e-2,
        'Zeffx': 1}

    slice_ = panda
    for col in slice_:
        try:
            slice_ = slice_[np.isclose(slice_[col],
                                       slice_dict[col], rtol=1e-2)]
        except:
            pass
    #slice_ = slice_[np.isclose(slice_['Ate'], slice_['Ati'], rtol=1e-2)]
    """

    # Start tensorflow session
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(datasets.train._target.dtypes.iloc[0],
                           [None, len(scan_dims)], name='x-input')
        y_ds = tf.placeholder(x.dtype, [None, 1], name='y-input')

    # Define NN structure
    hidden_neurons = [30, 30, 30]

    # Scale all input between -1 and 1
    with tf.name_scope('normalize'):
        scale_factor, scale_bias = normab(panda, 0, 1)
        in_factor = tf.constant(scale_factor[scan_dims].values, dtype=x.dtype)
        in_bias = tf.constant(scale_bias[scan_dims].values, dtype=x.dtype)

        x_scaled = in_factor * x + in_bias
    timediff(start, 'Scaling defined')

    # Define neural network
    layers = [x_scaled]
    for ii, neurons in enumerate(hidden_neurons):
        layers.append(nn_layer(layers[-1], neurons, 'layer' + str(i), dtype=x.dtype))

    #with tf.name_scope('dropout'):
    #    keep_prob = tf.placeholder(tf.float32)
    #    tf.summary.scalar('dropout_keep_probability', keep_prob)
    #    dropped = tf.nn.dropout(hidden1, keep_prob)

    # All output scaled between -1 and 1, denomalize it
    y_scaled = nn_layer(layers[-1], 1, 'layer4', dtype=x.dtype)
    with tf.name_scope('denormalize'):
        out_factor = tf.constant(scale_factor[train_dim], dtype=x.dtype)
        out_bias = tf.constant(scale_bias[train_dim], dtype=x.dtype)
        y = (y_scaled - out_bias) / out_factor

    timediff(start, 'NN defined')

    # Define loss functions
    with tf.name_scope('Loss'):
        with tf.name_scope('mse'):
            mse = tf.to_double(tf.reduce_mean(tf.square(tf.subtract(y_ds, y))))
            tf.summary.scalar('MSE', mse)
        with tf.name_scope('l2'):
            l2_scale = tf.Variable(.7, dtype=x.dtype, trainable=False)
            #l2_norm = tf.reduce_sum(tf.square())
            l2_norm = tf.to_double(
                tf.add_n([tf.reduce_sum(tf.square(var))
                          for var in tf.trainable_variables()]))
            #mse = tf.losses.mean_squared_error(y_, y)
            # TODO: Check normalization
            l2_loss = l2_scale * tf.divide(l2_norm, tf.to_double(tf.size(y)))
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('l2_scale', l2_scale)
            tf.summary.scalar('l2_loss', l2_loss)
        #loss = mse
        loss = mse + l2_loss
        tf.summary.scalar('loss', loss)

    optimizer = None
    train_step = None
    # Define fitting algorithm. Kept old algorithms commented out.
    with tf.name_scope('train'):
        #train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
        #train_step = tf.train.AdadeltaOptimizer(
        #    FLAGS.learning_rate, 0.60).minimize(loss)
        #train_step = tf.train.RMSPropOptimizer(
        #    FLAGS.learning_rate).minimize(loss)
        #train_step = tf.train.GradientDescentOptimizer(
        #    FLAGS.learning_rate).minimize(loss)
        optimizer = opt.ScipyOptimizerInterface(loss,
                                                options={'maxiter': 1000})
        #tf.logging.set_verbosity(tf.logging.INFO)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()

    # Initialze writers and variables
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
    tf.global_variables_initializer().run()
    timediff(start, 'Variables initialized')

    epoch = 0

    timediff(start, 'Starting loss calculation')
    # This is dependent on dataset size
    batch_size = int(np.floor(datasets.train.num_examples/10))
    step_per_report = 1
    xs, ys = datasets.train.next_batch(batch_size)
    feed_dict = {x: xs, y_ds: ys}
    summary, lo = sess.run([merged, loss], feed_dict=feed_dict)
    timediff(start, 'Algorithm started')
    print()
    print('epoch {:07} {:23} {:5.0f}'.format(epoch, 'loss is', np.round(lo)))
    print()

    # Define variables for early stopping
    not_improved = 0
    best_mse = np.inf
    early_stop = 0
    saver = tf.train.Saver(max_to_keep=early_stop)
    checkpoint_dir = 'checkpoints'
    tf.gfile.MkDir(checkpoint_dir)
    try:
        for ii in range(FLAGS.max_steps):
            # Write figures, summaries and check early stopping each epoch
            if datasets.train.epochs_completed > epoch:
                epoch = datasets.train.epochs_completed
                xs, ys = datasets.test.next_batch(-1, shuffle=False)
                feed_dict = {x: xs, y_ds: ys}
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, lo, meanse = sess.run([merged, loss, mse],
                                               feed_dict=feed_dict,
                                               options=run_options,
                                               run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)

                test_writer.add_summary(summary, ii)
                test_writer.add_run_metadata(run_metadata, 'step%d' % ii)

                #save_path = saver.save(sess, os.path.join(checkpoint_dir,
                #'model.ckpt'), global_step=ii)

                # Write checkpoint of NN
                model_to_json('nn_checkpoint.json', scan_dims.values.tolist(),
                              [train_dim],
                              datasets.train, scale_factor.astype('float64'),
                              scale_bias.astype('float64'))

                print('{:5} {:07} {:23} {:5.0f}'.format('epoch',
                                                        epoch,
                                                        'mse is',
                                                        np.round(meanse)))
                print('{:5} {:07} {:23} {:5.0f}'.format('epoch',
                                                        epoch,
                                                        'loss is',
                                                        np.round(lo)))
                timediff(start, 'completed')
                print()

                # Early stepping, check if MSE is better
                if meanse < best_mse:
                    copyfile('nn_checkpoint.json', 'nn_best.json')
                    best_mse = meanse
                    not_improved = 0
                else:
                    not_improved += 1
                # If not improved in 'early_stop' epoch, stop
                if not_improved >= early_stop:
                    print('Not improved for %s epochs, stopping..'
                          % (early_stop))
                    break
            else: # If NOT epoch done
                if not ii % step_per_report:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                xs, ys = datasets.train.next_batch(batch_size)
                feed_dict = {x: xs, y_ds: ys}
                if optimizer:
                    #optimizer.minimize(sess, feed_dict=feed_dict)
                    optimizer.minimize(sess,
                                       feed_dict=feed_dict,
                    #                   options=run_options,
                    #                   run_metadata=run_metadata)
                                      )
                    ce = loss.eval(feed_dict=feed_dict)
                    meanse = mse.eval(feed_dict=feed_dict)
                    summary = merged.eval(feed_dict=feed_dict)
                else:
                    ce, summary, meanse, _ = sess.run([loss, merged,
                                                       mse, train_step],
                                                      feed_dict=feed_dict,
                                                      options=run_options,
                                                      run_metadata=run_metadata
                                                      )
                train_writer.add_summary(summary, ii)

                if not ii % step_per_report:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline_run.json', 'w') as f:
                        f.write(ctf)
                    print('{:5} {:06} {:23} {:5.0f}'.format('step',
                                                            ii,
                                                            'loss is',
                                                            np.round(ce)))
                    print('{:5} {:06} {:23} {:5.0f}'.format('step',
                                                            ii,
                                                            'mse is',
                                                            np.round(meanse)))
                if np.isnan(ce):
                    raise Exception('Loss is NaN! Stopping..')

    except KeyboardInterrupt:
        print('KeyboardInterrupt Stopping..')

    train_writer.close()
    test_writer.close()

    try:
        saver.restore(sess, saver.last_checkpoints[epoch - not_improved])
    except IndexError:
        print("Can't restore old checkpoint, just saving current values")
    model_to_json('nn.json', scan_dims.values.tolist(), [train_dim],
                  datasets.train,
                  scale_factor.astype('float64'),
                  scale_bias.astype('float64'),
                  l2_scale)

    # Finally, check against validation set
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    ests = y.eval({x: xs, y_ds: ys})
    line_x = np.linspace(float(ys.min()), float(ys.max()))
    rms_val = np.round(np.sqrt(mse.eval({x: xs, y_ds: ys})), 2)
    print('{:22} {:5.2f}'.format('Validation RMS error: ', rms_val))

    # And to be sure, test against test and train set
    xs, ys = datasets.test.next_batch(-1, shuffle=False)
    rms_test = np.round(np.sqrt(mse.eval({x: xs, y_ds: ys})), 2)
    print('{:22} {:5.2f}'.format('Test RMS error: ', rms_test))
    xs, ys = datasets.train.next_batch(-1, shuffle=False)
    rms_train = np.round(np.sqrt(mse.eval({x: xs, y_ds: ys})), 2)
    print('{:22} {:5.2f}'.format('Train RMS error: ', rms_train))

    metadata = {'epoch': epoch,
                'rms_validation': rms_val,
                'rms_test': rms_test,
                'rms_train': rms_train,
                'nn_develop_version': nn_version,
                'activation': '2/(1+exp(-2x))-1'}

    with open('nn.json') as nn_file:
        data = json.load(nn_file)

    data['_metadata'].update(metadata)

    with open('nn.json', 'w') as nn_file:
        json.dump(data, nn_file, sort_keys=True,
                  indent=4, separators=(',', ': '))


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
    parser.add_argument('--data_dir', type=str,
                        default='train_NN_run/input_data/',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='train_NN_run/logs/',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
