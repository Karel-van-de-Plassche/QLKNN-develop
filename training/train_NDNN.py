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
from itertools import product


import numpy as np
import pandas as pd
from IPython import embed
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
                  train_set, scale_factor, scale_bias, l2_scale, settings):
    dict_ = {x.name: tf.to_double(x).eval().tolist() for x in tf.trainable_variables()}
    dict_['prescale_factor'] = scale_factor.astype('float64').to_dict()
    dict_['prescale_bias'] = scale_bias.astype('float64').to_dict()
    dict_['feature_min'] = dict(train_set._features.astype('float64').min())
    dict_['feature_max'] = dict(train_set._features.astype('float64').max())
    dict_['feature_names'] = feature_names
    dict_['target_names'] = target_names
    dict_['target_min'] = dict(train_set._target.astype('float64').min())
    dict_['target_max'] = dict(train_set._target.astype('float64').max())
    dict_['hidden_activation'] = settings['hidden_activation']
    dict_['output_activation'] = settings['output_activation']

    sp_result = subprocess.run('git rev-parse HEAD',
                               stdout=subprocess.PIPE,
                               shell=True,
                               check=True)
    nn_version = sp_result.stdout.decode('UTF-8').strip()
    metadata = {
        'nn_develop_version': nn_version,
        'c_L2': float(l2_scale.eval())
    }
    dict_['_metadata'] = metadata

    with open(name, 'w') as file_:
        json.dump(dict_, file_, sort_keys=True, indent=4, separators=(',', ': '))


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


def nn_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu,
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
        if act is not None:
            activations = act(preactivate, name='activation')
        else:
            activations = preactivate
        if debug:
            tf.summary.histogram('activations', activations)
        return activations


def normab(panda, a, b):
    factor = (b - a) / (panda.max() - panda.min())
    bias = (b - a) * panda.min() / (panda.max() - panda.min()) + a
    return factor, bias

def normsm(panda, s_t, m_t):
    m_s = np.mean(panda)
    s_s = np.std(panda)
    factor = s_t / s_s
    bias = -m_s * s_t / s_s + m_t
    return factor, bias

def print_last_row(df, header=False):
    print(df.iloc[[-1]].to_string(header=header,
                                  float_format=lambda self: u'{:.2f}'.format(self),
                                  col_space=12,
                                  justify='left'))

def train(settings):
    # Import data
    start = time.time()
    # Get train dimension from path name
    #train_dim = os.path.basename(os.getcwd())
    train_dim = settings['train_dim']
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
        try:
            input['logNustar'] = np.log10(input['Nustar'])
            del input['Nustar']
        except KeyError:
            print('No Nustar in dataset')
        input = input.loc[panda.index]
        panda = pd.concat([input, panda], axis=1)
        timediff(start, 'Dataset loaded')
    timediff(start, 'Dataset filtered')
    panda = panda.astype('float64')
    panda = panda.astype('float32')

    # Use pre-existing splitted dataset, or split in train, validation and test
    train_dim = panda.columns[-1]
    scan_dims = panda.columns[:-1]
    if os.path.exists('splitted.h5'):
        datasets = Datasets.read_hdf('splitted.h5')
    else:
        datasets = convert_panda(panda, 0.1, 0.1, scan_dims, train_dim)
        datasets = convert_panda(panda, 0.06, 0.06, scan_dims, train_dim)
        datasets.to_hdf('splitted.h5')
    # Convert back to float64 for tensorflow compatibility
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
    #config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
    #                    allow_soft_placement=True, device_count = {'CPU': 1})
    #session = tf.Session(config=config)
    #K.set_session(session)

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(datasets.train._target.dtypes.iloc[0],
                           [None, len(scan_dims)], name='x-input')
        y_ds = tf.placeholder(x.dtype, [None, 1], name='y-input')

    # Scale all input
    with tf.name_scope('normalize'):
        if settings['standardization'].startswith('minmax'):
            min = float(settings['standardization'].split('_')[-2])
            max = float(settings['standardization'].split('_')[-1])
            scale_factor, scale_bias = normab(panda, min, max)
        if settings['standardization'].startswith('normsm'):
            s_t = float(settings['standardization'].split('_')[-2])
            m_t = float(settings['standardization'].split('_')[-1])
            scale_factor, scale_bias = normsm(panda, s_t, m_t)
        in_factor = tf.constant(scale_factor[scan_dims].values, dtype=x.dtype)
        in_bias = tf.constant(scale_bias[scan_dims].values, dtype=x.dtype)

        x_scaled = in_factor * x + in_bias
    timediff(start, 'Scaling defined')

    # Define neural network
    layers = [x_scaled]
    debug = False
    drop_prob = tf.constant(settings['drop_chance'])
    is_train = tf.placeholder(tf.bool)
    for ii, (activation, neurons) in enumerate(zip(settings['hidden_activation'], settings['hidden_neurons']), start=1):
        if activation == 'tanh':
            act = tf.tanh
        elif activation == 'relu':
            act = tf.nn.relu
        elif activation == 'none':
            act = None
        layer = nn_layer(layers[-1], neurons, 'layer' + str(ii), dtype=x.dtype, act=act, debug=debug)
        dropout = tf.layers.dropout(layer, drop_prob, training=is_train)
        if debug:
            tf.summary.histogram('post_dropout_layer_' + str(ii), dropout)
        layers.append(dropout)

    # All output scaled between -1 and 1, denomalize it
    activation = settings['output_activation']
    if activation == 'tanh':
        act = tf.tanh
    elif activation == 'relu':
        act = tf.nn.relu
    elif activation == 'none':
        act = None
    y_scaled = nn_layer(layers[-1], 1, 'layer' + str(len(layers)), dtype=x.dtype, act=act, debug=debug)
    with tf.name_scope('denormalize'):
        out_factor = tf.constant(scale_factor[train_dim], dtype=x.dtype)
        out_bias = tf.constant(scale_bias[train_dim], dtype=x.dtype)
        y = (y_scaled - out_bias) / out_factor

    timediff(start, 'NN defined')

    # Define loss functions
    with tf.name_scope('Loss'):
        with tf.name_scope('mse'):
            mse = (tf.reduce_mean(tf.square(tf.subtract(y_ds, y))))
            tf.summary.scalar('MSE', mse)
        with tf.name_scope('mabse'):
            mabse = (tf.reduce_mean(tf.abs(tf.subtract(y_ds, y))))
            tf.summary.scalar('MABSE', mabse)
        with tf.name_scope('l2'):
            l2_scale = tf.Variable(settings['cost_l2_scale'], dtype=x.dtype, trainable=False)
            #l2_norm = tf.reduce_sum(tf.square())
            #l2_norm = tf.to_double(tf.add_n([tf.nn.l2_loss(var)
            #                        for var in tf.trainable_variables()
            #                        if 'weights' in var.name]))
            l2_norm = (tf.add_n([tf.nn.l2_loss(var)
                                    for var in tf.trainable_variables()
                                    if 'weights' in var.name]))
            #mse = tf.losses.mean_squared_error(y_, y)
            # TODO: Check normalization
            l2_loss = l2_scale * l2_norm
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('l2_scale', l2_scale)
            tf.summary.scalar('l2_loss', l2_loss)
        with tf.name_scope('l1'):
            l1_scale = tf.Variable(settings['cost_l1_scale'], dtype=x.dtype, trainable=False)
            #l1_norm = tf.to_double(tf.add_n([tf.reduce_sum(tf.abs(var))
            #                        for var in tf.trainable_variables()
            #                        if 'weights' in var.name]))
            l1_norm = (tf.add_n([tf.reduce_sum(tf.abs(var))
                                    for var in tf.trainable_variables()
                                    if 'weights' in var.name]))
            # TODO: Check normalization
            l1_loss = l1_scale * l1_norm
            tf.summary.scalar('l1_norm', l1_norm)
            tf.summary.scalar('l1_scale', l1_scale)
            tf.summary.scalar('l1_loss', l1_loss)
        #loss = mse
        if settings['goodness'] == 'mse':
            loss = mse
        elif settings['goodness'] == 'mabse':
            loss = mabse
        if settings['cost_l1_scale'] != 0:
            loss += l1_loss
        if settings['cost_l2_scale'] != 0:
            loss += l2_loss
        tf.summary.scalar('loss', loss)

    optimizer = None
    train_step = None
    # Define fitting algorithm. Kept old algorithms commented out.
    with tf.name_scope('train'):
        lr = settings['learning_rate']
        if settings['optimizer'] == 'adam':
            beta1 = settings['adam_beta1']
            beta2 = settings['adam_beta2']
            train_step = tf.train.AdamOptimizer(lr,
                                                beta1,
                                                beta2,
                                                ).minimize(loss)
        elif settings['optimizer'] == 'adadelta':
            rho = settings['adadelta_rho']
            train_step = tf.train.AdadeltaOptimizer(lr,
                                                    rho,
                                                    ).minimize(loss)
        elif settings['optimizer'] == 'rmsprop':
            decay = settings['rmsprop_decay']
            momentum = settings['rmsprop_momentum']
            train_step = tf.train.RMSPropOptimizer(lr,
                                                   decay,
                                                   momentum).minimize(loss)
        elif settings['optimizer'] == 'grad':
            train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        elif settings['optimizer'] == 'lbfgs':
            optimizer = opt.ScipyOptimizerInterface(loss,
                                                    options={'maxiter': settings['lbfgs_maxiter'],
                                                             'maxfun': settings['lbfgs_maxfun'],
                                                             'maxls': settings['lbfgs_maxls']})
        #tf.logging.set_verbosity(tf.logging.INFO)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()

    # Initialze writers and variables
    log_dir = 'train_NDNN/logs'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
    tf.global_variables_initializer().run()
    timediff(start, 'Variables initialized')

    epoch = 0

    train_log = pd.DataFrame(columns=['epoch', 'walltime', 'loss', 'mse', 'mabse', 'l1_norm', 'l2_norm'])
    validation_log = pd.DataFrame(columns=['epoch', 'walltime', 'loss', 'mse', 'mabse', 'l1_norm', 'l2_norm'])

    # This is dependent on dataset size
    batch_size = int(np.floor(datasets.train.num_examples/settings['minibatches']))

    timediff(start, 'Starting loss calculation')
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    summary, lo, meanse, meanabse, l1norm, l2norm  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm],
                                                              feed_dict=feed_dict)
    timediff(start, 'Algorithm started')
    train_log.loc[0] = (epoch, 0, lo, meanse, meanabse, l1norm, l2norm)
    validation_log.loc[0] = (epoch, 0, lo, meanse, meanabse, l1norm, l2norm)
    print_last_row(train_log, header=True)

    # Define variables for early stopping
    not_improved = 0
    best_early_measure = np.inf
    early_measure = np.inf

    saver = tf.train.Saver(max_to_keep=settings['early_stop_after'] + 1)
    checkpoint_dir = 'checkpoints'
    tf.gfile.MkDir(checkpoint_dir)

    steps_per_report = settings.get('steps_per_report') or np.inf
    epochs_per_report = settings.get('epochs_per_report') or np.inf
    save_checkpoint_networks = settings.get('save_checkpoint_networks') or False
    save_best_networks = settings.get('save_best_networks') or True
    train_start = time.time()

    steps_per_epoch = settings['minibatches'] + 1
    max_epoch = settings.get('max_epoch') or sys.maxsize
    try:
        for ii in range(steps_per_epoch * max_epoch):
            # Write figures, summaries and check early stopping each epoch
            if datasets.train.epochs_completed > epoch:
                epoch = datasets.train.epochs_completed
                xs, ys = datasets.validation.next_batch(-1, shuffle=False)
                feed_dict = {x: xs, y_ds: ys, is_train: False}
                if not ii % epochs_per_report:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                summary, lo, meanse, meanabse, l1norm, l2norm  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm],
                                               feed_dict=feed_dict,
                                               options=run_options,
                                               run_metadata=run_metadata)


                validation_writer.add_summary(summary, ii)
                if not ii % epochs_per_report:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                    validation_writer.add_run_metadata(run_metadata, 'step%d' % ii)

                save_path = saver.save(sess, os.path.join(checkpoint_dir,
                'model.ckpt'), global_step=ii)

                validation_log.loc[ii] = (epoch, time.time() - train_start, lo, meanse, meanabse, l1norm, l2norm)
                print()
                print_last_row(validation_log, header=True)
                timediff(start, 'completed')
                print()

                if settings['early_stop_measure'] == 'mse':
                    early_measure = meanse
                elif settings['early_stop_measure'] == 'loss':
                    early_measure = lo
                elif settings['early_stop_measure'] == 'none':
                    early_measure = np.nan

                # Early stopping, check if measure is better
                if early_measure < best_early_measure:
                    best_early_measure = early_measure
                    if save_best_networks:
                        nn_best_file = os.path.join(checkpoint_dir,
                                                      'nn_checkpoint_' + str(epoch) + '.json')
                        model_to_json(nn_best_file, scan_dims.values.tolist(),
                                      [train_dim],
                                      datasets.train, scale_factor.astype('float64'),
                                      scale_bias.astype('float64'),
                                      l2_scale,
                                      settings)
                    not_improved = 0
                else:
                    not_improved += 1
                # If not improved in 'early_stop' epoch, stop
                if settings['early_stop_measure'] != 'none' and not_improved >= settings['early_stop_after']:
                    if save_checkpoint_networks:
                        nn_checkpoint_file = os.path.join(checkpoint_dir,
                                                      'nn_checkpoint_' + str(epoch) + '.json')
                        model_to_json(nn_checkpoint_file, scan_dims.values.tolist(),
                                      [train_dim],
                                      datasets.train, scale_factor.astype('float64'),
                                      scale_bias.astype('float64'),
                                      l2_scale,
                                      settings)

                    print('Not improved for %s epochs, stopping..'
                          % (not_improved))
                    break
            else: # If NOT epoch done
                if not ii % steps_per_report:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                xs, ys = datasets.train.next_batch(batch_size)
                feed_dict = {x: xs, y_ds: ys, is_train: True}
                if optimizer:
                    #optimizer.minimize(sess, feed_dict=feed_dict)
                    optimizer.minimize(sess,
                                       feed_dict=feed_dict,
                    #                   options=run_options,
                    #                   run_metadata=run_metadata)
                                      )
                    lo = loss.eval(feed_dict=feed_dict)
                    meanse = mse.eval(feed_dict=feed_dict)
                    meanabse = mabse.eval(feed_dict=feed_dict)
                    l1norm = l1_norm.eval(feed_dict=feed_dict)
                    l2norm = l2_norm.eval(feed_dict=feed_dict)
                    summary = merged.eval(feed_dict=feed_dict)
                else:
                    summary, lo, meanse, meanabse, l1norm, l2norm, _  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm, train_step],
                                                      feed_dict=feed_dict,
                                                      options=run_options,
                                                      run_metadata=run_metadata
                                                      )
                train_writer.add_summary(summary, ii)

                if not ii % steps_per_report:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline_run.json', 'w') as f:
                        f.write(ctf)
                train_log.loc[ii] = (epoch, time.time() - train_start, lo, meanse, meanabse, l1norm, l2norm)
                print_last_row(train_log)
            if np.isnan(lo) or np.isinf(lo):
                print('Loss is {}! Stopping..'.format(lo))
                break

    except KeyboardInterrupt:
        print('KeyboardInterrupt Stopping..')

    train_writer.close()
    validation_writer.close()

    try:
        best_epoch = epoch - not_improved
        saver.restore(sess, saver.last_checkpoints[best_epoch - epoch])
    except IndexError:
        print("Can't restore old checkpoint, just saving current values")
        best_epoch = epoch
    model_to_json('nn.json', scan_dims.values.tolist(), [train_dim],
                  datasets.train,
                  scale_factor,
                  scale_bias.astype('float64'),
                  l2_scale,
                  settings)

    # Finally, check against validation set
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    ests = y.eval(feed_dict)
    line_x = np.linspace(float(ys.min()), float(ys.max()))
    rms_val = np.round(np.sqrt(mse.eval(feed_dict)), 4)
    loss_val = np.round(loss.eval(feed_dict), 4)
    print('{:22} {:5.2f}'.format('Validation RMS error: ', rms_val))

    # And to be sure, test against test and train set
    xs, ys = datasets.test.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    rms_test = np.round(np.sqrt(mse.eval(feed_dict)), 4)
    loss_test = np.round(loss.eval(feed_dict), 4)
    print('{:22} {:5.2f}'.format('Test RMS error: ', rms_test))
    #xs, ys = datasets.train.next_batch(-1, shuffle=False)
    #feed_dict = {x: xs, y_ds: ys, is_train: False}
    #rms_train = np.round(np.sqrt(mse.eval(feed_dict)), 4)
    #loss_train = np.round(loss.eval(feed_dict), 4)
    #print('{:22} {:5.2f}'.format('Train RMS error: ', rms_train))

    metadata = {'epoch':           epoch,
                'best_epoch':      best_epoch,
                'rms_validation':  float(rms_val),
                'rms_test':        float(rms_test),
    #            'rms_train':      float(rms_train),
                'loss_validation': float(loss_val),
                'loss_test':       float(loss_test),
    #            'loss_train':     float(loss_train)
                }

    with open('nn.json') as nn_file:
        data = json.load(nn_file)

    data['_metadata'].update(metadata)

    with open('nn.json', 'w') as nn_file:
        json.dump(data, nn_file, sort_keys=True, indent=4, separators=(',', ': '))

    train_log.to_csv('train_log.csv')
    validation_log.to_csv('validation_log.csv')


def main(_):
    with open('./settings.json') as file_:
        settings = json.load(file_)
    train(settings)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
    #                    default=True,
    #                    help='If true, uses fake data for unit testing.')
    #parser.add_argument('--max_steps', type=int, default=100000,
    #parser.add_argument('--max_steps', type=int, default=sys.maxsize,
    #                    help='Number of steps to run trainer.')
    #parser.add_argument('--learning_rate', type=float, default=10.,
    #                    help='Initial learning rate')
    #parser.add_argument('--dropout', type=float, default=0.9,
    #                    help='Keep probability for training dropout.')
    #parser.add_argument('--data_dir', type=str,
    #                    default='train_NN_run/input_data/',
    #                    help='Directory for storing input data')
    #parser.add_argument('--log_dir', type=str, default='train_NN_run/logs/',
    #                    help='Summaries log directory')
    #FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
