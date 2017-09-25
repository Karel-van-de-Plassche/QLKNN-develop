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
from datasets import Dataset, Datasets, convert_panda, split_panda, shuffle_panda
from nn_primitives import model_to_json, weight_variable, bias_variable, variable_summaries, nn_layer, normab, normsm

FLAGS = None

def timediff(start, event):
    print('{:35} {:5.0f}s'.format(event + ' after', time.time() - start))

def print_last_row(df, header=False):
    print(df.iloc[[-1]].to_string(header=header,
                                  float_format=lambda x: u'{:.2f}'.format(x),
                                  col_space=12,
                                  justify='left'))


def train(settings):
    # Import data
    start = time.time()
    train_dims = settings['train_dims']
    # Open HDF store. This is usually a soft link to our filtered dataset
    store = pd.HDFStore('filtered_everything_nions0.h5', 'r')

    # Get the targets (train_dims) and features (input)
    target_df = store.get(train_dims[0]).to_frame()
    for target_name in train_dims[1:]:
        target_df = pd.concat([target_df, store.get(target_name)], axis=1)
    input_df = store.select('input')

    try:
        del input_df['nions']  # Delete leftover artifact from dataset split
    except KeyError:
        pass

    # Nustar relates to the targets with a log
    try:
        input_df['logNustar'] = np.log10(input_df['Nustar'])
        del input_df['Nustar']
    except KeyError:
        print('No Nustar in dataset')

    # Use only samples in the feature set that are in the target set
    input_df = input_df.loc[target_df.index]

    # Convert to dtype in settings file. Usually float32 or float64
    input_df = input_df.astype(settings['dtype'])
    target_df = target_df.astype(settings['dtype'])

    train_dims = target_df.columns
    scan_dims = input_df.columns
    # Keep option to restore splitted dataset from file for debugging
    restore_split_backup = False
    if restore_split_backup and os.path.exists('splitted.h5'):
        datasets = Datasets.read_hdf('splitted.h5')
    else:
        datasets = convert_panda(input_df, target_df, settings['validation_fraction'], settings['test_fraction'])
        datasets.to_hdf('splitted.h5')

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
        y_ds = tf.placeholder(x.dtype, [None, len(train_dims)], name='y-input')

    # Standardize input
    with tf.name_scope('standardize'):
        if settings['standardization'].startswith('minmax'):
            min = float(settings['standardization'].split('_')[-2])
            max = float(settings['standardization'].split('_')[-1])
            scale_factor, scale_bias = normab(pd.concat([input_df, target_df], axis=1), min, max)
        if settings['standardization'].startswith('normsm'):
            s_t = float(settings['standardization'].split('_')[-2])
            m_t = float(settings['standardization'].split('_')[-1])
            scale_factor, scale_bias = normsm(pd.concat([input_df, target_df], axis=1), s_t, m_t)
        in_factor = tf.constant(scale_factor[scan_dims].values, dtype=x.dtype)
        in_bias = tf.constant(scale_bias[scan_dims].values, dtype=x.dtype)

        x_scaled = in_factor * x + in_bias
    timediff(start, 'Scaling defined')

    # Define fully connected feed-forward NN. Potentially with dropout.
    layers = [x_scaled]
    debug = False
    drop_prob = tf.constant(settings['drop_chance'], dtype=x.dtype)
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

    # Last layer (output layer) usually has no activation
    activation = settings['output_activation']
    if activation == 'tanh':
        act = tf.tanh
    elif activation == 'relu':
        act = tf.nn.relu
    elif activation == 'none':
        act = None
    # Scale output back to original values
    y_scaled = nn_layer(layers[-1], 1, 'layer' + str(len(layers)), dtype=x.dtype, act=act, debug=debug)
    with tf.name_scope('destandardize'):
        out_factor = tf.constant(scale_factor[train_dims].values, dtype=x.dtype)
        out_bias = tf.constant(scale_bias[train_dims].values, dtype=x.dtype)
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
    # Define optimizer algorithm.
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

    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Initialze writers, variables and logdir
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

    # Split dataset in minibatches
    minibatches = settings['minibatches']
    batch_size = int(np.floor(datasets.train.num_examples/minibatches))

    timediff(start, 'Starting loss calculation')
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    summary, lo, meanse, meanabse, l1norm, l2norm  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm],
                                                              feed_dict=feed_dict)
    train_log.loc[0] = (epoch, 0, lo, meanse, meanabse, l1norm, l2norm)
    validation_log.loc[0] = (epoch, 0, lo, meanse, meanabse, l1norm, l2norm)
    print_last_row(train_log, header=True)

    # Save checkpoints of training to restore for early-stopping
    saver = tf.train.Saver(max_to_keep=settings['early_stop_after'] + 1)
    checkpoint_dir = 'checkpoints'
    tf.gfile.MkDir(checkpoint_dir)

    # Define variables for early stopping
    not_improved = 0
    best_early_measure = np.inf
    early_measure = np.inf

    steps_per_epoch = minibatches + 1
    max_epoch = settings.get('max_epoch') or sys.maxsize

    # Set debugging parameters
    steps_per_report = settings.get('steps_per_report') or np.inf
    epochs_per_report = settings.get('epochs_per_report') or np.inf
    save_checkpoint_networks = settings.get('save_checkpoint_networks') or False
    save_best_networks = settings.get('save_best_networks') or True

    # Set up log files
    train_log_file = open('train_log.csv', 'a', 1)
    train_log_file.truncate(0)
    train_log.to_csv(train_log_file)
    validation_log_file = open('validation_log.csv', 'a', 1)
    validation_log_file.truncate(0)
    validation_log.to_csv(validation_log_file)

    timediff(start, 'Training started')
    train_start = time.time()
    try:
        for ii in range(steps_per_epoch * max_epoch):
            # Write figures, summaries and check early stopping each epoch
            if datasets.train.epochs_completed > epoch:
                epoch = datasets.train.epochs_completed
                xs, ys = datasets.validation.next_batch(-1, shuffle=False)
                feed_dict = {x: xs, y_ds: ys, is_train: False}
                # Run with full trace every epochs_per_report Gives full runtime information
                if not ii % epochs_per_report:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None

                # Calculate all variables with the validation set
                summary, lo, meanse, meanabse, l1norm, l2norm  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm],
                                               feed_dict=feed_dict,
                                               options=run_options,
                                               run_metadata=run_metadata)


                validation_writer.add_summary(summary, ii)
                # More debugging every epochs_per_report
                if not ii % epochs_per_report:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                    validation_writer.add_run_metadata(run_metadata, 'step%d' % ii)

                # Save checkpoint
                save_path = saver.save(sess, os.path.join(checkpoint_dir,
                'model.ckpt'), global_step=ii)

                # Update CSV logs
                validation_log.loc[ii] = (epoch, time.time() - train_start, lo, meanse, meanabse, l1norm, l2norm)

                print()
                print_last_row(validation_log, header=True)
                timediff(start, 'completed')
                print()

                validation_log.loc[ii:].to_csv(validation_log_file, header=False)
                validation_log = validation_log[0:0]
                train_log.loc[ii - minibatches:].to_csv(train_log_file, header=False)
                train_log = train_log[0:0]

                # Determine early-stopping criterion
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
                                      train_dims.values.tolist(),
                                      datasets.train, scale_factor.astype('float64'),
                                      scale_bias.astype('float64'),
                                      l2_scale,
                                      settings)
                    not_improved = 0
                else: # If early measure is not better
                    not_improved += 1
                # If not improved in 'early_stop' epoch, stop
                if settings['early_stop_measure'] != 'none' and not_improved >= settings['early_stop_after']:
                    if save_checkpoint_networks:
                        nn_checkpoint_file = os.path.join(checkpoint_dir,
                                                      'nn_checkpoint_' + str(epoch) + '.json')
                        model_to_json(nn_checkpoint_file, scan_dims.values.tolist(),
                                      train_dims.values.tolist(),
                                      datasets.train, scale_factor.astype('float64'),
                                      scale_bias.astype('float64'),
                                      l2_scale,
                                      settings)

                    print('Not improved for %s epochs, stopping..'
                          % (not_improved))
                    break
            else: # If NOT epoch done
                # Extra debugging every steps_per_report
                if not ii % steps_per_report:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                xs, ys = datasets.train.next_batch(batch_size)
                feed_dict = {x: xs, y_ds: ys, is_train: True}
                # If we have a scipy-style optimizer
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
                else: # If we have a TensorFlow-style optimizer
                    summary, lo, meanse, meanabse, l1norm, l2norm, _  = sess.run([merged, loss, mse, mabse, l1_norm, l2_norm, train_step],
                                                      feed_dict=feed_dict,
                                                      options=run_options,
                                                      run_metadata=run_metadata
                                                      )
                train_writer.add_summary(summary, ii)

                # Extra debugging every steps_per_report
                if not ii % steps_per_report:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline_run.json', 'w') as f:
                        f.write(ctf)
                # Add to CSV log buffer
                train_log.loc[ii] = (epoch, time.time() - train_start, lo, meanse, meanabse, l1norm, l2norm)
                print_last_row(train_log)

            # Stop if loss is nan or inf
            if np.isnan(lo) or np.isinf(lo):
                print('Loss is {}! Stopping..'.format(lo))
                break

    # Stop on Ctrl-C
    except KeyboardInterrupt:
        print('KeyboardInterrupt Stopping..')

    train_writer.close()
    validation_writer.close()

    # Restore checkpoint with best epoch
    try:
        best_epoch = epoch - not_improved
        saver.restore(sess, saver.last_checkpoints[best_epoch - epoch])
    except IndexError:
        print("Can't restore old checkpoint, just saving current values")
        best_epoch = epoch

    train_log_file.close()
    del train_log
    validation_log_file.close()
    del validation_log

    model_to_json('nn.json',
                  scan_dims.values.tolist(),
                  train_dims.values.tolist(),
                  datasets.train,
                  scale_factor,
                  scale_bias.astype('float64'),
                  l2_scale,
                  settings)

    # Finally, check against validation set
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    ests = y.eval(feed_dict)
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

    # Add metadata dict to nn.json
    with open('nn.json') as nn_file:
        data = json.load(nn_file)

    data['_metadata'].update(metadata)

    with open('nn.json', 'w') as nn_file:
        json.dump(data, nn_file, sort_keys=True, indent=4, separators=(',', ': '))


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
