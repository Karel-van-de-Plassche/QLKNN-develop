import matplotlib.pyplot as plt
import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed
from collections import OrderedDict
from itertools import product, chain

from tensorflow.contrib.learn.python.learn.learn_io import pandas_io
from tensorflow.python.ops import init_ops

tf.logging.set_verbosity(tf.logging.INFO)

ds = xr.open_dataset('/mnt/hdd/4D_sepflux.nc')
ds = xr.open_dataset('/mnt/hdd/4D.nc')
#ds = ds.sel(smag=0, Ti_Te=1, method='nearest')
scan_dims = [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']
train_dim = 'efe_GB'
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
def fake_panda():
    fakepanda = pd.DataFrame(columns = scan_dims[:2] + [train_dim])
    fakedims = {}
    npoint = 60
    fakedims = {scan_dim: np.linspace(0, 1, npoint) for scan_dim in scan_dims}
    for scan_dim in scan_dims:
        fakepanda[scan_dim] = np.full(npoint**len(scan_dims), np.nan) 
    fakepanda[train_dim] = np.full_like(fakepanda[scan_dims[0]], np.nan)
    for ii, dims in enumerate(product(*fakedims.values())):
        fakepanda.iloc[ii] = list(dims) + [fake_func(dims)]
    return fakepanda


df = ds.drop([coord for coord in ds.coords if coord not in ds.dims]).drop('kthetarhos').to_dataframe()
panda = pd.DataFrame(df.to_records())
scan_dims = scan_dims[:2]
panda = fake_panda()
#panda = panda[panda[train_dim] > 0]
#panda = panda[panda[train_dim] < 100]
#normalizer = {}
#for name, col in panda.items():
#    normalizer[name] = normalize(col)
#    panda[name] = panda[name] / normalizer[name]
panda_test, panda_train = split_panda(panda, frac=0.2)
#panda_train = pd.DataFrame(np.linspace(0,20, 1000))
#panda_train[train_dim] = np.linspace(0,20, 1000)

#test_input_fn = generate_input_fn(panda_test)
#train_input_fn = generate_input_fn(panda_train)
#feature_columns = [tf.contrib.layers.real_valued_column(col) for col in [dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']]

# Parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = int(np.floor(panda_train.shape[0] / 40))
display_step = 1

# Network Parameters
n_hidden_1 = 20 # 1st layer number of features
n_hidden_2 = 20 # 2nd layer number of features
n_input = len(scan_dims) # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

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

def my_input_fn(panda):
    features = {}
    for name in scan_dims:
        features[name] = tf.constant(panda[name].values)

    return features, tf.constant(panda[train_dim].values)

def train_input_fn():
    return my_input_fn(panda_train)


#scan_dims = [0]
def qualikiz_sigmoid(x, name=""):
    return tf.divide(2., tf.add(1., tf.exp(tf.multiply(-2., x)))) - 1.

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=qualikiz_sigmoid):
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

def multilayer_perceptron(x, weights, biases, activation):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = activation(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
init_func = tf.random_normal
weights = {
    'h1': tf.Variable(init_func([n_input, n_hidden_1])),
    'h2': tf.Variable(init_func([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(init_func([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(init_func([n_hidden_1])),
    'b2': tf.Variable(init_func([n_hidden_2])),
    'out': tf.Variable(init_func([n_classes]))
}

# Construct model

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
hidden1 = nn_layer(x, n_input, n_hidden_1, 'layer1')
hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
pred = nn_layer(hidden2, n_hidden_2, n_classes, 'layero')
merged = tf.summary.merge_all()
log_dir = 'test'
#x_norm = tf.nn.l2_normalize(x, 0)
#y_norm = tf.nn.l2_normalize(y, 0)
#y_norm = tf.nn.l2_normalize(y, 0)
#pred = multilayer_perceptron(x, weights, biases, qualikiz_sigmoid)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_norm))
#cost = tf.reduce_mean(tf.square(tf.subtract(y_norm, pred))e
wsum = 0.000 * tf.add_n([tf.reduce_sum(tf.square(tens)) for tens in chain(weights.values(), biases.values())])
cost = tf.contrib.losses.mean_squared_error(pred, y) + wsum
#cost = tf.contrib.losses.sigmoid_cross_entropy(pred, y)
#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#    panda_test[scan_dims],
#    panda_test[train_dim])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.contrib.opt.python.training.external_optimizer.ScipyOptimizerInterface(cost, options={'maxiter': 1000})
#optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': iterations}).minimize(session)
#feature_columns = [tf.contrib.layers.real_valued_column(k)
                   #for k in scan_dims]
#regressor = tf.contrib.learn.DNNRegressor([20, 20], model_dir='test', feature_columns=feature_columns)
#regressor.fit(input_fn=train_input_fn)


# Initializing the variables

right = np.tile(panda[scan_dims[1:]].median().values, (100,1))
left = np.atleast_2d(np.linspace(np.min(panda_train[scan_dims[0]]),np.max(panda_train[scan_dims[0]]), 100)).T
eval_x = np.concatenate((left, right), axis=1)
early_counter = 0
early_limit = 50
min_avg_error = np.inf

#sv = tf.train.Supervisor(logdir='test')
#with sv.managed_session("") as sess:
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    #tf.train.LoggingTensorHook({'cost': cost},  every_n_iter=1)

    # Training cycle
    #tf.train.basic_train_loop(sv, train_step_fn, kwargs={'train': panda_train, 'test':panda_test})
    #for epoch in range(training_epochs):
    train_set = panda_train
    test_set = panda_test
    total_batch = int(train_set.shape[0] / batch_size)
    # Loop over all batches
    train_set = shuffle_panda(panda_train)
    for epoch in range(training_epochs):
        if epoch % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if epoch % 100 == 99:  # Record execution stats
              run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
              run_metadata = tf.RunMetadata()
              summary, _ = sess.run([merged, train_step],
                                    feed_dict=feed_dict(True),
                                    options=run_options,
                                    run_metadata=run_metadata)
              train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
              train_writer.add_summary(summary, i)
              print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
       # avg_cost = 0.
       # avg_val = 0.
       # for ii in range(total_batch):
       #     #batch_x, batch_y = mnist.train.next_batch(batch_size)
       #     batch_x = train_set[scan_dims][ii*batch_size:(ii+1)*batch_size]
       #     batch_y = np.atleast_2d(train_set[train_dim][ii*batch_size:(ii+1)*batch_size]).T
       #     # Run optimization op (backprop) and cost op (to get loss value)
       #     _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
       #                                                   y: batch_y})
       #     #optimizer.minimize(sess, feed_dict={x: batch_x,
       #     #                                y: batch_y})
       #     # Compute average loss
       #     avg_cost += np.sqrt(c) / total_batch
       #     eval_y_norm = pred.eval(session=sess, feed_dict={x: test_set[scan_dims]})
       #     #avg_error = np.mean(np.abs(1 - eval_y_norm[0] / test_set[train_dim]))
       #     avg_error = np.mean(np.square(eval_y_norm[0] - test_set[train_dim]))
       # # Display logs per epoch step
       # if epoch % display_step == 0:
       #     print("Epoch:", '%04d' % (epoch+1),
       #           "cost=", "{:.9f}".format(avg_cost),
       #           "error=", "{:.9f}".format(avg_error))
       #     #foo = y_norm.eval({y: batch_y, x: batch_x})
       #     #bar = x_norm.eval({y: batch_y, x: batch_x})
       #     #eval_y = pred.eval({x: eval_x})
    print ("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(pred, y)
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #batch_x = panda_test[scan_dims].sample(batch_size)
    #batch_y = np.atleast_2d(panda_test['efeETG_GB'].sample(batch_size)).T
    #print("Accuracy:", 
    #print(pred.eval({x: batch_x, y:batch_y}))
    #print(y_norm.eval({x: batch_x, y:batch_y}))
    #print(np.mean(pred.eval({x: batch_x, y:batch_y}) / y_norm.eval({x: batch_x, y:batch_y})))
    eval_y_norm = pred.eval({x: panda_test[scan_dims]})
    line_x = np.linspace(np.min(panda_test[train_dim]), np.max(panda_test[train_dim]), 10000)
    plt.plot(line_x, line_x)
    plt.scatter(panda_test[train_dim], eval_y_norm)
    #plt.xlim([np.min(panda_test[train_dim]), np.max(panda_test[train_dim])])
    #plt.ylim([np.min(panda_test[train_dim]), np.max(panda_test[train_dim])])
    avg_error = np.abs(1 - eval_y_norm[0] / panda_test[train_dim])


    plt.figure()
    slice_y_norm = pred.eval({x: eval_x})
    plt.scatter(eval_x[:, 0], slice_y_norm)
    plt.plot(eval_x[:, 0], [np.sum(np.square(np.hstack((eval_x[0, 1:], [x])))) / len(scan_dims) for x in eval_x[:, 0]])
    print(eval_x[0, :])
    plt.show()

    embed()







#x = panda[[dim for dim in ds.dims if dim != 'kthetarhos' and dim != 'nions' and dim!='numsols']].copy()
#y = panda['efeETG_GB'].copy()
#x_test = x.sample(n=1000)
#x_test = x.sample(n=1000)
#test_i = np.random.randint(0, x.shape[0], 1000)
#x_train = x.drop(test_i)
#y_train = y.drop(test_i)
#x_test = x.loc[test_i]
#y_test = y.loc[test_i]
#index = np.arange(100, 104)
#a = np.arange(4)
#b = np.arange(32, 36)
#x = pd.DataFrame({'a': a, 'b': b}, index=index)
#y = pd.Series(np.arange(-32, -28), index=index)
#test_input_fn = pandas_io.pandas_input_fn(x_test, y_test, batch_size=8, num_epochs=1)
#train_input_fn = pandas_io.pandas_input_fn(x_train, y_train, batch_size=8, num_epochs=1)
# Prints data
#with tf.Session() as sess:
#    results = input_fn()
#    coord = coordinator.Coordinator()
#    threads = queue_runner_impl.start_queue_runners(sess, coord=coord)
#    result_values = sess.run(results)
#    coord.request_stop()
#    coord.join(threads)
