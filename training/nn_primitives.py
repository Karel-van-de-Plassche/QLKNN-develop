import tensorflow as tf
import numpy as np
import subprocess
import json

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

    #sp_result = subprocess.run('git rev-parse HEAD',
    #                           stdout=subprocess.PIPE,
    #                           shell=True,
    #                           check=True)
    #nn_version = sp_result.stdout.decode('UTF-8').strip()
    metadata = {
    #    'nn_develop_version': nn_version,
        'c_L2': float(l2_scale.eval())
    }
    dict_['_metadata'] = metadata

    with open(name, 'w') as file_:
        json.dump(dict_, file_, sort_keys=True, indent=4, separators=(',', ': '))

def weight_variable(shape, init='norm_1_0', dtype=tf.float64, **kwargs):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.truncated_normal(shape, stddev=0.1)
    if isinstance(init, np.ndarray):
        initial = tf.constant(init, dtype=dtype)
    else:
        if init == 'norm_1_0':
            initial = tf.random_normal(shape, **kwargs)
    return tf.Variable(initial)

def bias_variable(shape, init='norm_1_0', dtype=tf.float64, **kwargs):
    """Create a bias variable with appropriate initialization."""
    #initial = tf.constant(0.1, shape=shape)
    if isinstance(init, np.ndarray):
        initial = tf.constant(init, dtype=dtype)
    else:
        if init == 'norm_1_0':
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
             dtype=tf.float32, debug=False, weight_init='norm_1_0', bias_init='norm_1_0'):
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
            weights = weight_variable([input_dim, output_dim], init=weight_init, dtype=dtype)
            if debug:
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], dtype=dtype, init=bias_init)
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
