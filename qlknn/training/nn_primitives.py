import tensorflow as tf
import numpy as np
import subprocess
import json
import pandas as pd
from collections import OrderedDict

def scale_panda(panda, factor, bias):
    if isinstance(panda, pd.Series):
        filter = panda.index
    if isinstance(panda, pd.DataFrame):
        filter = panda.columns
    panda = factor[filter] * panda + bias[filter]
    return panda

def descale_panda(panda, factor, bias):
    if isinstance(panda, pd.Series):
        filter = panda.index
    if isinstance(panda, pd.DataFrame):
        filter = panda.columns
    panda = (panda - bias[filter]) / factor[filter]
    return panda

def descale_variable(a, b, var):
    return (var - b) / a

def model_to_json(name, trainable=None, feature_names=None, target_names=None, scale_factor=None, scale_bias=None, train_set=None, settings=None):
    """
    trainable: dict with all trainable values (e.g. weights and baises) get this with: `{x.name: tf.to_double(x).eval(session=sess).tolist() for x in tf.trainable_variables()}`
    feature_names: List of feature names. Order matters!
    target_names: List of target names. Order matters!
    scale_factor: Series with the 'a' of y_scaled = a * y + b
    scale_bias: Series with the 'b' of y_scaled = a * y + b
    train_set:  The full DataFrame used for training. To caclulate min/max values
    settings: The settings dict used for training. Used to extract the activation functions.
    """
    from IPython import embed
    nn_dict = OrderedDict()
    nn_dict['target_names'] = target_names
    nn_dict['feature_names'] = feature_names
    nn_dict['hidden_activation'] = settings['hidden_activation']
    nn_dict['output_activation'] = settings['output_activation']
    feature_a = scale_factor[feature_names].values
    feature_b = scale_bias[feature_names].values
    target_a = scale_factor[target_names].values
    target_b = scale_bias[target_names].values
    nn_dict['feature_min'] = OrderedDict(zip(feature_names,
                                             descale_variable(feature_a,
                                                              feature_b,
                                                              train_set._features.min(axis=0)).tolist()
    ))
    nn_dict['feature_max'] = OrderedDict(zip(feature_names,
                                             descale_variable(feature_a,
                                                              feature_b,
                                                              train_set._features.max(axis=0)).tolist()
    ))
    nn_dict['target_min'] = OrderedDict(zip(target_names,
                                            descale_variable(target_a,
                                                             target_b,
                                                             train_set._target.min(axis=0)).tolist()
    ))
    nn_dict['target_max'] = OrderedDict(zip(target_names,
                                            descale_variable(target_a,
                                                             target_b,
                                                             train_set._target.max(axis=0)).tolist()
    ))
    nn_dict['prescale_factor'] = OrderedDict((name, val) for name, val in scale_factor.items())
    nn_dict['prescale_bias'] = OrderedDict((name, val) for name, val in scale_bias.items())
    nn_dict.update(trainable)
    #nn_dict['target_min'] = OrderedDict(descale_variable())
    with open(name, 'w') as file_:
        json.dump(nn_dict, file_, indent=4, separators=(',', ': '))




def model_to_json_legacy(name, trainable, feature_names, target_names,
                  train_set,
                  feature_scale_factor, feature_scale_bias,
                  target_scale_factor, target_scale_bias,
                  l2_scale, settings):
    trainable['prescale_factor'] = dict(zip(feature_names + target_names, feature_scale_factor + target_scale_factor))
    trainable['prescale_bias'] = dict(zip(feature_names + target_names, feature_scale_bias + target_scale_bias))
    trainable['feature_min'] = dict(zip(feature_names, (train_set._features.min() - target_scale_bias) / target_scale_factor))
    trainable['feature_min'] = dict(zip(feature_names, (train_set._features.max() - target_scale_bias) / target_scale_factor))
    trainable['feature_names'] = feature_names
    trainable['target_names'] = target_names
    trainable['target_min'] = dict(zip(target_names, (train_set._target.min() - target_scale_bias) / target_scale_factor))
    trainable['target_min'] = dict(zip(target_names, (train_set._target.max() - target_scale_bias) / target_scale_factor))
    trainable['hidden_activation'] = settings['hidden_activation']
    trainable['output_activation'] = settings['output_activation']

    #sp_result = subprocess.run('git rev-parse HEAD',
    #                           stdout=subprocess.PIPE,
    #                           shell=True,
    #                           check=True)
    #nn_version = sp_result.stdout.decode('UTF-8').strip()
    #metadata = {
    #    'nn_develop_version': nn_version,
    #    'c_L2': float(l2_scale.eval())
    #}
    #trainable['_metadata'] = metadata

    with open(name, 'w') as file_:
        json.dump(trainable, file_, sort_keys=True, indent=4, separators=(',', ': '))

def weight_variable(shape, init='normsm_1_0', dtype=tf.float64, **kwargs):
    """Create a weight variable with appropriate initialization."""
    initial = parse_init(shape, init, dtype=dtype, **kwargs)
    return tf.Variable(initial)

def parse_init(shape, init, dtype=np.float64, **kwargs):
    if isinstance(init, str) or isinstance(init, unicode):
        if init.startswith('normsm'):
            __, s, m = init.split('_')
            initial = tf.random_normal(shape, dtype=dtype, mean=float(m), stddev=float(s), **kwargs)
        elif init == 'glorot_normal':
            initial = tf.glorot_normal_initializer()(shape)

    elif isinstance(init, np.ndarray):
        initial = tf.constant(init, dtype=dtype)
    try:
        initial
    except:
        raise Exception('Could not parse init {!s}'.format(init))
    return initial


def bias_variable(shape, init='normsm_1_0', dtype=tf.float64, **kwargs):
    """Create a bias variable with appropriate initialization."""
    initial = parse_init(shape, init, dtype=dtype, **kwargs)
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
             dtype=tf.float32, debug=False, weight_init='normsm_1_0', bias_init='normsm_1_0'):
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
