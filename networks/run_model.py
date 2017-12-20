#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import numpy as np
from IPython import embed
import os
from collections import OrderedDict
import pandas as pd
from warnings import warn
def sigm_tf(x):
    return 1./(1 + np.exp(-1 * x))

#def sigm(x):
#    return 2./(1 + np.exp(-2 * x)) - 1

class QuaLiKizMultiNN():
    def __init__(self, nns):
        self._nns = nns
        feature_names = nns[0]
        for nn in self._nns:
            if len(nn._target_names) == 1:
                name = nn._target_names[0]
            else:
                NotImplementedError('Multitarget not implemented yet')
            if np.all(nn._feature_names.ne(feature_names)):
                Exception('Supplied NNs have different feature names')
        if np.any(self._feature_min > self._feature_max):
            raise Exception('Feature min > feature max')
        self._target_min = pd.concat(
            [nn._target_min for nn in self._nns])
        self._target_max = pd.concat(
            [nn._target_max for nn in self._nns])

    @property
    def _target_names(self):
        targets = []
        for nn in self._nns:
            targets.extend(list(nn._target_names))
        return targets

    def get_output(self, input, output_pandas=True, clip_low=True, clip_high=True, low_bound=None, high_bound=None, **kwargs):
        results = pd.DataFrame()
        feature_max = -np.inf
        feature_min = np.inf
        out_tot = np.empty((input.shape[0], len(self._nns)))
        out_name = []
        nn_input, kwargs['safe'], clip_low, clip_high, low_bound, high_bound = \
            determine_settings(self, input, kwargs['safe'], clip_low, clip_high, low_bound, high_bound)
        for ii, nn in enumerate(self._nns):
            if len(nn._target_names) == 1:
                out = nn.get_output(input, clip_low=False, clip_high=False, **kwargs)
                out_tot[:, ii] = np.squeeze(out)
                if output_pandas:
                    out_name.extend(out.columns.values)
            elif target in nn.target_names.values:
                NotImplementedError('Multitarget not implemented yet')


        out_tot = clip_to_bounds(out_tot, clip_low=clip_low, clip_high=clip_high, low_bound=low_bound, high_bound=high_bound)
        if output_pandas == True:
            results = pd.DataFrame(out_tot, columns=out_name)
        else:
            results = out_tot
        return results

    @property
    def _target_names(self):
        target_names = []
        for nn in self._nns:
            target_names.extend(nn._target_names)
        return list(set(target_names))

    @property
    def _feature_names(self):
        return self._nns[0]._feature_names

    @property
    def _feature_max(self):
        feature_max = pd.Series(np.full_like(self._nns[0]._feature_max, np.inf),
                                index=self._nns[0]._feature_max.index)
        for nn in self._nns:
            feature_max = nn._feature_max.combine(feature_max, min)
        return feature_max

    @property
    def _feature_min(self):
        feature_min = pd.Series(np.full_like(self._nns[0]._feature_min, -np.inf),
                                index=self._nns[0]._feature_min.index)
        for nn in self._nns:
            feature_min = nn._feature_min.combine(feature_min, max)
        return feature_min

class QuaLiKizComboNN():
    def __init__(self, target_names, nns, combo_func):
        self._nns = nns
        feature_names = nns[0]
        for nn in self._nns:
            if np.all(nn._feature_names.ne(feature_names)):
                Exception('Supplied NNs have different feature names')
        if np.any(self._feature_min > self._feature_max):
            raise Exception('Feature min > feature max')

        self._combo_func = combo_func
        self._target_names = target_names
        self._target_min = pd.Series(
            self._combo_func(*[nn._target_min.values for nn in nns]),
            index=self._target_names)
        self._target_max = pd.Series(
            self._combo_func(*[nn._target_max.values for nn in nns]),
            index=self._target_names)

    def get_output(self, input, output_pandas=True, clip_low=True, clip_high=True, low_bound=None, high_bound=None, **kwargs):
        nn_input, kwargs['safe'], clip_low, clip_high, low_bound, high_bound = \
            determine_settings(self, input, kwargs['safe'], clip_low, clip_high, low_bound, high_bound)
        output = self._combo_func(*[nn.get_output(input, output_pandas=False, clip_low=False, clip_high=False, **kwargs) for nn in self._nns])
        output = clip_to_bounds(output, clip_low=clip_low, clip_high=clip_high, low_bound=low_bound, high_bound=high_bound)
        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names)
        return output

    @property
    def _feature_names(self):
        return self._nns[0]._feature_names

    @property
    def _feature_max(self):
        feature_max = pd.Series(np.full_like(self._nns[0]._feature_max, np.inf),
                                index=self._nns[0]._feature_max.index)
        for nn in self._nns:
            feature_max = nn._feature_max.combine(feature_max, min)
        return feature_max

    @property
    def _feature_min(self):
        feature_min = pd.Series(np.full_like(self._nns[0]._feature_min, -np.inf),
                                index=self._nns[0]._feature_min.index)
        for nn in self._nns:
            feature_min = nn._feature_min.combine(feature_min, max)
        return feature_min

class QuaLiKizDuoNN():
    def __init__(self, target_names, nn1, nn2, combo_funcs):
        self._nn1 = nn1
        self._nn2 = nn2
        if np.any(self._feature_min > self._feature_max):
            raise Exception('Feature min > feature max')
        if np.all(nn1._feature_names.ne(nn2._feature_names)):
            raise Exception('Supplied NNs have different feature names')
        if not len(target_names) == len(combo_funcs):
            raise Exception('len(target_names) = {.f} and len(combo_func) = {.f}'
                            .format(len(target_names),  len(combo_funcs)))
        self._combo_funcs = combo_funcs
        self._target_names = target_names

    def get_output(self, input, **kwargs):
        output = pd.DataFrame()
        output1 = self._nn1.get_output(input, **kwargs)
        output2 = self._nn2.get_output(input, **kwargs)
        for target_name, combo_func in zip(self._target_names, self._combo_funcs):
            output[target_name] = np.squeeze(combo_func(output1, output2))
        return output

    @property
    def _feature_names(self):
        return self._nn1._feature_names

    @property
    def _feature_max(self):
        return self._nn1._feature_max.combine(self._nn2._feature_max, min)

    @property
    def _feature_min(self):
        return self._nn1._feature_min.combine(self._nn2._feature_min, max)

class QuaLiKizNDNN():
    def __init__(self, nn_dict, target_names_mask=None, layer_mode=None):
        """ General ND fully-connected multilayer perceptron neural network

        Initialize this class using a nn_dict. This dict is usually read
        directly from JSON, and has a specific structure. Generate this JSON
        file using the supplied function in QuaLiKiz-Tensorflow
        """
        parsed = {}
        try:
            import qlknn
        except:
            layer_mode = 'classic'
        else:
            layer_mode = 'intel'

        # Read and parse the json. E.g. put arrays in arrays and the rest in a dict
        for name, value in nn_dict.items():
            if name == 'hidden_activation' or name == 'output_activation':
                parsed[name] = value
            elif value.__class__ == list:
                parsed[name] = np.array(value)
            else:
                parsed[name] = dict(value)
        # These variables do not depend on the amount of layers in the NN
        for set in ['feature', 'target']:
            setattr(self, '_' + set + '_names', pd.Series(parsed.pop(set + '_names')))
        for set in ['feature', 'target']:
            for subset in ['min', 'max']:
                setattr(self, '_'.join(['', set, subset]), pd.Series(parsed.pop('_'.join([set, subset])))[getattr(self, '_' + set + '_names')])
        for subset in ['bias', 'factor']:
            setattr(self, '_'.join(['_feature_prescale', subset]), pd.Series(parsed['prescale_' + subset])[self._feature_names])
            setattr(self, '_'.join(['_target_prescale', subset]), pd.Series(parsed.pop('prescale_' + subset))[self._target_names])
        self.layers = []
        # Now find out the amount of layers in our NN, and save the weigths and biases
        activations = parsed['hidden_activation'] + [parsed['output_activation']]
        for ii in range(1, len(activations) + 1):
            try:
                name = 'layer' + str(ii)
                weight = parsed.pop(name + '/weights/Variable:0')
                bias = parsed.pop(name + '/biases/Variable:0')
                activation = activations.pop(0)
                if layer_mode == 'classic':
                    if activation == 'tanh':
                        act = np.tanh
                    elif activation == 'relu':
                        act = _act_relu
                    elif activation == 'none':
                        act = _act_none
                    self.layers.append(QuaLiKizNDNN.NNLayer(weight, bias, act))
                elif layer_mode  == 'intel':
                    self.layers.append(qlknn.Layer(weight, bias, activation))
            except KeyError:
                # This name does not exist in the JSON,
                # so our previously read layer was the output layer
                break
        if len(activations) == 0:
            del parsed['hidden_activation']
            del parsed['output_activation']
        try:
            self._clip_bounds = parsed['_metadata']['clip_bounds']
        except KeyError:
            self._clip_bounds = False

        self._target_names_mask = target_names_mask
        # Ignore metadata
        try:
            self._metadata = parsed.pop('_metadata')
        except KeyError:
            pass
        if any(parsed):
            warn('nn_dict not fully parsed! ' + str(parsed))

    def apply_layers(self, input, output=None):
        """ Apply all NN layers to the given input

        The given input has to be array-like, but can be of size 1
        """
        input = np.ascontiguousarray(input)
        # 3x30 network:
        #14.1 µs ± 913 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        #20.9 µs ± 2.43 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        #19.1 µs ± 240 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        #2.67 µs ± 29.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

        for layer in self.layers:
            output = np.empty([input.shape[0], layer._weights.shape[1]])
            output = layer.apply(input, output)
            input = output
        return input


    class NNLayer():
        """ A single (hidden) NN layer
        A hidden NN layer is just does

        output = activation(weight * input + bias)

        Where weight is generally a matrix; output, input and bias a vector
        and activation a (sigmoid) function.
        """
        def __init__(self, weight, bias, activation):
            self._weights = weight
            self._biases = bias
            self._activation = activation
            #@jit(float64[:,:](float64[:,:]), nopython=True)
            #def _apply_layer(input):
            #    preactivation = np.dot(input, weight) + bias
            #    result = activation(preactivation)
            #    return result
            #self.apply = lambda input: activation(np.dot(input, weight) + bias)
            #_create_apply(weight, bias, activation)

        def apply(self, input, output=None):
            preactivation = np.dot(input, self._weights) + self._biases
            result = self._activation(preactivation)
            return result

        def shape(self):
            return self.weight.shape

        def __str__(self):
            return ('NNLayer shape ' + str(self.shape()))

    def get_output(self, input, clip_low=True, clip_high=True, low_bound=None, high_bound=None, safe=True, output_pandas=True):
        """ Calculate the output given a specific input

        This function accepts inputs in the form of a dict with
        as keys the name of the specific input variable (usually
        at least the feature_names) and as values 1xN same-length
        arrays.
        """
        #49.1 ns ± 1.53 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(self, input, safe, clip_low, clip_high, low_bound, high_bound)

        #nn_input = self._feature_prescale_factors.values[np.newaxis, :] * nn_input + self._feature_prescale_biases.values
        #14.3 µs ± 1.08 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        nn_input = _prescale(nn_input,
                             self._feature_prescale_factor.values,
                             self._feature_prescale_bias.values)

        # Apply all NN layers an re-scale the outputs
        # 104 µs ± 19.7 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        # 70.9 µs ± 384 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each) (only apply layers)
        output = (self.apply_layers(nn_input) - np.atleast_2d(self._target_prescale_bias)) / np.atleast_2d(self._target_prescale_factor)
        #for name in self._target_names:
        #    nn_output = (np.squeeze(self.apply_layers(nn_input)) - self._target_prescale_biases[name]) / self._target_prescale_factors[name]
        #    output[name] = nn_output
        output = clip_to_bounds(output, clip_low=clip_low, clip_high=clip_high, low_bound=low_bound, high_bound=high_bound)

        # 118 µs ± 3.83 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        if output_pandas:
            output = pd.DataFrame(output, columns=self._target_names)

        # 47.4 ns ± 1.79 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        if self._target_names_mask is not None:
            output.columns = self._target_names_mask
        return output

    @classmethod
    def from_json(cls, json_file, **kwargs):
        with open(json_file) as file_:
            dict_ = json.load(file_)
        nn = QuaLiKizNDNN(dict_, **kwargs)
        return nn

    @property
    def l2_norm(self):
        l2_norm = 0
        for layer in self.layers:
            l2_norm += np.sum(np.square(layer.weight))
        l2_norm /= 2
        return l2_norm

    @property
    def l1_norm(self):
        l1_norm = 0
        for layer in self.layers:
            l1_norm += np.sum(np.abs(layer.weight))
        return l1_norm

def clip_to_bounds(output, clip_low, clip_high, low_bound, high_bound):
    if clip_low:
        for ii, bound in enumerate(low_bound):
            output[:, ii][output[:, ii] < bound] = bound

    if clip_high:
        for ii, bound in enumerate(high_bound):
            output[:, ii][output[:, ii] > bound] = bound

    return output

def determine_settings(network, input, safe, clip_low, clip_high, low_bound, high_bound):
        if safe:
            if input.__class__ == pd.DataFrame:
                nn_input = input[network._feature_names]
            else:
                raise Exception('Please pass a pandas.DataFrame for safe mode')
            if low_bound is not None:
                low_bound = low_bound[network._target_names].values
            if high_bound is not None:
                high_bound = high_bound[network._target_names].values
        else:
            if input.__class__ == pd.DataFrame:
                nn_input = input.values
            elif input.__class__ == np.ndarray:
                nn_input = input

        if clip_low is True and (low_bound is None):
            low_bound = network._target_min.values
        if clip_high is True and (high_bound is None):
            high_bound = network._target_max.values
        return nn_input, safe, clip_low, clip_high, low_bound, high_bound

#@jit(float64[:,:](float64[:,:], float64[:], float64[:]), nopython=True)
def _prescale(nn_input, factors, biases):
    return np.atleast_2d(factors) * nn_input + biases
#    #return factors[np.newaxis, :] * nn_input + biases
#
#@jit(float64[:,:](float64[:,:]), nopython=True)
def _act_none(x):
    return x
#
#@jit(float64[:,:](float64[:,:]), nopython=True)
def _act_relu(x):
    return x * (x > 0)
#
##@jit(float64[:,:](float64[:,:], float64[:,:,:]), nopython=True)
##def _apply_layers(self, input, layers):
##    for layer in layers:
##        input = layer.apply(input)
##    return input
#
#def _create_apply(weight, bias, activation):
#    #self.weight = weight
#    #self.bias = bias
#    #self.activation = activation
#    #if activation is None:
#    #    @jit(float64[:,:](float64[:,:]), nopython=True)
#    #    def _apply_layer(input):
#    #        preactivation = np.dot(input, weight) + bias
#    #        result = preactivation
#    #        return result
#    #else:
#    @jit(float64[:,:](float64[:,:]), nopython=True)
#    def _apply_layer(input):
#        preactivation = np.dot(input, weight) + bias
#        result = activation(preactivation)
#        return result
#
#    return _apply_layer

if __name__ == '__main__':
    # Test the function
    root = os.path.dirname(os.path.realpath(__file__))
    #nn1 = QuaLiKizNDNN.from_json(os.path.join(root, 'nn_efe_GB.json'))
    #nn2 = QuaLiKizNDNN.from_json(os.path.join(root, 'nn_efi_GB.json'))
    #nn3 = QuaLiKizDuoNN('nn_eftot_GB', nn1, nn2, lambda x, y: x + y)
    #nn = QuaLiKizMultiNN([nn1, nn2])
    nn = QuaLiKizNDNN.from_json('nn.json', layer_mode='intel')

    scann = 100
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeffx']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['qx'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    input = input[nn._feature_names]

    fluxes = nn.get_output(input.values, safe=False)

    nn2 = QuaLiKizNDNN.from_json('nn.json', layer_mode='classic')
    fluxes2 = nn2.get_output(input.values, safe=False)

    #print(fluxes)

    #import qlknn;
    layer0 = nn.layers[0]
    out = np.full((input.shape[0], layer0._weights.shape[1]), 123.);
    embed()
