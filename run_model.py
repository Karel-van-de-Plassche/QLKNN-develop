import json
import numpy as np
from IPython import embed
import os
from collections import OrderedDict
import pandas as pd
#def NNlayer(inp, weights, biases, act):
#    preactivate = np.matmul(inp, weights) + biases
#    activations = act(preactivate)
#    return activations

def sigm(x):
    return 1./(1 + np.exp(-1 * x))

def sigm(x):
    return 2./(1 + np.exp(-2 * x)) - 1

class QuaLiKizNDNN():
    def __init__(self, nn_dict):
        parsed = {}
        for name, value in nn_dict.items():
            if value.__class__ == list:
                parsed[name] = np.array(value)
            else:
                parsed[name] = dict(value)
        for name in ['prescale_bias', 'prescale_factor', 'feature_min', 'feature_max', 'feature_names', 'target_names']:
            setattr(self, name, pd.Series(parsed.pop(name), name=name))
        self.layers = []
        for ii in range(1, len(parsed)+1):
            try:
                name = 'layer' + str(ii)
                weight = parsed.pop(name + '/weights/Variable:0')
                bias = parsed.pop(name + '/biases/Variable:0')
                act = sigm
                self.layers.append(QuaLiKizNDNN.NNLayer(weight, bias, act))
            except KeyError:
                pass
                
        assert not any(parsed), 'nn_dict not fully parsed!'

    def apply_layers(self, input):
        for layer in self.layers:
            input = layer.apply(input)
        return input


    class NNLayer():
        def __init__(self, weight, bias, act):
            self.weight = weight
            self.bias = bias
            self.act = act

        def apply(self, input):
            preact = np.matmul(input, self.weight) + self.bias
            result = self.act(preact)
            return result

        def shape(self):
            return self.weight.shape

        def __str__(self):
            return ('NNLayer shape ' + str(self.shape()))

    def get_output(self, **kwargs):
        nn_input = pd.DataFrame()
        for name in self.feature_names:
            try:
                value = kwargs.pop(name)
                nn_input[name] = self.prescale_factor[name] * value + self.prescale_bias[name]
            except KeyError as e:
                raise Exception('NN needs \'' + name + '\' as input')
        output = pd.DataFrame()
        for name in self.target_names:
            nn_output = (np.squeeze(self.apply_layers(nn_input)) - self.prescale_bias[name]) / self.prescale_factor[name]
            output[name] = nn_output
        return output

    @classmethod
    def from_json(cls, json_file):
        with open(json_file) as file_:
            dict_ = json.load(file_)
        nn = QuaLiKizNDNN(dict_)
        return nn



if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    nn = QuaLiKizNDNN.from_json(os.path.join(root, 'nn.json'))
    scann = 24
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
    fluxes = nn.get_output(**input)
    embed()
