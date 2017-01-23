import json
import numpy as np
from IPython import embed
import os
def NNlayer(inp, weights, biases, act):
    preactivate = np.matmul(inp, weights) + biases
    activations = act(preactivate)
    return activations

def sigm(x):
    return 1./(1 + np.exp(-1 * x))
def NNoutput(inp, net):
    layer1 = NNlayer(inp, net['layer1/weights/Variable:0'], net['layer1/biases/Variable:0'], sigm)
    layer2 = NNlayer(layer1, net['layer2/weights/Variable:0'], net['layer2/biases/Variable:0'], sigm)
    layer3 = NNlayer(layer2, net['layer3/weights/Variable:0'], net['layer3/biases/Variable:0'], sigm)
    return layer3

class QuaLiKiz4DNN():
    def __init__(self):
        root = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(root, 'nn.json')) as file_:
            dict_ = json.load(file_)
        self.net = {}
        for name, value in dict_.items():
            if value.__class__ == list:
                self.net[name] = np.array(value)
            else:
                 self.net[name] = dict(value)

    def get_fluxes(self, Ati, Ti_Te, qx, smag, **kwargs):
        Ati = self.net['scale_factor']['Ati'] * Ati + self.net['scale_bias']['Ati']
        Ti_Te = self.net['scale_factor']['Ti_Te'] * Ti_Te + self.net['scale_bias']['Ti_Te']
        qx = self.net['scale_factor']['qx'] * qx + self.net['scale_bias']['qx']
        smag = self.net['scale_factor']['smag'] * smag + self.net['scale_bias']['smag']
        inp = np.array([Ati, Ti_Te, qx, smag]).T
        chie =  (NNoutput(inp, self.net) - self.net['scale_bias']['efe_GB']) / self.net['scale_factor']['efe_GB']
        #chie =  NNoutput(inp, self.net)
        return chie

if __name__ == '__main__':
    nn = QuaLiKiz4DNN()
    scann = 24
    Ati = np.array(np.linspace(2,13, scann))
    qx = np.full_like(Ati, 2.)
    smag = np.full_like(Ati, 1.)
    Ti_Te = np.full_like(Ati, 1.)
    fluxes = nn.get_fluxes(Ati, Ti_Te, qx, smag)
    embed()
