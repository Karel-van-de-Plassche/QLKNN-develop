import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux

class ClipNN():
    def __init__(self, network, clip_var):
        if not isinstance(network, QuaLiKizComboNN):
            print('WARNING! Untested for network not QuaLiKizCombo')

        self._clip_var = clip_var
        self._internal_network = network

        self._target_names = self._internal_network._target_names
        self._feature_names = self._internal_network._feature_names

    def get_output(self, input, clip_low=True, clip_high=True, low_bound=None, high_bound=None, safe=False, output_pandas=True):
        nn = self._internal_network
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(nn, input, safe, clip_low, clip_high, low_bound, high_bound)
        del input

        output = nn.get_output(nn_input, output_pandas=False, clip_low=False, clip_high=False, safe=False)
        clip_idx = nn._target_names[(nn._target_names == self._clip_var)].index[0]
        output[output[:, clip_idx] < 0, :] = 0

        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names)
        return output

if __name__ == '__main__':
    scann = 100
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    # Test the function
    root = os.path.dirname(os.path.realpath(__file__))
    nn1 = QuaLiKizNDNN.from_json('../../tests/gen2_test_files/network_1393/nn.json', layer_mode='classic')
    nn2 = QuaLiKizNDNN.from_json('../../tests/gen2_test_files/network_1440/nn.json', layer_mode='classic')
    target_names = nn1._target_names.append(nn2._target_names, ignore_index=True)
    nn_combo = QuaLiKizComboNN(target_names, [nn1, nn2], lambda *x: np.hstack(x))
    nn = ClipNN(nn_combo, 'efiITG_GB')

    scann = 100
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeff']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    #input['q'] = np.full_like(input['Ati'], 0.660156)
    input['qx'] = np.full_like(input['Ati'], 1.4)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    input = input[nn._feature_names]

    fluxes = nn.get_output(input.values, safe=False)

    print(fluxes)
    embed()
