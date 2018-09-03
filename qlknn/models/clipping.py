import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux, split_name

def xorxor(list_of_arrays):
    lenlist = [len(arr) for arr in list_of_arrays]
    if lenlist[1:] != lenlist[:-1]:
        raise ValueError('Passed lists have unequal lenghts')
    full_xor = np.full_like(list_of_arrays[0], False)
    for arr in list_of_arrays:
        full_xor = np.logical_xor(full_xor, arr)
    return full_xor

leading_flux_dict = {'ETG': 'efeETG',
                     'ITG': 'efiITG',
                     'TEM': 'efeTEM'}
class LeadingFluxNN(QuaLiKizNDNN):
    def __init__(self, network):
        if not isinstance(network, QuaLiKizComboNN):
            print('WARNING! Untested for network not QuaLiKizCombo')

        #clip_identifiers = [split_name(name)[2] for name in leading_fluxes]
        modes = ['ETG', 'ITG', 'TEM']
        self._clip_idx = {id: np.flatnonzero(network._target_names.apply(lambda x: x.find(id) >= 0)) for id in modes}
        clip_identifiers = [key for key, val in self._clip_idx.items() if len(val) > 0]
        leading_fluxes = ['_'.join([name, 'GB']) for mode, name in leading_flux_dict.items() if mode in clip_identifiers]
        if not set(np.hstack(self._clip_idx.values())) == set(range(0, len(network._target_names))):
            raise Exception('Unexpected target_names {!s} for identifiers {!s}'
                            .format(network._target_names.tolist(), clip_identifiers))
        self._leading_idx = {id: np.squeeze(np.flatnonzero(network._target_names == lead_id), 0) for id, lead_id in zip(clip_identifiers, leading_fluxes)}
        lenlist = [arr.size for arr in self._leading_idx.values()]
        if not (lenlist[1:] == lenlist[:-1] and lenlist[0] == 1):
            raise Exception('Could not find leading fluxes {!s} in target_names {!s}'
                            .format(leading_fluxes, network._target_names.tolist()))
        self._internal_network = network

        # Copy parts of internal network
        self._target_names = self._internal_network._target_names
        self._feature_names = self._internal_network._feature_names
        self._feature_min = self._internal_network._feature_min
        self._feature_max = self._internal_network._feature_max
        self._target_min = self._internal_network._target_min
        self._target_max = self._internal_network._target_max

    def get_output(self, input, clip_low=False, clip_high=False, low_bound=None, high_bound=None, safe=True, output_pandas=True):
        nn = self._internal_network
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(nn, input, safe, clip_low, clip_high, low_bound, high_bound)
        del input

        output = nn.get_output(nn_input, output_pandas=False, clip_low=False, clip_high=False, safe=safe)
        for id in self._leading_idx.keys():
            leading_idx = self._leading_idx[id]
            clip_idx = self._clip_idx[id]
            output[np.ix_(output[:, leading_idx] <= 0, clip_idx)] = 0

        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names)
        return output

    @staticmethod
    def add_leading_flux_clipping(network):
        network = LeadingFluxNN(network)
        return network

if __name__ == '__main__':
    # Test the function
    test_root = '../../tests/gen3_test_files/'
    nn_efi = QuaLiKizNDNN.from_json(test_root + 'Network_874_efiITG_GB/nn.json', layer_mode='classic')
    nn_div = QuaLiKizNDNN.from_json(test_root + 'Network_302_efeITG_GB_div_efiITG_GB/nn.json', layer_mode='classic')
    nn_efe = QuaLiKizComboNN('efeITG_GB', [nn_efi, nn_div], lambda *x: x[0] * x[1])
    target_names = nn_efe._target_names.append(nn_efi._target_names, ignore_index=True)
    nn_combo = QuaLiKizComboNN(target_names, [nn_efe, nn_efi], lambda *x: np.hstack(x))
    nn = LeadingFluxNN(nn_combo)

    #from qlknn.NNDB.model import Network
    #nn_combo = Network.get_by_id(1551).to_QuaLiKizNN()
    #nn = LeadingFluxNN(nn_combo)

    scann = 100
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeff']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    #input['q'] = np.full_like(input['Ati'], 0.660156)
    input['q'] = np.full_like(input['Ati'], 1.4)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['logNustar']  = np.full_like(input['Ati'], np.log10(0.009995))
    input['x']  = np.full_like(input['Ati'], 0.449951)
    input = input[nn._feature_names]

    pd.set_option('display.max_columns', 50)
    fluxes_clipped = nn.get_output(input.values, safe=False)
    fluxes = nn_combo.get_output(input.values, safe=False)
    fluxes = fluxes.merge(fluxes_clipped, left_index=True, right_index=True, suffixes=('', '_clip'))

    print(fluxes)
    embed()
