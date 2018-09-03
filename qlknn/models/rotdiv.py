import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux, split_parts

pot_rot_vars = ['Machtor', 'Autor', 'Machpar', 'Aupar', 'gammaE']

class RotDivNN():
    def __init__(self, network, rot0_div_network, allow_negative=True):
        """ Initialize a
        """
        if len(rot0_div_network._target_names) > 1 or len(network._target_names) > 1:
            raise NotImplementedError('Multi-target RotDivNNs. Passed {!s} and {!s}'.format(
                rot0_div_network._target_names, network._target_names))
        rot_target_name = rot0_div_network._target_names[0]
        target_name = network._target_names[0]
        if not rot_target_name.endswith('_rot0'):
            raise NotImplementedError('RotDivNN with rot0 var not at the end. Passed {!s}'.format(
                rot_target_name))
        if any(var in pot_rot_vars for var in network._feature_names):
            raise ValueError('Networks has a rotation var as feature! Weird! {!s}'.format(network._feature_names))
        num, op, denum = split_parts(rot_target_name[:-5])
        if op != '_div_':
            raise ValueError('rot0_div_network should be a div network, not {!s}'.format(
                rot_target_name))
        if denum != target_name:
            raise ValueError('Cannot combine network {!s} with rotdiv {!s}'.format(
                target_name, rot_target_name))
        rotvars = [var for var in rot0_div_network._feature_names if var in pot_rot_vars]
        if len(rotvars) != 1:
            raise ValueError('rotdiv network should have exactly one rot var, has {!s}'.format(rot0_div_network._feature_names))
        else:
            self.rotvar = rotvar = rotvars[0]

        #if network._feature_names.ne(gam_network._feature_names).any():
        #    Exception('Supplied NNs have different feature names')

        self._internal_network = network
        self._rotdiv_network = rot0_div_network

        self._target_names = network._target_names
        self._feature_names = self._internal_network._feature_names.append(pd.Series(rotvar), ignore_index=True)

        # Copy parts of internal network
        self._feature_min = self._internal_network._feature_min
        self._feature_min[rotvar] = self._rotdiv_network._feature_min[rotvar]
        self._feature_max = self._internal_network._feature_max
        self._feature_max[rotvar] = self._rotdiv_network._feature_max[rotvar]
        self._target_min = self._internal_network._target_min
        self._target_max = self._internal_network._target_max

        self.allow_negative = allow_negative

    def get_output(self, input, clip_low=False, clip_high=False, low_bound=None, high_bound=None, safe=True, output_pandas=True, apply_rotation=True):
        nn = self._internal_network
        if not safe:
            raise NotImplementedError('Unsafe RotDivNN')
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(nn, input, safe, clip_low, clip_high, low_bound, high_bound)
        if self.allow_negative:
            clip_kwargs = {}
        else:
            clip_kwargs = {'clip_low': True,
                           'low_bound': pd.Series({k: 0 for k in self._internal_network._target_names + self._rotdiv_network._target_names})}
        # Get indices for vars that victor rule needs: x, q, smag
        nn_out = self._internal_network.get_output(input[self._internal_network._feature_names].values, safe=False, output_pandas=False, **clip_kwargs)
        if apply_rotation:
            rot_out = self._rotdiv_network.get_output(input[self._rotdiv_network._feature_names].values, safe=False, output_pandas=False, **clip_kwargs)
            output = nn_out * rot_out

        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names)
        return output

if __name__ == '__main__':
    scann = 100

    root = os.path.dirname(os.path.realpath(__file__))
    nn_ITG = QuaLiKizNDNN.from_json('../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json', layer_mode='classic')
    nn_ITG_rot0 = QuaLiKizNDNN.from_json('../../tests/gen4_test_files/Network_xxx_efiITG_GB_div_efiITG_GB_rot0/nn.json', layer_mode='classic')
    nn = RotDivNN(nn_ITG, nn_ITG_rot0, allow_negative=False)

    scann = 100
    input = pd.DataFrame()
    input['Ati'] = np.array(np.linspace(2,13, scann))
    input['Ti_Te']  = np.full_like(input['Ati'], 1.)
    input['Zeff']  = np.full_like(input['Ati'], 1.)
    input['An']  = np.full_like(input['Ati'], 2.)
    input['Ate']  = np.full_like(input['Ati'], 5.)
    input['q'] = np.full_like(input['Ati'], 0.660156)
    input['smag']  = np.full_like(input['Ati'], 0.399902)
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['logNustar']  = np.full_like(input['Ati'], np.log10(0.009995))
    input['x']  = np.full_like(input['Ati'], 0.449951)
    #input = input.loc[:, nn_ITG._feature_names]
    input['Machtor'] = np.full_like(input['Ati'], 0.3)
    fluxes = nn.get_output(input, safe=True)
    print(fluxes)
