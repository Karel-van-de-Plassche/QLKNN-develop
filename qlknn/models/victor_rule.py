import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux


Rmin = a = 1
Ro = 3

def victor_rule(gamma_0, x, q, s_hat, gamma_E):
    epsilon = x * a / Ro
    return victor_rule_eps(gamma_0, epsilon, q, s_hat, gamma_E)

def victor_func(epsilon, q, s_hat):
    c = [0, 0.13, 0.09, 0.41, -1.65]
    n = [0, 1, -1, 1]
    return (c[1] * q ** n[1] + c[2] * epsilon ** n[2] + c[3] * s_hat ** n[3] + c[4])

def victor_rule_eps(gamma_0, epsilon, q, s_hat, gamma_E):
    print(victor_func(epsilon, q, s_hat))
    gamma_eff = gamma_0 + victor_func(epsilon, q, s_hat) * gamma_E
    return gamma_eff[:, np.newaxis]

class VictorNN():
    def __init__(self, network, gam_network):
        if nn._feature_names.ne(gam_network._feature_names).any():
            Exception('Supplied NNs have different feature names')
        if not isinstance(network, QuaLiKizNDNN):
            print('WARNING! Untested for network not QuaLiKizNDNN')

        target_names = network._target_names.append(gam_network._target_names, ignore_index=True)
        self._internal_network = QuaLiKizComboNN(target_names, [network, gam_network], lambda *x: np.hstack(x))

        self._target_names = network._target_names
        self._feature_names = self._internal_network._feature_names.append(pd.Series('gamma_E'), ignore_index=True)

    def get_output(self, input, clip_low=True, clip_high=True, low_bound=None, high_bound=None, safe=False, output_pandas=True):
        nn = self._internal_network
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(nn, input, safe, clip_low, clip_high, low_bound, high_bound)
        del input
        if not safe:
            print('WARNING! Applying victor_rule in unsafe mode. gamma_E should be last column!')
            gamma_E = nn_input[:, [-1]]
            nn_input = np.delete(nn_input, -1, 1)
            if len(nn._feature_names) != nn_input.shape[1]:
                raise Exception('Mismatch! shape feature names != shape input ({:d} != {:d})'.format(len(nn._feature_names), nn_input.shape[1]))
        vic_idx = [nn._feature_names[(nn._feature_names == var)].index[0] for var in ['x', 'q', 'smag']]

        output = nn.get_output(nn_input, output_pandas=False, clip_low=False, clip_high=False, safe=False)
        gamma_0_idx = nn._target_names[(nn._target_names == 'gam_leq_GB')].index[0]
        gamma_0 = output[:, [gamma_0_idx]]
        output = np.delete(output, gamma_0_idx, 1)
        gamma_0 = np.clip(gamma_0, 0, None)

        vic_input = nn_input[:, vic_idx]
        full_vic_input = np.hstack([gamma_0, nn_input[:, vic_idx], gamma_E])
        gamma_eff = victor_rule(*full_vic_input.T)

        for ii, name in enumerate(self._target_names):
            if is_flux(name) and not is_pure_flux(name):
                raise Exception('Cannot apply victor rule to non-pure flux {!s}!'.format(name))
            elif is_pure_flux(name):
                output[:, [ii]] = output[:, [ii]] * np.clip(gamma_eff/gamma_0, 0, None)

        if output_pandas is True:
            output = pd.DataFrame(output, columns=self._target_names)
        return output

if __name__ == '__main__':
    scann = 100
    input = pd.DataFrame()
    input['epsilon'] = np.array(np.linspace(1/1,1/33, scann))
    input['q']  = np.full_like(input.iloc[:, 0], 1.4)
    input['s_hat']  = np.full_like(input.iloc[:, 0], 0.4)
    import matplotlib.pyplot as plt
    plt.plot(1/input['epsilon'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    plt.title('s_hat = ' + str(input['s_hat'].iloc[0]) + ', q = ' + str(input['q'].iloc[0]))
    plt.xlabel('1/eps')
    plt.xlim([0, 35])
    plt.ylim([-1.5, 2.5])

    input = pd.DataFrame()
    input['q'] = np.array(np.linspace(0.5, 4.5, scann))
    input['epsilon']  = np.full_like(input.iloc[:, 0], 0.18)
    input['s_hat']  = np.full_like(input.iloc[:, 0], 0.4)
    plt.plot(input['q'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    plt.title('s_hat = ' + str(input['s_hat'].iloc[0]) + ', epsilon = ' + str(input['epsilon'].iloc[0]))
    plt.xlabel('q')
    plt.xlim([0.5, 4.5])
    plt.ylim([-1.5, 0.75])

    input = pd.DataFrame()
    input['s_hat'] = np.array(np.linspace(0, 3, scann))
    input['epsilon']  = np.full_like(input.iloc[:, 0], 0.18)
    input['q']  = np.full_like(input.iloc[:, 0], 1.4)
    plt.plot(input['s_hat'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    plt.title('q = ' + str(input['q'].iloc[0]) + ', epsilon = ' + str(input['epsilon'].iloc[0]))
    plt.xlabel('s_hat')
    plt.xlim([0, 3])
    plt.ylim([-1.2, 0.4])
    #plt.show()

    # Test the function
    root = os.path.dirname(os.path.realpath(__file__))
    nn = QuaLiKizNDNN.from_json('nn.json', layer_mode='classic')

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
    input['Nustar']  = np.full_like(input['Ati'], 0.009995)
    input['x']  = np.full_like(input['Ati'], 0.449951)
    input = input[nn._feature_names]
    input['gamma_E'] = np.full_like(input['Ati'], 1.0)

    nn = VictorNN(nn, QuaLiKizNDNN.from_json('nn_gam.json', layer_mode='classic'))
    fluxes = nn.get_output(input.values, safe=False)

    print(fluxes)
    embed()
