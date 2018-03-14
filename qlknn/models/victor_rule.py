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
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    #input = pd.DataFrame()
    #input['epsilon'] = np.array(np.linspace(1/1,1/33, scann))
    #input['q']  = np.full_like(input.iloc[:, 0], 1.4)
    #input['s_hat']  = np.full_like(input.iloc[:, 0], 0.4)
    #plt.plot(1/input['epsilon'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    #plt.title('s_hat = ' + str(input['s_hat'].iloc[0]) + ', q = ' + str(input['q'].iloc[0]))
    #plt.xlabel('1/eps')
    #plt.xlim([0, 35])
    #plt.ylim([-1.5, 2.5])

    def plot_victorplot(epsilon, q, s_hat, gamma_0, plotvar):
        n = 100
        idx = pd.MultiIndex.from_product([np.linspace(0,1,n), epsilon, q, s_hat], names=['gamma_E', 'epsilon', 'q', 's_hat'])
        data = pd.DataFrame(index=idx)
        data.reset_index(inplace=True)
        data['f_vic'] = victor_func(*data.loc[:, ('epsilon', 'q', 's_hat')].values.T)
        data['gamma_0'] = np.tile(gamma_0, [1, n]).T
        data['line'] = data['gamma_0'] + data['f_vic'] * data['gamma_E']
        data['line'].clip(0, inplace=True)
        gamma_E_plot = data.pivot(index='gamma_E', columns=plotvar, values='line')
        if plotvar == 'epsilon':
            gamma_E_plot = gamma_E_plot[gamma_E_plot.columns[::-1]]
            cmap = ListedColormap(['C1', 'C0', 'C2', 'C4', 'C3', 'C8'])
        else:
            cmap = ListedColormap(['C1', 'C2', 'C0', 'C4', 'C3', 'C8'])
        style = [':'] * data[plotvar].unique().size
        gamma_E_plot.plot(colormap=cmap, style=style)
        plt.show()
    plot_victorplot([0.03, 0.05, 0.1, 0.18, 0.26, 0.35], [1.4], [0.4], [0.22, 0.27, 0.4, 0.57, 0.65, 0.71], 'epsilon')
    plot_victorplot([0.18], [0.73, 1.4, 2.16, 2.88, 3.60, 4.32], [0.4], [0.27, 0.5, 0.64, 0.701, 0.74, 0.76], 'q')
    plot_victorplot([0.18], [0.73, 1.4, 2.16, 2.88, 3.60, 4.32], [0.8], [0.34, 0.54, 0.64, 0.69, 0.71, 0.73], 'q')
    plot_victorplot([0.18], [1.4], [0.2, 0.7, 1.2, 1.7, 2.2, 2.7], [.92, 1.18, 1.07, 0.85, 0.63, 0.52], 's_hat')


    #input = pd.DataFrame()
    #input['q'] = np.array(np.linspace(0.5, 4.5, scann))
    #input['epsilon']  = np.full_like(input.iloc[:, 0], 0.18)
    #input['s_hat']  = np.full_like(input.iloc[:, 0], 0.4)
    #plt.plot(input['q'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    #plt.title('s_hat = ' + str(input['s_hat'].iloc[0]) + ', epsilon = ' + str(input['epsilon'].iloc[0]))
    #plt.xlabel('q')
    #plt.xlim([0.5, 4.5])
    #plt.ylim([-1.5, 0.75])

    #input = pd.DataFrame()
    #input['s_hat'] = np.array(np.linspace(0, 3, scann))
    #input['epsilon']  = np.full_like(input.iloc[:, 0], 0.18)
    #input['q']  = np.full_like(input.iloc[:, 0], 1.4)
    #plt.plot(input['s_hat'], victor_func(*input.loc[:, ['epsilon', 'q', 's_hat']].values.T))
    #plt.title('q = ' + str(input['q'].iloc[0]) + ', epsilon = ' + str(input['epsilon'].iloc[0]))
    #plt.xlabel('s_hat')
    #plt.xlim([0, 3])
    #plt.ylim([-1.2, 0.4])
    #plt.show()

    # Test the function
    root = os.path.dirname(os.path.realpath(__file__))
    nn = QuaLiKizNDNN.from_json('../../tests/gen2_test_files/network_1393/nn.json', layer_mode='classic')

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
