import os

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, determine_settings
from qlknn.misc.analyse_names import is_pure_flux, is_flux

Rmin = a = 1
Ro = 3
mp = 1.672621777e-27
qe = 1.602176565e-19 #SI electron charge
mi = lambda Ai: Ai * mp #Ai ion atom number, mp mass proton
Te_SI = lambda Te: qe * Te * 1e3 # Te keV
c_sou = lambda Te, Ai0: np.sqrt(Te_SI(Te)/mi(Ai0)) #Te keV, Ai atom number first ion
gamGB = lambda Te: csou(Te, mi) / Rmin
c_ref = np.sqrt(qe*1e3/mp)
eps = lambda x: x * a / Ro

def gammaE_GB_to_gamma_QLK_in(gammaE_GB, Te, Ai0):
    """
    Args:
        gammaE_GB:   GyroBohm normalized gammaE
        Te:           Electron temperature (keV)
        Ai0:          Massnumber of main ion
    """
    gammaE_QLK = gammaE_GB * (Ro * c_sou(Te, Ai0)) / (a * c_ref)
    return gammaE_QLK

def gammaE_QLK_to_gammaE_GB(gammaE_QLK, Te, Ai0):
    """
    Args:
        gammaE_QLK:  gammaE normalized QLK-style
        Te:           Electron temperature (keV)
        Ai0:          Massnumber of main ion
    """
    return gammaE_QLK * (a * c_ref) / (Ro * c_sou(Te, Ai0))

def apply_victor_rule(gamma0, x, q, s_hat, gammaE_GB):
    """ Apply victor rule with x instead of epsilon. See victor_rule_eps."""
    gamma0_lower_bound = 1e-4
    gamma0 = np.clip(gamma0, gamma0_lower_bound, None)
    f_victorthesis = apply_victorthesis(x, q, s_hat)
    fvic = np.clip(1 + f_victorthesis * gammaE_GB / gamma0, 0, None)
    fvic[gamma0 == gamma0_lower_bound] = 0
    return fvic[:, np.newaxis]

def apply_victorthesis(x, q, s_hat):
    f_victorthesis = apply_victorthesis_eps(eps(x), q, s_hat)
    return f_victorthesis

def apply_victorthesis_eps(epsilon, q, s_hat):
    """ Return f(eps, q, s_hat) as defined by victor rule"""
    c = [0, 0.13, 0.09, 0.41, -1.65]
    n = [0, 1, -1, 1]
    return (c[1] * q ** n[1] + c[2] * epsilon ** n[2] + c[3] * s_hat ** n[3] + c[4])

#def scale_with_victor(gamma0, x, q, s_hat, gammaE_GB):
#    fvic = apply_victor_rule(gamma0, x, q, s_hat, gammaE_GB)

#def victor_rule_eps(gamma0, epsilon, q, s_hat, gamma_E_GB):
#    """ Return \gamma_{eff} as defined by victor rule
#    Args:
#        gamma0: Rotationless growth rate (As defined by Victor)
#        epsilon: Normalized radial location
#        q: Safety factor
#        s_hat: Magnetic shear (As defined by Victor)
#        gamma_E: Parallel flow shear (As defined by Victor)
#    """
#    gamma_eff = gamma0 + victor_func(epsilon, q, s_hat) * gamma_E_GB
#    return gamma_eff[:, np.newaxis]

class VictorNN():
    def __init__(self, network, gam_network):
        """ Initialize a victor network from a general QuaLiKizNDNN
        Args:
            network: A QuaLiKizComboNN. Should work for QuaLiKizNDNN, but untested.
                     Network should only have pure fluxes! No divs or sums!
            gam_network: A network that predicts gamma_leq_GB, the maximum
                         ion scale growth rate
        """
        if network._feature_names.ne(gam_network._feature_names).any():
            Exception('Supplied NNs have different feature names')
        if not isinstance(network, QuaLiKizNDNN):
            print('WARNING! Untested for network not QuaLiKizNDNN')

        target_names = network._target_names.append(gam_network._target_names, ignore_index=True)
        self._internal_network = QuaLiKizComboNN(target_names, [network, gam_network], lambda *x: np.hstack(x))

        self._target_names = network._target_names
        self._feature_names = self._internal_network._feature_names.append(pd.Series('gammaE'), ignore_index=True)

    def get_output(self, input, clip_low=False, clip_high=False, low_bound=None, high_bound=None, safe=False, output_pandas=True):
        nn = self._internal_network
        nn_input, safe, clip_low, clip_high, low_bound, high_bound = \
            determine_settings(nn, input, safe, clip_low, clip_high, low_bound, high_bound)
        del input
        if not safe:
            print('WARNING! Applying victor_rule in unsafe mode. gammaE should be last column!')
            gammaE = nn_input[:, [-1]]
            nn_input = np.delete(nn_input, -1, 1)
            if len(nn._feature_names) != nn_input.shape[1]:
                raise Exception('Mismatch! shape feature names != shape input ({:d} != {:d})'.format(len(nn._feature_names), nn_input.shape[1]))
        # Get indices for vars that victor rule needs: x, q, smag
        vic_idx = [nn._feature_names[(nn._feature_names == var)].index[0] for var in ['x', 'q', 'smag']]
        # Get output from underlying QuaLiKizNDNN
        output = nn.get_output(nn_input, output_pandas=False, clip_low=False, clip_high=False, safe=False)
        # Get the maximum ion scale growth rate. This is taken as gamma0 for the victor rule

        #gamma0_lower_bound = 1e-4
        gamma0_idx = nn._target_names[(nn._target_names == 'gam_leq_GB')].index[0]
        gamma0 = output[:, [gamma0_idx]]
        output = np.delete(output, gamma0_idx, 1)
        #gamma0 = np.clip(gamma0, gamma0_lower_bound, None) # Growth rate can't be negative, so clip

        # Get the input for victor rule, and calculate gamma_eff
        vic_input = nn_input[:, vic_idx]
        full_vic_input = np.hstack([gamma0, nn_input[:, vic_idx], gammaE])
        f_vic = apply_victor_rule(*full_vic_input.T)

        for ii, name in enumerate(self._target_names):
            if is_flux(name) and not is_pure_flux(name):
                raise Exception('Cannot apply victor rule to non-pure flux {!s}!'.format(name))
            elif is_pure_flux(name):
                # Scale flux by max(0, gamma_eff/gamma0)
                if 'ETG' not in name:
                    output[:, [ii]] = output[:, [ii]] * f_vic

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

    def plot_victorplot(epsilon, q, s_hat, gamma0, plotvar, gammaE_GB=None, qlk_data=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #Prep QuaLiKiz data
        dim_names = ['epsilon', 'q', 's_hat']
        if qlk_data is not None:
            qlk_data = qlk_data.copy()
            qlk_data['epsilon'] = eps(qlk_data['x'])
            qlk_data['gammaE'] = gammaE_QLK_to_gammaE_GB(qlk_data['gammaE'], Te, Ai0)
            qlk_data['gam_leq_GB'] = qlk_data['gam_leq_GB'] * 8
            qlk_data.rename(columns={'smag': 's_hat'}, inplace=True)
            for name in dim_names:
                if name != plotvar:
                    if len(locals()[name]) == 1:
                        const_var = locals()[name][0]
                    qlk_data = qlk_data.loc[np.isclose(qlk_data[name], const_var, rtol=1e-3, atol=1e-5)]
                else:
                    idx = qlk_data.isnull()['gam_leq_GB']
                    for const_var in locals()[name]:
                        idx |= np.isclose(qlk_data[name], const_var, rtol=1e-3, atol=1e-5)
                    qlk_data = qlk_data.loc[idx]

            qlk_data = qlk_data.pivot(index='gammaE', columns=plotvar, values='gam_leq_GB')

        if gammaE_GB is None:
            n = 100
            gammaE_GB = np.linspace(0,1,n)
        idx = pd.MultiIndex.from_product([gammaE_GB, epsilon, q, s_hat], names=['gammaE'] + dim_names)
        data = pd.DataFrame(index=idx)
        data.reset_index(inplace=True)
        data['f_vic'] = apply_victorthesis_eps(*data.loc[:, ('epsilon', 'q', 's_hat')].values.T)
        data['gamma0'] = np.tile(gamma0, [1, n]).T
        data['line'] = data['gamma0'] + data['f_vic'] * data['gammaE']
        data['line'] = data['line'].clip(0)
        gammaE_plot = data.pivot(index='gammaE', columns=plotvar, values='line')
        if plotvar == 'epsilon':
            gammaE_plot = gammaE_plot[gammaE_plot.columns[::-1]]
            cmap = ListedColormap(['C1', 'C0', 'C2', 'C4', 'C3', 'C8'])
            if qlk_data is not None:
                qlk_data = qlk_data[qlk_data.columns[::-1]]
        else:
            cmap = ListedColormap(['C1', 'C2', 'C0', 'C4', 'C3', 'C8'])
        style = [':'] * data[plotvar].unique().size
        gammaE_plot.plot(colormap=cmap, style=style, ax=ax)
        if qlk_data is not None:
            if np.all(np.isclose(qlk_data.columns, gammaE_plot.columns)):
                qlk_data.columns = gammaE_plot.columns
            else:
                print('Warning! {0!s} qlk != {0!s} scan ({1!s} != {2!s})'.format(plotvar, qlk_data.columns, gammaE_plot.columns))
            qlk_data.plot(colormap=cmap, ax=ax, marker='o', style=' ')
        return data, ax
        plt.show()

    Ai0 = 2.0
    Te = 8.0
    n = 100
    gammaE_GB = np.linspace(0,1,n)
    gammaE_QLK = gammaE_GB_to_gamma_QLK_in(gammaE_GB, Te, Ai0)
    qlk_data = None
    store = pd.HDFStore('victorrun.h5')
    qlk_data = store['flattened']
    data, ax = plot_victorplot([0.03, 0.05, 0.1, 0.18, 0.26, 0.35], [1.4], [0.4], [0.22, 0.27, 0.4, 0.57, 0.65, 0.71], 'epsilon', qlk_data=qlk_data)
    plot_victorplot([0.18], [0.73, 1.4, 2.16, 2.88, 3.60, 4.32], [0.4], [0.27, 0.5, 0.64, 0.701, 0.74, 0.76], 'q', qlk_data=qlk_data)
    plot_victorplot([0.18], [0.73, 1.4, 2.16, 2.88, 3.60, 4.32], [0.8], [0.34, 0.54, 0.64, 0.69, 0.71, 0.73], 'q', qlk_data=qlk_data)
    plot_victorplot([0.18], [1.4], [0.2, 0.7, 1.2, 1.7, 2.2, 2.7], [.92, 1.18, 1.07, 0.85, 0.63, 0.52], 's_hat', qlk_data=qlk_data)
    plt.show()


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
    nn_ITG = QuaLiKizNDNN.from_json('../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json', layer_mode='classic')
    nn_TEM = QuaLiKizNDNN.from_json('../../tests/gen3_test_files/Network_591_efeTEM_GB/nn.json', layer_mode='classic')
    nn_gam = QuaLiKizNDNN.from_json('../../tests/gen3_test_files/Network_711_gam_leq_GB/nn.json', layer_mode='classic')
    target_names = nn_ITG._target_names.append(nn_TEM._target_names, ignore_index=True)
    multi_nn = QuaLiKizComboNN(target_names, [nn_ITG, nn_TEM], lambda *x: np.hstack(x))

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
    input = input.loc[:, nn_ITG._feature_names]
    input['gammaE_GB'] = np.full_like(input['Ati'], 0.3)

    nn = VictorNN(multi_nn, nn_gam)
    fluxes_novic = multi_nn.get_output(input.values[:, :-1], safe=False)
    fluxes_vic = nn.get_output(input.values, safe=False)
    fluxes = fluxes_vic.merge(fluxes_novic, left_index=True, right_index=True, suffixes=('_vic', '_novic'))

    print(fluxes)
    embed()
