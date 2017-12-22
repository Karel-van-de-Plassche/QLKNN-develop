from IPython import embed
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
from model import Network, NetworkJSON
from run_model import QuaLiKizNDNN

import matplotlib.pyplot as plt

def load_data(id):
    store = pd.HDFStore('../7D_nions0_flat.h5')
    input = store['megarun1/input']
    data = store['megarun1/flattened']

    root_name = '/megarun1/nndb_nn/'

    query = (Network.select(Network.target_names).where(Network.id == id).tuples()).get()
    target_names = query[0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')

    print(target_name)
    parent_name = root_name + target_name + '/'
    network_name = parent_name + str(id)
    network_name += '_noclip'
    nn = load_nn(id)


    df = data[target_name].to_frame('target')
    df['prediction'] = store[network_name].iloc[:, 0]
    df = df.astype('float64')
    df['residuals'] = df['target'] - df['prediction']
    df['maxgam'] = pd.DataFrame({'leq': data['gam_leq_GB'],
                     'less': data['gam_less_GB']}).max(axis=1)
    return input, df, nn

def load_nn(id):
    subquery = (Network.select(NetworkJSON.network_json)
                .where(Network.id == id)
                .join(NetworkJSON)
                .tuples()).get()
    json_dict = subquery[0]
    nn = QuaLiKizNDNN(json_dict)
    return nn

nameconvert = {'Ate': '$R/L_{T_e}$',
               'Ati': '$R/L_{T_i}$',
               'An': '$R/L_n$',
               #'Nustar': '$\\nu^*$',
               'Nustar': '$log_{10}(\\nu^*)$',
               'Ti_Te': '$T_i/T_e$',
               'Zeffx': '$Z_{eff}$',
               'qx': '$q$',
               'smag': '$\hat{s}$',
               'x': '$\\varepsilon\,(r/R)$',

               'efe_GB': '$q_e\,[GB]$',
               'efi_GB': '$q_i\,[GB]$',
               'efiITG_GB': '$q_{ITG, i}\,[GB]$',
               'efeETG_GB': '$q_{ETG, e}\,[GB]$',
               'pfe_GB': '$\Gamma_e\,[GB]$',
               'pfi_GB': '$\Gamma_i\,[GB]$',
}

nameconvert = {'Ate': 'Normalized electron temperature gradient $R/L_{T_e}$',
               'Ati': 'Normalized ion temperature gradient $R/L_{T_i}$',
               'An': '$R/L_n$',
               #'Nustar': '$\\nu^*$',
               'Nustar': '$log_{10}(\\nu^*)$',
               'Ti_Te': 'Relative temperature $T_i/T_e$',
               'Zeffx': '$Z_{eff}$',
               'qx': '$q$',
               'smag': 'Magnetic shear $\hat{s}$',
               'x': '$\\varepsilon\,(r/R)$',

               'efe_GB': '$q_e\,[GB]$',
               'efi_GB': '$q_i\,[GB]$',
               'efiITG_GB': '$q_{ITG, i}\,[GB]$',
               'efeITG_GB': '$q_{ITG, e}\,[GB]$',
               'efeETG_GB': 'Normalized heat flux $q$',
               'pfe_GB': '$\Gamma_e\,[GB]$',
               'pfi_GB': '$\Gamma_i\,[GB]$',
               'pfeITG_GB': '$\Gamma_{ITG, i}\,[GB]$',
               'pfeTEM_GB': '$\Gamma_{TEM, i}\,[GB]$'
}
def prettify_df(input, data):
    try:
        del input['nions']
    except KeyError:
        pass

    for ii, col in enumerate(input):
        if col == u'Nustar':
            input[col] = input[col].apply(np.log10)
            #se = input[col]
            #se.name = nameconvert[se.name]
            input['x'] = (input['x'] / 3)
    input.rename(columns=nameconvert, inplace=True)
    data.rename(columns=nameconvert, inplace=True)

    #for ii, col in enumerate(data):
    #    se = data[col]
    #    try:
    #        se.name = nameconvert[se.name]
    #    except KeyError:
    #        warn('Did not translate name for ' + se.name)
    return input, data

