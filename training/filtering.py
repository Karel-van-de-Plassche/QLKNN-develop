import re
from itertools import product
from IPython import embed
import pandas as pd
import numpy as np

particle_vars = ['pf', 'df', 'vt', 'vr', 'vc']
heat_vars = ['ef']
momentum_vars = ['vf']

#'vti_GB', 'dfi_GB', 'vci_GB',
#       'pfi_GB', 'efi_GB',
#       
#       'efe_GB', 'vce_GB', 'pfe_GB',
#       'vte_GB', 'dfe_GB'
             #'chie', 'ven', 'ver', 'vec']
def regime_filter(data, gtr, less):
    bool = pd.Series(np.full(len(data), True), index=data.index)
    bool &= (data['efe_GB'] < less) & (data['efi_GB'] < less)
    bool &= (data['efe_GB'] > gtr) & (data['efi_GB'] > gtr)
    data = data.loc[bool]
    return data

def stability_filter(data):
    for col in data.columns:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if splitted[0] not in heat_vars + particle_vars + momentum_vars:
            print('skipping {!s}'.format(col))
            continue
        if splitted[2] in ['TEM', 'ITG'] or (splitted[1] == 'i'):
            gam_filter = 'ion'
        elif splitted[2] in ['ETG']:
            gam_filter = 'elec'
        else:
            gam_filter = 'both'

        pre = len(data[col])
        if gam_filter == 'ion':
            data[col] = data[col].loc[data['gam_less_GB'] != 0]
        elif gam_filter == 'elec':
            data[col] = data[col].loc[data['gam_leq_GB'] != 0]
        elif gam_filter == 'both':
            data[col] = data[col].loc[(data['gam_less_GB'] != 0) | (data['gam_leq_GB'] != 0)]
        print('{!s} {:.2f}% unstable on {!s} scale'.format(col, np.sum(~data[col].isnull()) / pre * 100, gam_filter))
    return data

def filter_negative(data):
    bool = pd.Series(np.full(len(data), True), index=data.index)
    for col in data.columns:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if splitted[0] in heat_vars:
            bool &= (data[col] >= 0)
        elif splitted[0] in particle_vars:
            pass
    return bool

def filter_ck(data, bound):
    return (np.abs(data['cki']) < bound) & (np.abs(data['cke']) < bound)

def filter_totsep(data, divsum_factor):
    bool = pd.Series(np.full(len(data), True), index=data.index)
    for type, spec in product(particle_vars + heat_vars, ['i', 'e']):
        totname = type + spec + '_GB'
        if totname != 'vre_GB' and totname != 'vri_GB':
            if type in particle_vars or spec == 'i': # no ETG
                seps = ['ITG', 'TEM']
            else: # All modes
                seps = ['ETG', 'ITG', 'TEM']
            for sep in seps:
                sepname = type + spec + sep + '_GB'
                #sepflux += data[sepname]
                bool &= np.abs(data[sepname]) <= divsum_factor * np.abs(data[totname])

            print('After filter divsum {!s} {:.2f}% left'.format(totname, 100*np.sum(bool)/startlen))
    return bool

def filter_ambipolar(data, bound):
    return (data['absambi'] < bound) & (data['absambi'] > 1/bound)

def sanity_filter(data, ck_bound, divsum_factor, ambi_bound):
    # Throw away point if negative heat flux
    data = data.loc[filter_negative(data)]
    print('After filter negative {:.2f}% left'.format(100*len(data)/startlen))


    # Throw away point if cke or cki too high
    data = data.loc[filter_ck(data, ck_bound)]
    print('After filter ck {:.2f}% left'.format(100*len(data)/startlen))

    # Throw away point if sep flux is way higher than tot flux
    data = data.loc[filter_totsep(data, divsum_factor)]
    print('After filter divsum {:.2f}% left'.format(100*len(data)/startlen))

    data = data.loc[filter_ambipolar(data, ambi_bound)]
    print('After filter ambipolar {:.2f}% left'.format(100*len(data)/startlen))

    # Alternatively:
    #data = data.loc[filter_negative(data) & filter_ck(data, ck_bound) & filter_totsep(data, divsum_factor)]

    return data
    #for col in data.columns:
    #    splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
    #    if splitted[0] in particle_vars + heat_vars:
    #        if splitted[2] != '':
    #            data.loc[]

def separate_to_store(input, data, name):
    store = pd.HDFStore(name + '.h5')
    store['input'] = input.loc[data.index]
    for col in data:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if splitted[0] in heat_vars + particle_vars + momentum_vars + ['gam_leq_GB', 'gam_less_GB']:
            store[col] = data[col].dropna()




if __name__ == '__main__':
    dim = 7

    store_name = ''.join([str(dim), 'D_nions0_flat.h5'])
    store = pd.HDFStore('../' + store_name, 'r')

    input = store['/megarun1/input']
    data = store['megarun1/flattened']
    data = data.join(store['megarun1/synthetic'])
    embed()

    startlen = len(data)
    filtered_store = pd.HDFStore('filtered_' + store_name, 'w')
    data = sanity_filter(data, 50, 1.5, 1.5)
    data = stability_filter(data)
    data = regime_filter(data, 0, 60)
