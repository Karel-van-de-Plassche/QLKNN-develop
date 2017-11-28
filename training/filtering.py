from __future__ import division
import re
from itertools import product
from IPython import embed
import pandas as pd
import numpy as np

particle_vars = [u'pf', u'df', u'vt', u'vr', u'vc']
heat_vars = [u'ef']
momentum_vars = [u'vf']

#'vti_GB', 'dfi_GB', 'vci_GB',
#       'pfi_GB', 'efi_GB',
#       
#       'efe_GB', 'vce_GB', 'pfe_GB',
#       'vte_GB', 'dfe_GB'
             #'chie', 'ven', 'ver', 'vec']
def regime_filter(data, leq, less):
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    bool &= (data['efe_GB'] < less) & (data['efi_GB'] < less)
    bool &= (data['efe_GB'] >= leq) & (data['efi_GB'] >= leq)
    data = data.loc[bool]
    return data

def stability_filter(data):
    for col in data.columns:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if splitted[0] not in heat_vars + particle_vars + momentum_vars:
            print('skipping {!s}'.format(col))
            continue
        if splitted[2] == 'TEM':
            gam_filter = 'tem'
        elif splitted[2] == 'ITG':
            gam_filter = 'itg'
        elif splitted[2] == 'ETG':
            gam_filter = 'elec'
        elif splitted[0] in heat_vars and splitted[1] == 'e':
            gam_filter = 'multi'
        else:
            gam_filter = 'ion'

        pre = len(data[col])
        if gam_filter == 'ion':
            data[col] = data[col].loc[data['gam_leq_GB'] != 0]
        elif gam_filter == 'elec':
            data[col] = data[col].loc[data['gam_great_GB'] != 0]
        elif gam_filter == 'multi':
            data[col] = data[col].loc[(data['gam_leq_GB'] != 0) | (data['gam_great_GB'] != 0)]
        elif gam_filter == 'tem':
            data[col] = data[col].loc[data['TEM']]
        elif gam_filter == 'itg':
            data[col] = data[col].loc[data['ITG']]
        print('{:.2f}% of sane {!s:<9} points unstable at {!s:<5} scale'.format(np.sum(~data[col].isnull()) / pre * 100), col, gam_filter)
    return data

def filter_negative(data):
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    for col in data.columns:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if splitted[0] in heat_vars:
            bool &= (data[col] >= 0)
        elif splitted[0] in particle_vars:
            pass
    return bool

def filter_ck(data, bound):
    return (np.abs(data['cki']) < bound) & (np.abs(data['cke']) < bound)

def filter_totsep(data, septot_factor):
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
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
                bool &= np.abs(data[sepname]) <= septot_factor * np.abs(data[totname])

            print('After filter {!s:<6} {!s:<6} {:.2f}% left'.format('septot', totname, 100*np.sum(bool)/startlen))
    return bool

def filter_ambipolar(data, bound):
    return (data['absambi'] < bound) & (data['absambi'] > 1/bound)

def filter_femtoflux(data, bound):
    fluxes = [col for col in data if len(re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)) > 1 if re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)[0] in particle_vars + heat_vars + momentum_vars]
    absflux = data[fluxes].abs()
    return ~((absflux < bound) & (absflux != 0)).any(axis=1)

def sanity_filter(data, ck_bound, septot_factor, ambi_bound, femto_bound):
    # Throw away point if negative heat flux
    data = data.loc[filter_negative(data)]
    print('After filter {!s:<13} {:.2f}% left'.format('negative', 100*len(data)/startlen))


    # Throw away point if cke or cki too high
    data = data.loc[filter_ck(data, ck_bound)]
    print('After filter {!s:<13} {:.2f}% left'.format('ck', 100*len(data)/startlen))

    # Throw away point if sep flux is way higher than tot flux
    data = data.loc[filter_totsep(data, septot_factor)]
    print('After filter {!s:<13} {:.2f}% left'.format('septot', 100*len(data)/startlen))

    data = data.loc[filter_ambipolar(data, ambi_bound)]
    print('After filter {!s:<13} {:.2f}% left'.format('ambipolar', 100*len(data)/startlen))

    data = data.loc[filter_femtoflux(data, femto_bound)]
    print('After filter {!s:<13} {:.2f}% left'.format('femtoflux', 100*len(data)/startlen))

    # Alternatively:
    #data = data.loc[filter_negative(data) & filter_ck(data, ck_bound) & filter_totsep(data, septot_factor)]

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
            store.put(col, data[col].dropna(), format='table')

if __name__ == '__main__':
    dim = 7

    store_name = ''.join(['gen2_', str(dim), 'D_nions0_flat'])
    store = pd.HDFStore('../' + store_name + '.h5', 'r')

    input = store['/megarun1/input']
    data = store['megarun1/flattened']

    startlen = len(data)
    filtered_store = pd.HDFStore('filtered_' + store_name, 'w')
    data = sanity_filter(data, 50, 1.5, 1.5, 1e-4)
    data = regime_filter(data, 0, 100)
    print('After filter {!s:<13} {:.2f}% left'.format('regime', 100*len(data)/startlen))
    data = stability_filter(data)
    separate_to_store(input, data, '../filtered_' + store_name + '_filter6')
