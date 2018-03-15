from __future__ import division
import re
from itertools import product
import gc
import os
import warnings
from collections import OrderedDict

from IPython import embed
import pandas as pd
import numpy as np

from qlknn.dataset.data_io import put_to_store_or_df, save_to_store, load_from_store, sep_prefix
from qlknn.misc.analyse_names import heat_vars, particle_vars, particle_diffusion_vars, momentum_vars, is_partial_diffusion, is_partial_particle

def drop_start_with(data, start_with):
    droplist = []
    for col in data.columns:
        if any(col.startswith(part) for part in start_with):
            droplist.append(col)
    data.drop(droplist, axis='columns', inplace=True)

def regime_filter(data, leq, less):
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    bool &= (data['efe_GB'] < less) & (data['efi_GB'] < less)
    bool &= (data['efe_GB'] >= leq) & (data['efi_GB'] >= leq)
    data = data.loc[bool]
    return data

def div_filter(store):
    for group in store:
        if isinstance(store, pd.HDFStore):
            name = group.lstrip(sep_prefix)

        if name in filter_defaults['div']:
            low, high = filter_defaults['div'][name]
        else:
            continue
        se = store[group]
        se.name = name
        pre = np.sum(~se.isnull())

        if is_partial_particle(name):
            se = se.abs()

        put_to_store_or_df(store, se.name, store[group].loc[(low < se) & (se < high)])
        print('{:5.2f}% of sane unstable {!s:<9} points inside div bounds'.format(np.sum(~store[group].isnull()) / pre * 100, group))
    return store


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

        pre = np.sum(~data[col].isnull())
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
        print('{:5.2f}% of sane {!s:<9} points unstable at {!s:<5} scale'.format(np.sum(~data[col].isnull()) / pre * 100, col, gam_filter))
    return data

def negative_filter(data):
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    for col in data.columns:
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if (splitted[0] in heat_vars) and (len(splitted) == 5):
            bool &= (data[col] >= 0)
    return bool

def ck_filter(data, bound):
    return (np.abs(data['cki']) < bound) & (np.abs(data['cke']) < bound)

def septot_filter(data, septot_factor, startlen=None):
    if startlen is None:
        startlen = len(data)
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

def ambipolar_filter(data, bound):
    return (data['absambi'] < bound) & (data['absambi'] > 1/bound)

def femtoflux_filter(data, bound):
    fluxes = [col for col in data if len(re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)) == 5 if re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)[0] in particle_vars + heat_vars + momentum_vars]
    bool = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    for flux in fluxes:
        absflux = data[flux].abs()
        bool &= ~((absflux < bound) & (absflux != 0))
    return bool

def sanity_filter(data, ck_bound, septot_factor, ambi_bound, femto_bound, 
                  stored_negative_filter=None, stored_ck_filter=None, stored_ambipolar_filter=None,
                  stored_septot_filter=None, stored_femtoflux_filter=None, startlen=None):
    if startlen is None:
        startlen = len(data)
    # Throw away point if negative heat flux
    if stored_negative_filter is None:
        data = data.loc[negative_filter(data)]
    else:
        data = data.reindex(index=data.index.difference(stored_negative_filter), copy=False)
    print('After filter {!s:<13} {:.2f}% left'.format('negative', 100*len(data)/startlen))
    gc.collect()

    # Throw away point if cke or cki too high
    if stored_ck_filter is None:
        data = data.loc[ck_filter(data, ck_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_ck_filter), copy=False)
    print('After filter {!s:<13} {:.2f}% left'.format('ck', 100*len(data)/startlen))
    gc.collect()

    # Throw away point if sep flux is way higher than tot flux
    if stored_septot_filter is None:
        data = data.loc[septot_filter(data, septot_factor, startlen=startlen)]
    else:
        data = data.reindex(index=data.index.difference(stored_septot_filter), copy=False)
    print('After filter {!s:<13} {:.2f}% left'.format('septot', 100*len(data)/startlen))
    gc.collect()

    if stored_ambipolar_filter is None:
        data = data.loc[ambipolar_filter(data, ambi_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_ambipolar_filter), copy=False)
    print('After filter {!s:<13} {:.2f}% left'.format('ambipolar', 100*len(data)/startlen))
    gc.collect()

    if stored_femtoflux_filter is None:
        data = data.loc[femtoflux_filter(data, femto_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_femtoflux_filter), copy=False)
    print('After filter {!s:<13} {:.2f}% left'.format('femtoflux', 100*len(data)/startlen))
    gc.collect()

    # Alternatively:
    #data = data.loc[filter_negative(data) & filter_ck(data, ck_bound) & filter_septot(data, septot_factor)]

    return data

filter_functions = {'negative': negative_filter,
                    'ck': ck_filter,
                    'septot': septot_filter,
                    'ambipolar': ambipolar_filter,
                    'femtoflux': femtoflux_filter}

filter_defaults = {'div':
                     {
                         'pfeTEM_GB': (0.02, 20),
                         'pfeITG_GB': (0.02, 10),
                         'efiTEM_GB': (0.05, np.inf),
                         'efeITG_GB_div_efiITG_GB': (0.05, 1.5),
                         'pfeITG_GB_div_efiITG_GB': (0.02, 0.6),
                         'efiTEM_GB_div_efeTEM_GB': (0.05, 2.0),
                         'pfeTEM_GB_div_efeTEM_GB': (0.03, 0.8),

                         'dfeITG_GB_div_efiITG_GB': (0.02, np.inf),
                         'dfiITG_GB_div_efiITG_GB': (0.15, np.inf),
                         'vceITG_GB_div_efiITG_GB': (-np.inf, np.inf),
                         'vciITG_GB_div_efiITG_GB': (0.1, np.inf),
                         'vteITG_GB_div_efiITG_GB': (0.02, np.inf),
                         'vtiITG_GB_div_efiITG_GB': (-np.inf, np.inf),

                         'dfeTEM_GB_div_efeTEM_GB': (0.10, np.inf),
                         'dfiTEM_GB_div_efeTEM_GB': (0.05, np.inf),
                         'vceTEM_GB_div_efeTEM_GB': (0.07, np.inf),
                         'vciTEM_GB_div_efeTEM_GB': (-np.inf, np.inf),
                         'vteTEM_GB_div_efeTEM_GB': (-np.inf, np.inf),
                         'vtiTEM_GB_div_efeTEM_GB': (-np.inf, np.inf)
                     },
                   'negative': None,
                   'ck': 50,
                   'septot': 1.5,
                   'ambipolar': 1.5,
                   'femtoflux': 1e-4
                   }

def create_stored_filter(store, data, filter_name, filter_var):
    filter_func = filter_functions[filter_name]
    name = ''.join(['/filter/', filter_name])
    if filter_var is not None:
        name = ''.join([name, '_', str(filter_var)])
        var = data.index[~filter_func(data, filter_var)].to_series()
    else:
        var = data.index[~filter_func(data)].to_series()
    store.put(name, var)

def load_stored_filter(store, filter_name, filter_var):
    name = ''.join(['/filter/', filter_name])
    try:
        if filter_var is not None:
            name = ''.join([name, '_', str(filter_var)])
            filter = store.get(name)
        else:
            filter = store.get(name)
    except KeyError:
        filter = None
    return filter


gen3_div_names_base = ['efeITG_GB_div_efiITG_GB',
                       'pfeITG_GB_div_efiITG_GB',
                       'efiTEM_GB_div_efeTEM_GB',
                       'pfeTEM_GB_div_efeTEM_GB']

gen3_div_names_dv = [
    'dfeITG_GB_div_efiITG_GB',
    'dfiITG_GB_div_efiITG_GB',
    'vceITG_GB_div_efiITG_GB',
    'vciITG_GB_div_efiITG_GB',
    'vteITG_GB_div_efiITG_GB',
    'vtiITG_GB_div_efiITG_GB',

    'dfeTEM_GB_div_efeTEM_GB',
    'dfiTEM_GB_div_efeTEM_GB',
    'vceTEM_GB_div_efeTEM_GB',
    'vciTEM_GB_div_efeTEM_GB',
    'vteTEM_GB_div_efeTEM_GB',
    'vtiTEM_GB_div_efeTEM_GB'
]
gen3_div_names = gen3_div_names_base + gen3_div_names_dv

def create_gen3_divsum(store, divnames=gen3_div_names):
    for name in divnames:
        one, two = re.compile('_div_').split(name)
        one, two = store[sep_prefix + one],  store[sep_prefix + two]
        res = (one / two).dropna()
        put_to_store_or_df(store, name, res)

def create_divsum(store):
    for group in store:
        if isinstance(store, pd.HDFStore):
            group = group[1:]
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(group)
        if splitted[0] in heat_vars and splitted[1] == 'i' and len(splitted) == 5:
            group2 = splitted[0] + 'e' + ''.join(splitted[2:])
            sets = [('_'.join([group, 'plus', group2]),
                     store[group] + store[group2]),
                    ('_'.join([group, 'div', group2]),
                     store[group] / store[group2]),
                    ('_'.join([group2, 'div', group]),
                     store[group2] / store[group])
            ]
        elif splitted[0] == 'pf' and splitted[1] == 'e' and len(splitted) == 5:
            group2 = 'efi' + ''.join(splitted[2:])
            group3 = 'efe' + ''.join(splitted[2:])
            sets = [
                ('_'.join([group, 'plus', group2, 'plus', group3]),
                 store[group] + store[group2] + store[group3]),
                ('_'.join([group, 'div', group2]),
                 store[group] / store[group2]),
                ('_'.join([group, 'div', group3]),
                 store[group] / store[group3])
            ]

        else:
            continue

        for name, set in sets:
            set.name = name
            put_to_store_or_df(store, set.name, set)
    return store

def filter_9D_to_7D(input, Zeff=1, Nustar=1e-3):
    if len(input.columns) != 9:
        print("Warning! This function assumes 9D input with ['Ati', 'Ate', 'An', 'q', 'smag', 'x', 'Ti_Te', 'Zeff', 'Nustar']")

    idx = input.index[(
        np.isclose(input['Zeff'], Zeff,     atol=1e-5, rtol=1e-3) &
        np.isclose(input['Nustar'], Nustar, atol=1e-5, rtol=1e-3)
    )]
    return idx

def filter_7D_to_4D(input, Ate=6.5, An=2, x=0.45):
    if len(input.columns) != 7:
        print("Warning! This function assumes 9D input with ['Ati', 'Ate', 'An', 'q', 'smag', 'x', 'Ti_Te']")

    idx = input.index[(
        np.isclose(input['Ate'], Ate,     atol=1e-5, rtol=1e-3) &
        np.isclose(input['An'], An, atol=1e-5, rtol=1e-3) &
        np.isclose(input['x'], x, atol=1e-5, rtol=1e-3)
    )]
    return idx

def split_input(input, const):
    idx = {}
    consts = {9: const.copy(),
              7: const.copy(),
              4: const.copy()}
    idx[7] = filter_9D_to_7D(input)

    inputs = {9: input}
    idx[9] = input.index
    inputs[7] = input.loc[idx[7]]
    for name in ['Zeff', 'Nustar']:
        consts[7][name] = float(inputs[7].head(1)[name].values)
    inputs[7].drop(['Zeff', 'Nustar'], axis='columns', inplace=True)

    idx[4] = filter_7D_to_4D(inputs[7])
    inputs[4] = inputs[7].loc[idx[4]]
    for name in ['Ate', 'An', 'x']:
        consts[4][name] = float(inputs[4].head(1)[name].values)
    inputs[4].drop(['Ate', 'An', 'x'], axis='columns', inplace=True)

    return idx, inputs, consts

def split_dims(input, data, const, gen, prefix=''):
    idx, inputs, consts = split_input(input, const)
    for dim in [7, 4]:
        print('splitting', dim)
        store_name = prefix + 'gen' + str(gen) + '_' + str(dim) + 'D_nions0_flat' + '_filter' + str(filter_num) + '.h5'
        save_to_store(inputs[dim], data.loc[idx[dim]], consts[dim], store_name)

def split_subsets(input, data, const, gen, frac=0.1):
    idx, inputs, consts = split_input(input, const)

    rand_index = pd.Int64Index(np.random.permutation(input.index))
    sep_index = int(frac * len(rand_index))
    idx['test'] = rand_index[:sep_index]
    idx['training'] = rand_index[sep_index:]

    print('Splitting subsets')
    for dim, set in product([4, 7, 9], ['test', 'training']):
        print(dim, set)
        store_name = set + '_' + 'gen' + str(gen) + '_' + str(dim) + 'D_nions0_flat' + '_filter' + str(filter_num) + '.h5'
        save_to_store(inputs[dim].reindex(index=(idx[dim] & idx[set]), copy=False),
                      data.reindex(index=(idx[dim] & idx[set]), copy=False),
                      consts[dim],
                      store_name)

if __name__ == '__main__':
    dim = 9
    gen = 3
    filter_num = 8

    root_dir = '.'
    basename = ''.join(['gen', str(gen), '_', str(dim), 'D_nions0_flat'])
    store_name = basename + '.h5'

    input, data, const = load_from_store(store_name)
    # Summarize the diffusion stats in a single septot filter
    store_filters = False
    if store_filters:
        with pd.HDFStore(store_name) as store:
            for filter_name in filter_functions.keys():
                create_stored_filter(store, data, filter_name, filter_defaults[filter_name])
    create_gen3_divsum(data)
    split_dims(input, data, const, gen)

    startlen = len(data)

    # As the 9D dataset is too big for memory, we have saved the septot filter seperately
    filters = {}
    with pd.HDFStore(store_name) as store:
        for filter_name in filter_functions.keys():
            name = ''.join(['stored_', filter_name, '_filter'])
            filters[name] = load_stored_filter(store, filter_name, filter_defaults[filter_name])


    data = sanity_filter(data,
                         filter_defaults['ck'],
                         filter_defaults['septot'],
                         filter_defaults['ambipolar'],
                         filter_defaults['femtoflux'],
                         startlen=startlen, **filters)
    data = regime_filter(data, 0, 100)
    gc.collect()
    input = input.loc[data.index]
    print('After filter {!s:<13} {:.2f}% left'.format('regime', 100*len(data)/startlen))
    sane_store_name = os.path.join(root_dir, 'sane_' + basename + '_filter' + str(filter_num) + '.h5')
    save_to_store(input, data, const, sane_store_name)
    split_dims(input, data, const, gen, prefix='sane_')
    #input, data, const = load_from_store(sane_store_name)
    split_subsets(input, data, const, gen, frac=0.1)
    del data, input, const
    gc.collect()


    for dim, set in product([4, 7, 9], ['test', 'training']):
        print(dim, set)
        basename = set + '_' + 'gen' + str(gen) + '_' + str(dim) + 'D_nions0_flat_filter' + str(filter_num) + '.h5'
        input, data, const = load_from_store(basename)

        data = stability_filter(data)
        #data = create_divsum(data)
        data = div_filter(data)
        save_to_store(input, data, const, 'unstable_' + basename)
    #separate_to_store(input, data, '../filtered_' + store_name + '_filter6')
