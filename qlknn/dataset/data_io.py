import gc
from collections import OrderedDict
import warnings
import re

import pandas as pd
import numpy as np
from IPython import embed

try:
    profile
except NameError:
    from qlknn.misc.tools import profile

from qlknn.misc.analyse_names import heat_vars, particle_vars, particle_diffusion_vars, momentum_vars, is_flux, is_growth
from qlknn.misc.tools import first

store_format = 'fixed'
sep_prefix = '/output/'

@profile
def convert_nustar(input_df):
    # Nustar relates to the targets with a log
    try:
        input_df['logNustar'] = np.log10(input_df['Nustar'])
        del input_df['Nustar']
    except KeyError:
        print('No Nustar in dataset')
    return input_df

def put_to_store_or_df(store_or_df, name, var, store_prefix=sep_prefix):
    if isinstance(store_or_df, pd.HDFStore):
        store_or_df.put(''.join([store_prefix, name]),
                        var, format=store_format)
    else:
        store_or_df[name] = var

def separate_to_store(data, store, save_flux=True, save_growth=True, save_all=False, verbose=False, **put_kwargs):
    for col in data:
        key = ''.join([sep_prefix, col])
        splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)').split(col)
        if ((is_flux(col) and save_flux) or
            (is_growth(col) and save_growth) or
            save_all):
            if verbose:
                print('Saving', col)
            store.put(key, data[col].dropna(), format=store_format, **put_kwargs)
        else:
            if verbose:
                print('Do not save', col)

def save_to_store(input, data, const, store_name, style='both', zip=False, prefix='/'):
    if zip is True:
        kwargs = {'complevel': 1,
                  'complib': 'zlib'}
        store_name += '.1'
    else:
        kwargs = {}
    store = pd.HDFStore(store_name)
    if style == 'sep' or style == 'both':
        separate_to_store(data, store, save_all=True, **kwargs)
    if style == 'flat' or style == 'both':
        if len(data) > 0:
            store.put('flattened', data, format=store_format, **kwargs)
        else:
            store.put('flattened', data, format='fixed', **kwargs)

    store.put(prefix + 'input', input, format=store_format, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        store.put(prefix + 'constants', const)
    store.close()

@profile
def load_from_store(store_name=None, store=None, fast=True, mode='bare', how='left', columns=None, prefix='', load_input=True, nustar_to_lognustar=True):
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, pd.Series):
        columns = columns.values
    if store_name is not None and store is not None:
        raise Exception('Specified both store and store name!')

    if store is None:
        store = pd.HDFStore(store_name, 'r')

    is_legacy = lambda store: all(['megarun' in name for name in store.keys()])
    if is_legacy(store):
        warnings.warn('Using legacy datafile!')
        prefix = '/megarun1/'

    has_flattened = lambda store: any(['flattened' in group for group in store.keys()])
    have_sep = lambda columns: columns is None or (len(names) == len(columns))
    return_all = lambda columns: columns is None
    return_no = lambda columns: columns is False

    names = store.keys()
    # Associate 'nice' name with 'ugly' HDF5 node path
    names = [(name, name.replace(prefix + sep_prefix, '', 1))
                   for name in names
                   if (('input' not in name) and
                       ('constants' not in name) and
                       ('flattened' not in name))]
    # Only return columns the user asked for
    if not return_all(columns):
        names = [(varname, name) for (varname, name) in names if name in columns]
    names = OrderedDict(names)

    # Load input and constants
    if load_input:
        input = store[prefix + 'input']
        if nustar_to_lognustar:
            input = convert_nustar(input)
    else:
        input = pd.DataFrame()
    try:
        const = store[prefix + 'constants']
    except ValueError as ee:
        # If pickled with a too new version, old python version cannot read it
        warnings.warn('Could not load const.. Skipping for now')
        const = pd.Series()

    if has_flattened(store) and (return_all(columns) or not have_sep(columns)):
        #print('Taking "old" code path')
        if return_all(columns):
            data = store.select(prefix + 'flattened')
        elif return_no(columns):
            data = pd.DataFrame(index=input.index)
        else:
            data = store.select(prefix + 'flattened', columns=columns)
    else: #If no flattened
        #print('Taking "new" code path')
        if not have_sep(columns):
            raise Exception('Could not find {!s} in store {!s}'.format(columns, store))
        if not return_no(columns):
            if fast:
                output = []
                for varname, name in names.items():
                    var = store[varname]
                    var.name = name
                    output.append(var)
                data = pd.concat(output, axis=1)
                del output
            else:
                if (mode != 'update') and (mode != 'bare'):
                    data = store[first(names)[0]].to_frame()
                elif mode == 'update':
                    df = store[first(names)[0]]
                    data = pd.DataFrame(columns=names.values(), index=df.index)
                    df.name = first(names)[1]
                    data.update(df, raise_conflict=True)
                elif mode == 'bare':
                    if not load_input:
                        raise Exception('Need to load input for mode {!s}'.format(mode))
                    raw_data = np.empty([len(input), len(names)])
                    ii = 0
                    varname = first(names)[0]
                    df = store[varname]
                    if df.index.equals(input.index):
                        raw_data[:, ii] = df.values
                    else:
                        raise Exception('Nonmatching index on {!s}!'.format(varname))
                for ii, (varname, name) in enumerate(names.items()):
                    if ii == 0:
                        continue
                    if ('input' not in varname) and ('constants' not in varname):
                        if mode == 'join':
                            data = data.join(store[varname], how=how)
                        elif mode == 'concat':
                            data = pd.concat([data, store[varname]], axis=1, join='outer', copy=False)
                        elif mode == 'merge':
                            data = data.merge(store[varname].to_frame(), left_index=True, right_index=True,
                                              how=how, copy=False)
                        elif mode == 'assign':
                            data = data.assign(**{name: store[varname]})
                        elif mode == 'update':
                            df = store[varname]
                            df.name = name
                            data.update(df, raise_conflict=True)
                        elif mode == 'bare':
                            df = store[varname].reindex(index=input.index)
                            if df.index.equals(input.index):
                                raw_data[:, ii] = df.values
                            else:
                                raise Exception('Nonmatching index on {!s}!'.format(varname))
                            del df
                    gc.collect()
                if mode == 'bare':
                    data = pd.DataFrame(raw_data, columns=names.values(), index=input.index)
        else: #Don't return any data
            data = pd.DataFrame(index=input.index)
    store.close()
    gc.collect()
    return input, data, const

