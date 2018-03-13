import gc
from collections import OrderedDict
import warnings
import re

import pandas as pd
import numpy as np
from IPython import embed

from qlknn.misc.analyse_names import heat_vars, particle_vars, particle_diffusion_vars, momentum_vars, is_flux, is_growth

store_format = 'fixed'

def put_to_store_or_df(store_or_df, name, var):
    if isinstance(store_or_df, pd.HDFStore):
        store_or_df.put(name, var, format=store_format)
    else:
        store_or_df[name] = var

def separate_to_store(data, store, save_flux=True, save_growth=True, save_all=False, verbose=False, **put_kwargs):
    for col in data:
        key = ''.join(['output/', col])
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

def first(s):
    '''Return the first element from an ordered collection
       or an arbitrary element from an unordered collection.
       Raise StopIteration if the collection is empty.
    '''
    return next(iter(s.items()))

def load_from_store(store_name=None, store=None, fast=True, mode='bare', how='left', columns=None, prefix='/'):
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, pd.Series):
        columns = columns.values
    if store_name is not None and store is not None:
        raise Exception('Specified both store and store name!')
    if store is None:
        store = pd.HDFStore(store_name, 'r')
    has_flattened = lambda store: any(['flattened' in group for group in store.keys()])
    return_all = lambda columns: columns is None
    return_no = lambda columns: columns is False
    is_legacy = lambda store: all(['megarun' in name for name in store.keys()])
    names = store.keys()
    # Associate 'nice' name with 'ugly' HDF5 node path, and only use data columns
    names = [(name, name.lstrip(prefix))
                   for name in names
                   if (('input' not in name) and
                       ('constants' not in name) and
                       ('flattened' not in name))]
    have_sep = lambda columns: columns is None or (len(names) == len(columns))
    if not return_all(columns):
        names = [(varname, name) for (varname, name) in names if name in columns]
    names = OrderedDict(names)
    if has_flattened(store) and (return_all(columns) or not have_sep(columns)):
        #print('Taking "old" code path')
        if is_legacy(store):
            warnings.warn('Using legacy datafile!')
            prefix = '/megarun1/'
        input = store[prefix + 'input']
        try:
            const = store[prefix + 'constants']
        except ValueError as ee:
            warnings.warn('Could not load const.. Skipping for now')
            const = pd.Series()
        if return_all(columns):
            data = store.select(prefix + 'flattened')
        elif return_no(columns):
            data = pd.DataFrame(index=input.index)
        else:
            data = store.select(prefix + 'flattened', columns=columns)
    else: #If no flattened
        #print('Taking "new" code path')
        const = store[prefix + 'constants']
        input = store[prefix + 'input']
        if not return_no(columns):
            if fast:
                output = []
                for varname, __ in names.items():
                    output.append(store[varname])
                data = pd.concat(output, axis=1)
                del output
            else:
                if (mode != 'update') and (mode != 'bare'):
                    data = store[first(names)[0]].reindex(index=input.index).to_frame()
                elif mode == 'update':
                    data = pd.DataFrame(columns=names.values(), index=input.index)
                    df = store[first(names)[0]]
                    df.name = first(names)[1]
                    data.update(df, raise_conflict=True)
                else:
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

