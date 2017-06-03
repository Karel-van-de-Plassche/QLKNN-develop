#from train_NDNN import filter_panda, convert_panda, Datasets
from IPython import embed
import os
import shutil
import tarfile
import pandas as pd
import numpy as np

list_train_dims = ['efe_GB',
                   'efi_GB',
                   'efiITG_GB',
                   'efiTEM_GB',
                   'efeETG_GB',
                   'efeITG_GB',
                   'efeTEM_GB',
                   ['efiITG_GB', 'div', 'efeITG_GB'],
                   ['efiITG_GB', 'plus', 'efeITG_GB'],
                   ['efiTEM_GB', 'div', 'efeTEM_GB'],
                   ['efiTEM_GB', 'plus', 'efeTEM_GB'],
                   'gam_GB_less2max',
                   'gam_GB_leq2max']

def create_folders(store_name):
    try:
        shutil.rmtree('nns')
    except FileNotFoundError:
        pass
    root = os.path.join(os.curdir, 'nns')
    os.mkdir(root)
    for train_dims in list_train_dims:
        if train_dims.__class__ == str:
            name = train_dims
        else:
            name = '_'.join(train_dims)
        if 'gam' in name:
            continue
        print(name)
        dir = os.path.join(root, name)
        os.mkdir(dir)
        os.symlink(os.path.abspath('train_NDNN.py'),
                   os.path.join(dir, 'train_NDNN.py'))
        os.symlink(os.path.join(os.path.abspath(os.curdir),
                                store_name),
                   os.path.join(dir, store_name))

def extract_nns(path):
    tar = tarfile.open("nns.tar.gz", "w:gz")
    root = os.path.join(os.curdir, path)
    for train_dims in list_train_dims:
        if train_dims.__class__ == str:
            name = train_dims
        else:
            name = '_'.join(train_dims)
        print(name)
        dir = os.path.join(root, name)
        try:
            filename = 'nn_' + name + '.json'
            tar.add(os.path.join(dir, 'nn.json'), 'nns/' + filename)
        except OSError:
            print('NN not done')


def filter_all(store_name):
    store = pd.HDFStore(store_name, 'r')
    # Pre-load everything
    totflux = store['/megarun1/totflux']
    input = store['/megarun1/input']
    sepflux = store['/megarun1/sepflux']
    gam_less = store['/megarun1/gam_GB_less2max']
    gam_leq = store['/megarun1/gam_GB_leq2max']

    filtered_store = pd.HDFStore('filtered_' + store_name, 'w')
    # Define filter
    max = 60
    min = 0.1
    try:
        index = filtered_store.get('index')
    except KeyError:
        index = input.index[(
                             np.isclose(input['Zeffx'], 1,     atol=1e-5, rtol=1e-3) &
                             np.isclose(input['Nustar'], 1e-3, atol=1e-5, rtol=1e-3)
                             )]
        sepflux = sepflux.loc[index]
        for flux in ['efeETG_GB',
                     'efeITG_GB',
                     'efeTEM_GB',
                     'efiITG_GB',
                     'efiTEM_GB']:

            sepflux = sepflux.loc[(sepflux[flux] >= min) & (sepflux[flux] < max)]
        index = sepflux.index
        totflux = totflux.loc[index]
        for flux in ['efe_GB',
                     'efi_GB']:
            totflux = totflux.loc[(totflux[flux] >= min) & (totflux[flux] < max)]
        index = totflux.index

        # Save index
        filtered_store.put('index', index.to_series())

    try:
        input = filtered_store.get('input')
    except KeyError:
        print('processing input')
        input = input.loc[index]
        input = input.loc[:, (input != input.iloc[0]).any()] # Filter constant values
        if input['Ate'].equals(input['Ati']):
            del input['Ati']
            input = input.rename(columns={'Ate': 'At'})
        filtered_store.put('input', input)

    print('processing gam')
    for gam_store, name in zip([gam_leq, gam_less], ['gam_GB_leq2max', 'gam_GB_less2max']):
        if name in list_train_dims:
            try:
                filtered_store.get(name)
            except KeyError:
                filtered_store.put(name, gam_store.loc[index].squeeze())
            finally:
                list_train_dims.remove(name)

    not_done = []
    for train_dims in list_train_dims:
        name = None
        set = None
        print('starting on')
        print(train_dims)
        if train_dims.__class__ == str:
            if train_dims in totflux:
                set = totflux
            elif train_dims in sepflux:
                set = sepflux
            if set is not None:
                name = train_dims
                df = set[train_dims].loc[index]
                if name == 'efe_GB':
                    efe_GB = df
        else:
            if train_dims[0] in totflux:
                set0 = totflux
            elif train_dims[0] in sepflux:
                set0 = sepflux
            if train_dims[2] in totflux:
                set2 = totflux
            elif train_dims[2] in sepflux:
                set2 = sepflux
            name = '_'.join(train_dims)
            df1 = set0[train_dims[0]].loc[index]
            df2 = set2[train_dims[2]].loc[index]
            if train_dims[1] == 'plus':
                df = df1 + df2
            elif train_dims[1] == 'min':
                df = df1 - df2
            elif train_dims[1] == 'div':
                df = df1 / df2
            elif train_dims[1] == 'times':
                df = df1 * df2
            df.name = name
        if name is not None:
            print('putting ' + name)
            filtered_store.put(name, df.squeeze())
            print('putting ' + name + ' done')
        else:
            not_done.append(train_dims)

    #really_not_done = []
    #for train_dims in not_done:
    #    if train_dims[0] == 'efe_GB' and train_dims[2] == 'efeETG_GB':
    #        name = '_'.join(train_dims)
    #        df1 = efe_GB
    #        df2 = sepflux[train_dims[2]].loc[index]
    #        if train_dims[1] == 'plus':
    #            df = df1 + df2
    #        elif train_dims[1] == 'min':
    #            df = df1 - df2
    #    if name is not None:
    #        print('putting ' + name)
    #        filtered_store.put(name, df.squeeze())
    #        print('putting ' + name + ' done')
    #    else:
    #        really_not_done.append(train_dims)

    if len(not_done) != 0:
        print('Some filtering failed..')
        print(not_done)
    store.close()
    filtered_store.close()

def filter_individual(store_name):
    store = pd.HDFStore(store_name, 'r')
    newstore = pd.HDFStore('filtered_' + store_name, 'w')
    gam_less = store['gam_GB_less2max']
    gam_leq = store['gam_GB_leq2max']
    index = pd.Int64Index(store['index'])
    for name in store.keys():
        print(name)
        var = store[name]
        if 'gam' not in name and 'input' not in name and 'index' not in name:
            var = var.loc[var != 0]
        if 'efi' in name and 'efe' not in name:
            print('efi_style')
            var = var.loc[gam_less != 0]
        elif 'efe' in name and 'efi' not in name:
            print('efe_style')
            var = var.loc[gam_leq != 0]
        elif 'efe' in name and 'efi' in name:
            print('mixed_style')
            var = var.loc[gam_less != 0]
            var = var.loc[(var != np.inf) & (var != -np.inf) & (var != np.nan)]
        elif 'index' in name:
            pass
        else:
            print('weird_style')
            pass
        newstore[name] = var
    store.close()
    newstore.close()
#extract_nns()
#filter_all('everything_nions0.h5')
#filter_individual('filtered_everything_nions0.h5')
#create_folders('filtered_everything_nions0.h5')
extract_nns('7D_filtered_NNs')
print('Script done')
