#from train_NDNN import filter_panda, convert_panda, Datasets
from IPython import embed
import os
import shutil
import tarfile
import pandas as pd
import numpy as np
import json

list_train_dims = [['both', 'efe_GB'],
                   ['ion', 'efi_GB'],
                   ['ion', 'efiITG_GB'],
                   ['ion', 'efiTEM_GB'],
                   ['elec', 'efeETG_GB'],
                   ['ion', 'efeITG_GB'],
                   ['ion', 'efeTEM_GB'],
                   ['ion', ['efe_GB', 'min', 'efeETG_GB']],
                   ['ion', ['efiITG_GB', 'div', 'efeITG_GB']],
                   ['ion', ['efiITG_GB', 'plus', 'efeITG_GB']],
                   ['ion', ['efiTEM_GB', 'div', 'efeTEM_GB']],
                   ['ion', ['efiTEM_GB', 'plus', 'efeTEM_GB']],
                   ['both', ['efi_GB', 'div', 'efe_GB']],
                   ['both', ['efi_GB', 'plus', 'efe_GB']],
                   ['ion', ['efi_GB', 'div', '9', 'efe_GB', 'min', 'efeETG_GB', '0']],
                   ['ion', ['efi_GB', 'plus', '9', 'efe_GB', 'min', 'efeETG_GB', '0']],
                   ['none', 'gam_less_GB'],
                   ['none', 'gam_leq_GB']]

def create_folders(store_name):
    try:
        shutil.rmtree('nns')
    except FileNotFoundError:
        pass
    root = os.path.join(os.curdir, 'nns')
    os.mkdir(root)
    for gam_filter, train_dims in list_train_dims:
        if train_dims.__class__ == str:
            name = train_dims
        else:
            name = '_'.join(train_dims)
        if 'gam' in name:
            continue
        print(name)
        dir = os.path.join(root, name)
        os.mkdir(dir)
        os.symlink(os.path.abspath('training/train_NDNN.py'),
                   os.path.join(dir, 'train_NDNN.py'))
        os.symlink(os.path.abspath('networks/run_model.py'),
                   os.path.join(dir, 'run_model.py'))
        os.symlink(os.path.join(os.path.abspath(os.curdir),
                                store_name),
                   os.path.join(dir, store_name))
        with open('training/default_settings.json') as file_:
            settings = json.load(file_)
            settings['train_dim'] = name
        with open(os.path.join(dir, 'settings.json'), 'w') as file_:
            json.dump(settings, file_)

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
    #totflux = store['/megarun1/totflux']
    input = store['/megarun1/input']
    data = store['megarun1/flattened']
    #sepflux = store['/megarun1/sepflux']
    #gam_less = store['/megarun1/gam_GB_less2max']
    #gam_leq = store['/megarun1/gam_GB_leq2max']

    filtered_store = pd.HDFStore('filtered_' + store_name, 'w')
    # Define filter
    try:
        index = filtered_store.get('index')
    except KeyError:
        #index = input.index[(
        #                     np.isclose(input['Zeffx'], 1,     atol=1e-5, rtol=1e-3) &
        #                     np.isclose(input['Nustar'], 1e-3, atol=1e-5, rtol=1e-3)
        #                     )]
        index = input.index
        min = 0
        max = 60
        data = data.loc[index]
        for flux in ['efeETG_GB',
                     'efeITG_GB',
                     'efeTEM_GB',
                     'efiITG_GB',
                     'efiTEM_GB',
                     'efe_GB',
                     'efi_GB']:
            data = data.loc[(data[flux] >= 0)]
        data = data.loc[(data['efe_GB'] != 0) | (data['efi_GB'] != 0)]
        data = data.loc[(data['efe_GB'] < max) & (data['efi_GB'] < max)]


        index = data.index

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

    not_done = []
    for gam_filter, train_dims in list_train_dims:
        name = None
        print('starting on')
        print(train_dims)
        if train_dims.__class__ == str:
            name = train_dims
            df = data[train_dims]
        else:
            if len(train_dims) == 3:
                if train_dims[0] in data and train_dims[2] in data:
                    name = '_'.join(train_dims)
                    df1 = data[train_dims[0]]
                    df2 = data[train_dims[2]]
                    if train_dims[1] == 'plus':
                        df = df1 + df2
                    elif train_dims[1] == 'min':
                        df = df1 - df2
                    elif train_dims[1] == 'div':
                        df = df1 / df2
                    elif train_dims[1] == 'times':
                        df = df1 * df2
        if name is not None:
            if gam_filter == 'ion':
                df = df.loc[data['gam_less_GB'] != 0]
            elif gam_filter == 'elec':
                df = df.loc[data['gam_leq_GB'] != 0]
            elif gam_filter == 'both':
                df = df.loc[(data['gam_less_GB'] != 0) | (data['gam_leq_GB'] != 0)]
            else:
                pass
            df = df.loc[(df != np.inf) & (df != -np.inf) & (df != np.nan)]
            print('putting ' + name)
            df.name = name
            filtered_store.put(name, df.squeeze())
            print('putting ' + name + ' done')
        else:
            not_done.append(train_dims)


    if len(not_done) != 0:
        print('Some filtering failed..')
        print(not_done)

    # some specials..
    df1 = data['efi_GB']
    df2 = data['efe_GB']
    df3 = data['efeETG_GB']
    name = 'efi_GB_div_9_efe_GB_min_efeETG_GB_0'
    df = (df1 / (df2 - df3))
    df.name = name
    df = df.loc[data['gam_less_GB'] != 0]
    df = df.loc[(df != np.inf) & (df != -np.inf) & (df != np.nan)]
    filtered_store[name]  = df

    name = 'efi_GB_plus_9_efe_GB_min_efeETG_GB_0'
    df = (df1 + (df2 - df3))
    df.name = name
    df = df.loc[data['gam_less_GB'] != 0]
    df = df.loc[(df != np.inf) & (df != -np.inf) & (df != np.nan)]
    filtered_store[name]  = df

    store.close()
    filtered_store.close()

def filter_individual(store_name):
    store = pd.HDFStore(store_name, 'r')
    newstore = pd.HDFStore('filtered_' + store_name, 'w')
    gam_leq = store['gam_leq_GB']
    gam_less = store['gam_less_GB']
    index = pd.Int64Index(store['index'])
    min = 0
    max = 60
    for name in store.keys():
        print(name)
        var = store[name]
        if 'gam' not in name and 'input' not in name and 'index' not in name:
            var = var.loc[var != 0]
        if 'efi' in name and 'efe' not in name:
            print('efi_style')
            var = var.loc[(var > min) & (var < max)]
            var = var.loc[gam_less != 0]
        elif 'efe' in name and 'efi' not in name:
            print('efe_style')
            var = var.loc[(var > min) & (var < max)]
            var = var.loc[gam_leq != 0]
        elif 'efe' in name and 'efi' in name:
            print('mixed_style')
            var = var.loc[gam_less != 0]
            var = var.loc[(var != np.inf) & (var != -np.inf) & (var != np.nan)]
        elif 'index' in name:
            print('index_style')
            pass
        elif 'input' in name:
            print('input_style')
            pass
        else:
            print('weird_style')
            pass
        if 'input' not in name:
            var = var.loc[~var.isnull()]
        newstore[name] = var
    store.close()
    newstore.close()
#extract_nns()
filter_all('everything_nions0_flat.h5.1')
#filter_individual('filtered_7D_nions0_flat.h5')
#create_folders('filtered_everything_nions0.h5')
#extract_nns('7D_filtered_NNs')
print('Script done')
