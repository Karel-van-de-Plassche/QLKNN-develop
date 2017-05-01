#from train_NDNN import filter_panda, convert_panda, Datasets
from IPython import embed
import os
import shutil
import tarfile
import pandas as pd
import numpy as np

store_name = 'everything_nions0.h5'
filtered_store_name = 'filtered_' + store_name
list_train_dims = ['efe_GB',
                   'efeETG_GB',
                   ['efe_GB', 'min', 'efeETG_GB'],
                   'efi_GB',
                   ['vte_GB', 'plus', 'vce_GB'],
                   ['vti_GB', 'plus', 'vci_GB'],
                   'dfe_GB',
                   'dfi_GB',
                   'vte_GB',
                   'vce_GB',
                   'vti_GB',
                   'vci_GB',
                   'gam_GB_less2max',
                   'gam_GB_leq2max']


def create_folders():
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
        print(name)
        dir = os.path.join(root, name)
        os.mkdir(dir)
        os.symlink(os.path.abspath('train_NDNN.py'),
                   os.path.join(dir, 'train_NDNN.py'))
        os.symlink(os.path.join('/home/kvdplassche/working/nn_data',
                                filtered_store_name),
                   os.path.join(dir, filtered_store_name))

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


def filter_all():
    store = pd.HDFStore(store_name, 'r')
    # Pre-load everything
    totflux = store['/megarun1/totflux']
    input = store['/megarun1/input']
    sepflux = store['/megarun1/sepflux']
    gam_less = store['/megarun1/gam_GB_less2max']
    gam_leq = store['/megarun1/gam_GB_leq2max']

    filtered_store = pd.HDFStore(filtered_store_name, 'a')
    # Define filter
    embed()
    max = 60
    try:
        index = filtered_store.get('index')
    except KeyError:
        index = totflux.index[(0 < totflux['efe_GB']) &
                              (totflux['efe_GB'] < max) &
                              (0 < totflux['efi_GB']) &
                              (totflux['efi_GB'] < max) &
                              (input['Zeffx'] == 1.0) &
                              (np.isclose(input['Nustar'], 1e-3, atol=1e-5))]
        # Save index
        filtered_store.put('index', index.to_series())

    try:
        input = filtered_store.get('input')
    except KeyError:
        print('processing input')
        input = input.loc[index]
        input = input.loc[:, (input != input.iloc[0]).any()] # Filter constant values
        filtered_store.put('input', input)

    print('processing gam')
    name = 'gam_GB_leq2max'
    for gam_store, name in zip([gam_leq, gam_less], ['gam_GB_leq2max', 'gam_GB_less2max']):
        try:
            filtered_store.get(name)
        except KeyError:
            filtered_store.put(name, gam_store.loc[index])
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
            if train_dims[0] in totflux and train_dims[2] in totflux:
                name = '_'.join(train_dims)
                df1 = totflux[train_dims[0]].loc[index]
                df2 = totflux[train_dims[2]].loc[index]
                if train_dims[1] == 'plus':
                    df = df1 + df2
                elif train_dims[1] == 'min':
                    df = df1 - df2
        if name is not None:
            print('putting ' + name)
            filtered_store.put(name, df.squeeze())
            print('putting ' + name + ' done')
        else:
            not_done.append(train_dims)
    del totflux

    really_not_done = []
    for train_dims in not_done:
        if train_dims[0] == 'efe_GB' and train_dims[2] == 'efeETG_GB':
            name = '_'.join(train_dims)
            df1 = efe_GB
            df2 = sepflux[train_dims[2]].loc[index]
            if train_dims[1] == 'plus':
                df = df1 + df2
            elif train_dims[1] == 'min':
                df = df1 - df2
        if name is not None:
            print('putting ' + name)
            filtered_store.put(name, df.squeeze())
            print('putting ' + name + ' done')
        else:
            really_not_done.append(train_dims)

    if len(really_not_done) != 0:
        print('Some filtering failed..')
        print(really_not_done)


#extract_nns()
filter_all()
#create_folders()
#extract_nns('9D_RAPTOR_NNs')
print('Script done')
