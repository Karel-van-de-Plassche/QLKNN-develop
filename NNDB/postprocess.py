from IPython import embed
from multiprocessing import Pool, cpu_count
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
from itertools import product, chain, zip_longest
import pickle
import os
import sys
import time
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
training_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../training'))
plots_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../plots'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
sys.path.append(plots_path)
import model
from model import Network, NetworkJSON, PostprocessSlice, ComboNetwork, MultiNetwork, no_elements_in_list, Postprocess, Filter
from run_model import QuaLiKizNDNN, QuaLiKizDuoNN
from train_NDNN import shuffle_panda
from functools import partial
from matplotlib import gridspec, cycler
from load_data import load_data, load_nn, prettify_df
from collections import OrderedDict
from peewee import Param, fn
import re
from slicer import get_similar_not_in_table
from filtering import regime_filter, stability_filter

def nns_from_nndb(max=20):
    non_processed = get_similar_not_in_table(Postprocess, max, only_sep=False, no_particle=False)

    nns = OrderedDict()
    for dbnn in non_processed:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = '_'.join([str(el) for el in [dbnn.__class__.__name__ , dbnn.id]])
        nns[nn.label] = nn
    return nns

def process_nns(nns, filter_path_name, leq_bound, less_bound):
    #store = pd.HDFStore('../filtered_gen2_7D_nions0_flat_filter6.h5')
    filter_id = Filter.find_by_path_name(filter_path_name)
    try:
        filter = Filter.by_id(filter_id).get()
    except Filter.DoesNotExist:
        #raise
        pass
    store = pd.HDFStore(filter_path_name)
    nn0 = list(nns.values())[0]
    target_names = nn0._target_names
    feature_names = nn0._feature_names
    regime = regime_filter(pd.concat([store['efe_GB'], store['efi_GB']], axis='columns'), leq_bound, less_bound).index
    target = store[target_names[0]].to_frame().loc[regime]
    for name in target_names[1:]:
        target[name] = store[name].loc[regime]
    print('target loaded')
    target.columns = pd.MultiIndex.from_product([['target'], target.columns])

    input = store['input']
    input = input.loc[target.index]
    input = input[feature_names]
    input.index.name = 'dimx'
    input.reset_index(inplace=True)
    dimx = input['dimx']
    input.drop('dimx', inplace=True, axis='columns')
    target.reset_index(inplace=True, drop=True)

    print('Dataset prepared')
    results = pd.DataFrame()
    for label, nn in nns.items():
        print('Starting on {!s}'.format(label))
        out = nn.get_output(input, safe=False)
        out.columns = pd.MultiIndex.from_product([[label], out.columns])
        print('Done! Merging')
        results = pd.concat([results, out], axis='columns')
    diff = results.stack().sub(target.stack().squeeze(), axis=0).unstack()
    rms = diff.pow(2).mean().mean(level=0).pow(0.5)

    for col in rms.index:
        cls, id = col.split('_')
        dbnn = getattr(model, cls).by_id(int(id)).get()
        dict_ = {}
        if isinstance(dbnn, Network):
            dict_['network'] = dbnn
        elif isinstance(dbnn, ComboNetwork):
            dict_['combo_network'] = dbnn
        elif isinstance(dbnn, MultiNetwork):
            dict_['multi_network'] = dbnn
        dict_['leq_bound'] = leq_bound
        dict_['less_bound'] = less_bound
        dict_['rms'] = rms[col]
        dict_['filter'] = filter
        post = Postprocess(**dict_)
        post.save()

    return rms

if __name__ == '__main__':
    #filter_path_name = '../filtered_7D_nions0_flat_filter5.h5'
    filter_path_name = '../unstable_test_gen2_7D_nions0_flat_filter7.h5'
    leq_bound = 0
    less_bound = 10
    nns = nns_from_nndb(100)
    rms = process_nns(nns, filter_path_name, leq_bound, less_bound)

#results = pd.DataFrame([], index=pd.MultiIndex.from_product([['target'] + list(nns.keys()), target_names]))
