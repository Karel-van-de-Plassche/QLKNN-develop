import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.NNDB.model import Network, NetworkJSON, PostprocessSlice, Postprocess, Filter
from qlknn.models.ffnn import QuaLiKizNDNN
from qlknn.plots.slicer import get_similar_not_in_table
from qlknn.dataset.filtering import regime_filter, stability_filter
from qlknn.plots.load_data import load_data, load_nn, prettify_df

def nns_from_nndb(max=20):
    non_processed = get_similar_not_in_table(Postprocess, max,
                                             only_sep=False,
                                             no_particle=False,
                                             no_mixed=False,
                                             no_gam=False)

    nns = OrderedDict()
    for dbnn in non_processed:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = '_'.join([str(el) for el in [dbnn.__class__.__name__ , dbnn.id]])
        nns[nn.label] = nn
    return nns

def process_nns(nns, root_path, set, filter, leq_bound, less_bound):
    #store = pd.HDFStore('../filtered_gen2_7D_nions0_flat_filter6.h5')
    nn0 = list(nns.values())[0]
    target_names = nn0._target_names
    feature_names = nn0._feature_names

    dim = len(feature_names)
    filter_name = set + '_' + str(dim) + 'D_nions0_flat_filter' + str(filter) + '.h5.1'
    filter_path_name = os.path.join(root_path, filter_name)

    store = pd.HDFStore(filter_path_name)
    regime = regime_filter(pd.concat([store['efe_GB'], store['efi_GB']], axis='columns'), leq_bound, less_bound).index
    target = store[target_names[0]].to_frame().loc[regime]
    for name in target_names[1:]:
        target[name] = store[name].loc[regime]
    print('target loaded')
    target.columns = pd.MultiIndex.from_product([['target'], target.columns])

    input = store['input']
    try:
        input['logNustar'] = np.log10(input['Nustar'])
        del input['Nustar']
    except KeyError:
        print('No Nustar in dataset')
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
    rms = diff.pow(2).mean().pow(0.5)

    for col in rms.index.levels[0]:
        cls, id = col.split('_')
        dbnn = Network.get_by_id(int(id))
        dict_ = {'network': dbnn}
        dict_['leq_bound'] = leq_bound
        dict_['less_bound'] = less_bound
        dict_['rms'] = rms[col]
        dict_['filter'] = filter
        post = Postprocess(**dict_)
        post.save()

    return rms

if __name__ == '__main__':
    #filter_path_name = '../filtered_7D_nions0_flat_filter5.h5'
    root_path = '../..'
    set = 'unstable_test_gen3'
    filter = 8
    leq_bound = 0
    less_bound = 10
    nns = nns_from_nndb(100)
    rms = process_nns(nns, root_path, set, filter, leq_bound, less_bound)

#results = pd.DataFrame([], index=pd.MultiIndex.from_product([['target'] + list(nns.keys()), target_names]))
