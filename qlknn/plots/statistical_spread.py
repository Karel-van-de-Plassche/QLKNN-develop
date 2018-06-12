import re

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from peewee import AsIs, JOIN, prefetch, SQL
from IPython import embed

from qlknn.NNDB.model import Network, PureNetworkParams, PostprocessSlice, NetworkMetadata, TrainMetadata, Postprocess, db, Hyperparameters

def is_same_hyperpar(hyperpars):
    same_hyperpar = True
    for name, val in hyperpars.items():
        attr = getattr(Hyperparameters, name)
        if isinstance(val, float):
            attr = attr.cast('numeric')
        if val is not None:
            same_hyperpar &= attr == val
        else:
            same_hyperpar &= attr.is_null()
    return same_hyperpar

def get_stats_from_query(query):
    if query.count() > 0:
        results = list(query.dicts())
        df = pd.DataFrame(results)
        #df['network'] = df['network'].apply(lambda el: 'pure_' + str(el))
        #df['l2_norm'] = df['l2_norm'].apply(np.nanmean)
        df.drop(['id', 'network'], inplace=True, axis='columns')
        df.set_index('network_id', inplace=True)
        stats = df
    stats = stats.applymap(np.array)
    stats = stats.applymap(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x)
    stats.dropna(axis='columns', how='all', inplace=True)
    stats.dropna(axis='rows', how='any', inplace=True)

    return stats

def extract_statistics(stats, hyperpars, goodness_pars):
    stats = stats.loc[:, hyperpars + goodness_pars]
    report = pd.DataFrame()
    for statname in goodness_pars:
        stat = stats[statname]
        subreport = pd.Series(name=statname)
        subreport['mean'] = stat.mean()
        subreport['stddev'] = stat.std()
        subreport['stderr'] = subreport['stddev'] / np.sqrt(2 * len(stat) - 2)
        report[statname] = subreport
    return report

def get_base_stats(target_names, hyperpars, goodness_pars, hyperpars_const=None):
    if hyperpars_const is None:
        hyperpars_const = {
            'cost_l2_scale': 5e-5,
            'cost_stable_positive_scale': 0,
            'cost_stable_positive_offset': None,
        }
    same_hyperpar = is_same_hyperpar(hyperpars_const)
    query = (Network.select(Network.id.alias('network_id'),
                            PostprocessSlice,
                            Postprocess.rms,
                            Hyperparameters
             )
             .join(PostprocessSlice, JOIN.LEFT_OUTER)
             .switch(Network)
             .join(Postprocess, JOIN.LEFT_OUTER)
             .switch(Network)
             .where(Network.target_names == target_names)
             .switch(Network)
             .join(PureNetworkParams)
             .join(Hyperparameters)
             .where(same_hyperpar)
             )
    stats = get_stats_from_query(query)

    report = extract_statistics(stats, hyperpars, goodness_pars)
    return report

if __name__ == '__main__':
    target_names = ['efeTEM_GB']
    hyperpars = ['cost_stable_positive_scale', 'cost_l2_scale']
    goodness_pars = ['no_pop_frac', 'no_thresh_frac', 'rms', 'pop_abs_mis_median', 'thresh_rel_mis_median', 'wobble_qlkunstab']
    report = get_base_stats(target_names, hyperpars, goodness_pars, hyperpars_const=None )
    print(report)
