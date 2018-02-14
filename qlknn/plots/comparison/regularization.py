from IPython import embed
import numpy as np
import scipy.stats as stats
import pandas as pd

import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
training_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../training'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
sys.path.append(training_path)
from model import Network, NetworkJSON, Hyperparameters
from run_model import QuaLiKizNDNN
from train_NDNN import shuffle_panda

from peewee import Param, Passthrough

import matplotlib.pyplot as plt
from matplotlib import gridspec
from load_data import load_data, load_nn

from query_to_networks import results_to_sorted

def find_similar_regularization(network_id):
    query = Network.find_similar_topology_by_id(network_id)
    query &= Network.find_similar_trainingpar_by_id(network_id)

    train_dim, goodness, cost_l2_scale, cost_l1_scale, early_stop_measure, filter_id = (
        Network
        .select(Network.target_names,
                Hyperparameters.goodness,
                Hyperparameters.cost_l2_scale,
                Hyperparameters.cost_l1_scale,
                Hyperparameters.early_stop_measure,
                Network.filter_id)
        .where(Network.id == network_id)
        .join(Hyperparameters)
    ).tuples().get()

    query &= (Network.select()
             .where(Network.target_names ==
                    Param(train_dim))
             .where(Network.filter_id ==
                    filter_id)
             .join(Hyperparameters)
             .where(Hyperparameters.goodness ==
                    goodness)
             #.where(Hyperparameters.cost_l2_scale ==
             #       Passthrough(str(cost_l2_scale)))
             .where(Hyperparameters.cost_l1_scale ==
                    Passthrough(str(cost_l1_scale)))
             .where(Hyperparameters.early_stop_measure ==
                    early_stop_measure)
    )
    df = []
    for res in query:
        df.append((res.id, res.hyperparameters.get().cost_l2_scale, res.network_metadata.get().rms_test))

    varname = 'c_L2'
    df_trim = results_to_sorted('c_L2', df)
    print(df_trim)
    labels = [(line[0], '$c_{L2} = ' + str(line[1]) + '$') for line in df_trim[['id', 'c_L2']].values]
    print('nn_list = OrderedDict([', end='')
    print(*labels, sep=',\n', end='')
    print('])')
    embed()

find_similar_regularization(37)
