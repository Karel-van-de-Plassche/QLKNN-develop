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
    similar = [(res.hyperparameters.get().cost_l2_scale, res.id) for res in query]
    similar.sort()
    labels = [(line[1], '$c_{L2} = ' + str(line[0]) + '$') for line in similar]
    print('nn_list = OrderedDict([', end='')
    print(*labels, sep=',\n', end='')
    print('])', end='')

find_similar_regularization(37)
