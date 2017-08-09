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

def find_similar_topology(network_id):
    #query = Network.find_similar_topology_by_id(network_id)
    query = Network.find_similar_networkpar_by_id(network_id)
    query &= Network.find_similar_trainingpar_by_id(network_id)

    train_dim, hidden_neurons, hidden_activation, output_activation, filter_id = (
        Network
        .select(Network.target_names,
                Hyperparameters.hidden_neurons,
                Hyperparameters.hidden_activation,
                Hyperparameters.output_activation,
                Network.filter_id)
        .where(Network.id == network_id)
        .join(Hyperparameters)
    ).tuples().get()

    query &= (Network.select()
             .where(Network.target_names == Param(train_dim))
             #.where(Hyperparameters.hidden_neurons == hidden_neurons)
             #.where(Hyperparameters.hidden_activation == Param(hidden_activation))
             #.where(Hyperparameters.output_activation == output_activation)
             .join(Hyperparameters)
    )
    similar = [(res.hyperparameters.get().hidden_neurons , res.id) for res in query]
    similar.sort()
    embed()
    labels = [(line[1], '$topo = ' + str(line[0]) + '$') for line in similar]
    print('nn_list = OrderedDict([', end='')
    print(*labels, sep=',\n', end='')
    print('])', end='')

find_similar_topology(37)
