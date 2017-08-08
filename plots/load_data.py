from IPython import embed
#import mega_nn
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import sys
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
NNDB_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../NNDB'))
sys.path.append(networks_path)
sys.path.append(NNDB_path)
from model import Network, NetworkJSON
from run_model import QuaLiKizNDNN

import matplotlib.pyplot as plt

def load_data(id):
    store = pd.HDFStore('../7D_nions0_flat.h5')
    input = store['megarun1/input']
    data = store['megarun1/flattened']

    root_name = '/megarun1/nndb_nn/'

    query = (Network.select(Network.target_names).where(Network.id == id).tuples()).get()
    target_names = query[0]
    if len(target_names) == 1:
        target_name = target_names[0]
    else:
        NotImplementedError('Multiple targets not implemented yet')

    print(target_name)
    parent_name = root_name + target_name + '/'
    network_name = parent_name + str(id)
    network_name += '_noclip'
    nn = load_nn(id)


    df = data[target_name].to_frame('target')
    df['prediction'] = store[network_name].iloc[:, 0]
    df = df.astype('float64')
    df['residuals'] = df['target'] - df['prediction']
    return input, df, nn

def load_nn(id):
    subquery = (Network.select(NetworkJSON.network_json)
                .where(Network.id == id)
                .join(NetworkJSON)
                .tuples()).get()
    json_dict = subquery[0]
    nn = QuaLiKizNDNN(json_dict)
    return nn
