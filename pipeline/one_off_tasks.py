import json
from itertools import product
from pipeline import TrainBatch

class TrainRegressionOneBatch(TrainBatch):
    dim = 7
    plan = {'cost_l2_scale': [0.05, 0.01, 0.005, 0.002, 0.0015, 0.001, 0.0005, 0.0002, 0.0001],
            'hidden_neurons': [[30] * 3, [64] * 3],
            'filter': [5],
            'activations': ['tanh']
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter))

    with open(os.path.join(os.path.dirname(__file__), 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')
    settings['early_stop_after'] = 10

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())

class TrainReluBatch(TrainBatch):
    dim = 7
    plan = {'cost_l2_scale': [0.05, 0.1, 0.2],
            'hidden_neurons': [[30] * 3, [64] * 3, [60] * 2],
            'filter': [3, 5],
            'activations': ['relu']
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter))

    with open(os.path.join(os.path.dirname(__file__), 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')
    settings['early_stop_after'] = 10

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())

class TrainFilterTwoBatch(TrainBatch):
    dim = 7
    plan = {'cost_l2_scale': [0.05, 0.1, 0.2],
            'hidden_neurons': [[30] * 3, [64] * 3, [60] * 2],
            'filter': [2],
            'activations': ['tanh', 'relu']
            }

    plan['dataset_path'] = []
    for filter in plan.pop('filter'):
        plan['dataset_path'].append('../filtered_{!s}D_nions0_flat_filter{!s}.h5'.format(dim, filter))

    with open(os.path.join(os.path.dirname(__file__), 'default_settings.json')) as file_:
        settings = json.load(file_)
        settings.pop('train_dims')

    settings_list = []
    for val in product(*plan.values()):
        par = dict(zip(plan.keys(), val))
        par['hidden_activation'] = [par.pop('activations')] * len(par['hidden_neurons'])
        settings.update(par)
        settings_list.append(settings.copy())
