import os
import shutil
import tempfile
import json

from unittest import TestCase
from IPython import embed

def skip_if(expr):
    def decorator(method):
        @wraps(method)
        def inner(self):
            should_skip = expr() if callable(expr) else expr
            if not should_skip:
                return method(self)
            elif VERBOSITY > 1:
                print('Skipping %s test.' % method.__name__)
        return inner
    return decorator


def skip_unless(expr):
    return skip_if((lambda: not expr()) if callable(expr) else not expr)


def skip_case_if(expr):
    def decorator(klass):
        should_skip = expr() if callable(expr) else expr
        if not should_skip:
            return klass
        elif VERBOSITY > 1:
            print('Skipping %s test.' % klass.__name__)
            class Dummy(object): pass
            return Dummy
    return decorator


def skip_case_unless(expr):
    return skip_case_if((lambda: not expr()) if callable(expr) else not expr)

test_files_dir = os.path.abspath(os.path.join(__file__, '../gen3_test_files'))
efi_network_path = os.path.join(test_files_dir, 'network_1393')

default_train_settings = {'dataset_path': os.path.join(test_files_dir, 'unstable_training_gen3_4D_nions0_flat_filter8.h5.1'),
                    'drop_outlier_above': 0.999,
                    'drop_outlier_below': 0.001,
                    'hidden_neurons': [16, 16],
                    'hidden_activation': ['tanh', 'tanh'],
                    'drop_chance': 0.0,
                    'output_activation': 'none',
                    'standardization': 'normsm_1_0',
                    'calc_standardization_on_nonzero': True,
                    'goodness_only_on_unstable': True,
                    'goodness': 'mse',
                    'cost_l2_scale': 8e-06,
                    'cost_l1_scale': 0.0,
                    'cost_stable_positive_scale': 0.0,
                    'cost_stable_positive_offset': -5.0,
                    'cost_stable_positive_function': "block",
                    'early_stop_after': 15,
                    'early_stop_measure': 'loss',
                    'minibatches': 10,
                    'weight_init': 'normsm_1_0',
                    'bias_init': 'normsm_1_0',
                    'validation_fraction': 0.05,
                    'test_fraction': 0.05,
                    'dtype': 'float32',
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'lbfgs_maxfun': 1000,
                    'lbfgs_maxiter': 15000,
                    'lbfgs_maxls': 20,
                    'adam_beta1': 0.9,
                    'adam_beta2': 0.999,
                    'adadelta_rho': 0.95,
                    'rmsprop_decay': 0.9,
                    'rmsprop_momentum': 0.0,
                    'max_epoch': 1,
                    'steps_per_report': None,
                    'epochs_per_report': None,
                    'save_checkpoint_networks': None,
                    'save_best_networks': None,
                    'track_training_time': None,
                    'train_dims': ['efiITG_GB']}
