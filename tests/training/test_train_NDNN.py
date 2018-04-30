import os
import shutil
import tempfile

from unittest import TestCase, skip
from IPython import embed

from qlknn.training.train_NDNN import *

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen3_test_files'))
efi_network_path = os.path.join(test_files_dir, 'network_1393')

class TrainNDNNTestCase(TestCase):
    def setUp(self):
        self.settings = {'dataset_path': os.path.join(test_files_dir, 'unstable_training_gen3_4D_nions0_flat_filter8.h5.1'),
                    'drop_outlier_above': 0.999,
                    'drop_outlier_below': 0.001,
                    'hidden_neurons': [16, 16],
                    'hidden_activation': ['tanh', 'tanh'],
                    'drop_chance': 0.0,
                    'output_activation': 'none',
                    'standardization': 'normsm_1_0',
                    'goodness': 'mse',
                    'cost_l2_scale': 8e-06,
                    'cost_l1_scale': 0.0,
                    'early_stop_after': 15,
                    'early_stop_measure': 'loss',
                    'minibatches': 10,
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

        #self.train_nn = TrainNN(settings=settings,
        #                        train_dims=['efiITG_GB'],
        #                        uid='test')
        self.test_dir = tempfile.mkdtemp(prefix='test_')
        with open(os.path.join(self.test_dir, 'settings.json'), 'w') as file_:
            settings = json.dump(self.settings, file_)
        #train(settings, warm_start_nn=warm_start_nn)
        os.old_dir = os.curdir
        os.chdir(self.test_dir)

        super(TrainNDNNTestCase, self).setUp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        os.chdir(os.old_dir)
        super(TrainNDNNTestCase, self).setUp()

class TestTrainNN(TrainNDNNTestCase):

    def test_launch_train_NDNN(self):
        train(self.settings)
