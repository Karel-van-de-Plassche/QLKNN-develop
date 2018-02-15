import os
from base import DatabaseTestCase, ModelDatabaseTestCase, ModelTestCase, requires_models
from unittest import TestCase, skip
from pytest import fixture, mark
from qlknn.NNDB.model import *
from IPython import embed

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen2_test_files'))
train_script_path = os.path.join(test_files_dir, 'train_NDNN.py')
filter_script_path = os.path.join(test_files_dir, 'filtering.py')
efi_network_path = os.path.join(test_files_dir, 'network_1393')
efi_div_efe_network_path = os.path.join(test_files_dir, 'network_1440')

class FilterTestCase(ModelTestCase):
    requires = [Filter]

    @classmethod
    @db.atomic()
    def create_filter(cls):
        filter = Filter.create(script='')
        Filter.update({'id': 7}).where(Filter.id == filter.id).execute()
        return Filter.get_by_id(7)


class PureNetworkTestCase(ModelTestCase):
    requires = [PureNetworkParams, TrainScript, Network] + FilterTestCase.requires

    @db.atomic()
    def create_pure_network_params(self):
        self.train_script = TrainScript.create(script='', version='')
        self.network = Network.create(feature_names=['Ate'],
                                      target_names=['efeETG_GB'])
        self.filter = FilterTestCase.create_filter()

        self.pure_network_params = PureNetworkParams.create(
            network=self.network,
            filter=self.filter,
            train_script=self.train_script,
            feature_prescale_bias={'Ate': '1'},
            feature_prescale_factor={'Ate': '0.1'},
            target_prescale_bias={'efeETG_GB': '2'},
            target_prescale_factor={'efeETG_GB': '0.2'},
            feature_min={'Ate': '0'},
            feature_max={'Ate': '3'},
            target_min={'efeETG_GB': '-3'},
            target_max={'efeETG_GB': '-1'}
        )

    def tearDown(self):
        super().tearDown()


class TestTrainScript(ModelTestCase):
    requires = [TrainScript]

    def test_creation(self):
        TrainScript.create(script='', version='')

    def test_from_file(self):
        TrainScript.from_file(train_script_path)


class TestFilter(FilterTestCase):
    requires = [Filter]

    def test_creation(self):
        Filter.create(script='')

    def test_from_file(self):
        Filter.from_file(filter_script_path)

    def test_find_by_path_name(self):
        # name: (stable/unstable, sane/test/training, dim, filter#
        name_map = {'unstable_training_7D_nions0_flat_filter3.h5':
                        ('unstable', 'training', 7, 3),
                    'sane_gen2_7D_nions0_flat_filter9.h5':
                        ('', 'sane', 7, 9),
                    'test_gen2_9D_nions0_flat_filter4.h5':
                        ('', 'test', 9, 4),
                    './test_gen2_9D_nions0_flat_filter4.h5':
                        ('', 'test', 9, 4),
                    'folder/unstable_training_gen2_9D_nions0_flat_filter4.h5':
                        ('', 'test', 9, 4),
                    '/folder/unstable_training_gen2_9D_nions0_flat_filter4.h5':
                        ('', 'test', 9, 4),
                    }
        for name, (stab_unstab, subset, dim, filter_id) in name_map.items():
            calc_filter_id = Filter.find_by_path_name(name)
            self.assertEqual(calc_filter_id, filter_id)


class TestNetwork(ModelTestCase):
    requires = [Network]

    def test_creation(self):
        Network.create(feature_names=['Ate'],
                       target_names=['efeETG_GB'])


class TestAdamOptimizer(PureNetworkTestCase):
    requires = PureNetworkTestCase.requires + [AdamOptimizer]

    def test_creation(self):
        self.create_pure_network_params()
        adam = AdamOptimizer.create(
            pure_network_params_id=self.pure_network_params,
            learning_rate=0.001,
            beta1=.9,
            beta2=.999
        )


class TestHyperparameters(PureNetworkTestCase):
    requires = PureNetworkTestCase.requires + [Hyperparameters]

    def test_creation(self):
        self.create_pure_network_params()
        hyper = Hyperparameters.create(
            pure_network_params=self.pure_network_params,
            hidden_neurons=[30, 30, 30],
            hidden_activation=['tanh', 'tanh', 'tanh'],
            output_activation='none',
            standardization='normsm_1_0',
            goodness='mse',
            drop_chance=0.0,
            optimizer='adam',
            cost_l2_scale=8e-6,
            cost_l1_scale=0.0,
            early_stop_after=15,
            early_stop_measure='loss',
            minibatches=10,
            drop_outlier_above=.999,
            drop_outlier_below=.0,
            validation_fraction=.1,
            dtype='float32'
        )


class TestPureNetworkParams(PureNetworkTestCase):
    requires = PureNetworkTestCase.requires + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]

    def test_from_folder(self):
        FilterTestCase.create_filter()
        PureNetworkParams.from_folder(efi_network_path)
