import os
from peewee import DoesNotExist
from base import DatabaseTestCase, ModelDatabaseTestCase, ModelTestCase, requires_models
from unittest import TestCase, skip
from pytest import fixture, mark
from qlknn.NNDB.model import *
from base import db
from IPython import embed

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen2_test_files'))
train_script_path = os.path.join(test_files_dir, 'train_NDNN.py')
hypercube_script_path = os.path.join(test_files_dir, 'hypercube_to_pandas.py')
filter_script_path = os.path.join(test_files_dir, 'filtering.py')
efi_network_path = os.path.join(test_files_dir, 'network_1393')
efi_div_efe_network_path = os.path.join(test_files_dir, 'network_1440')

db.execute_sql('SET ROLE testuser')


require_lists = {
    'train_script': [TrainScript],
    'network': [Network],
    'filter': [Filter]
}
require_lists['pure_network_params'] = (require_lists['network'] +
                                        require_lists['train_script'] +
                                        require_lists['filter'] +
                                        [PureNetworkParams])
require_lists['hyperparameters'] = require_lists['pure_network_params'] + [Hyperparameters]
require_lists['adam_optimizer'] = require_lists['pure_network_params'] + [AdamOptimizer]

default_dicts = {
    'train_script': {
        'script': '',
        'filter': '',
        'version': ''
    },
    'network': {
        'feature_names': ['Ate'],
        'target_names': ['efeETG_GB']
    },
    'filter': {
        'script': '',
        'id': 7
    },
    'pure_network_params': {
        'feature_prescale_bias': {'Ate': '1'},
        'feature_prescale_factor': {'Ate': '0.1'},
        'target_prescale_bias': {'efeETG_GB': '2'},
        'target_prescale_factor': {'efeETG_GB': '0.2'},
        'feature_min': {'Ate': '0'},
        'feature_max': {'Ate': '3'},
        'target_min': {'efeETG_GB': '-3'},
        'target_max': {'efeETG_GB': '-1'}
    },
    'hyperparameters': {
        'hidden_neurons'      : [30, 30, 30],
        'hidden_activation'   : ['tanh', 'tanh', 'tanh'],
        'output_activation'   : 'none',
        'standardization'     : 'normsm_1_0',
        'goodness'            : 'mse',
        'drop_chance'         : 0.0,
        'optimizer'           : 'adam',
        'cost_l2_scale'       : 8e-6,
        'cost_l1_scale'       : 0.0,
        'early_stop_after'    : 15,
        'early_stop_measure'  : 'loss',
        'minibatches'         : 10,
        'drop_outlier_above'  : .999,
        'drop_outlier_below'  : .0,
        'validation_fraction' : .1,
        'dtype'               : 'float32'
    },
    'adam_optimizer': {
        'learning_rate': 0.001,
        'beta1': .9,
        'beta2': .999
    }
}

class TestTrainScript(ModelTestCase):
    requires = require_lists['train_script']

    def test_creation(self):
        TrainScript.create(**default_dicts['train_script'])

    def test_from_file(self):
        TrainScript.from_file(train_script_path)


class TestFilter(ModelTestCase):
    requires = require_lists['filter']

    def test_creation(self):
        Filter.create(**default_dicts['filter'])

    def test_from_file(self):
        Filter.from_file(filter_script_path, hypercube_script_path)

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
    requires = require_lists['network']

    def test_creation(self):
        Network.create(**default_dicts['network'])


#def create_pure_network_params(**kwargs):
#    dicts = {}
#    for name in ['filter', 'train_script', 'network', 'pure_network_params']:
#        dicts[name] = getattr(default_dicts[' name)
#        if name in kwargs:
#            dicts[name].update(kwargs[name])
#    filter = Filter.create(**dicts['filter'])
#    train_script = TrainScript.create(**dicts['train_script'])
#    network = Network.create(**dicts['network'])
#    pure_network_params = PureNetworkParams.create(filter=filter,
#                                                   train_script=train_script,
#                                                   network=network,
#                                                   **dicts['pure_network_params'])

class TestAdamOptimizer(ModelTestCase):
    requires = require_lists['adam_optimizer']

    def test_creation(self):
        filter = Filter.create(**default_dicts['filter'])
        train_script = TrainScript.create(**default_dicts['train_script'])
        network = Network.create(**default_dicts['network'])
        pure_network_params = PureNetworkParams.create(filter=filter,
                                                train_script=train_script,
                                                network=network,
                                                **default_dicts['pure_network_params'])
        adam = AdamOptimizer.create(
            pure_network_params=pure_network_params,
            **default_dicts['adam_optimizer']
        )

class TestPureNetworkParams(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]

    @classmethod
    @db.atomic()
    def create_pure_network(cls, filter, train_script, diffdict=None):
        dict_ = default_dicts['network'].copy()
        if diffdict is not None:
            if 'network' in diffdict:
                dict_.update(diffdict['network'])
        network = Network.create(**dict_)
        nn = PureNetworkParams.create(filter=filter,
                                       train_script=train_script,
                                       network=network,
                                       **default_dicts['pure_network_params'])
        dict_ = default_dicts['hyperparameters'].copy()
        if diffdict is not None:
            if 'hyperparameters' in diffdict:
                dict_.update(diffdict['hyperparameters'])

        hyper = Hyperparameters.create(
            pure_network_params=nn,
            **dict_)
        return network

    def test_from_folder(self):
        filter = Filter.create(**default_dicts['filter'])
        PureNetworkParams.from_folder(efi_network_path)

    def test_from_function(self):
        filter = Filter.create(**default_dicts['filter'])
        train_script = TrainScript.create(**default_dicts['train_script'])
        self.create_pure_network(filter, train_script)

class TestHyperparameters(ModelTestCase):
    requires = require_lists['hyperparameters']

    @db.atomic()
    def test_creation(self):
        filter = Filter.create(**default_dicts['filter'])
        train_script = TrainScript.create(**default_dicts['train_script'])
        network = Network.create(**default_dicts['network'])
        pure_network_params = PureNetworkParams.create(filter=filter,
                                                train_script=train_script,
                                                network=network,
                                                **default_dicts['pure_network_params'])
        hyper = Hyperparameters.create(
            pure_network_params=pure_network_params,
            **default_dicts['hyperparameters']
        )


class TestPureNetworkPartnerFinding(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters]
    net_vars = ['goodness', 'cost_l2_scale', 'cost_l1_scale', 'early_stop_measure', 'early_stop_after']
    topo_vars = ['hidden_neurons', 'hidden_activation', 'output_activation']

    def setUp(self):
        super().setUp()
        self.filter = Filter.create(**default_dicts['filter'])
        self.train_script = TrainScript.create(**default_dicts['train_script'])

    @db.atomic()
    def test_find_similar_topology_by_id(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)

        query = PureNetworkParams.find_similar_topology_by_id(net1.pure_network_params.get().id)

        self.assertEqual(query.count(), 1)

        nn = query.get()
        for quantity in TestPureNetworkPartnerFinding.topo_vars:
            hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
            self.assertEqual(getattr(hyperpar1, quantity),
                             getattr(nn.hyperparameters.get(), quantity))
        self.assertEqual(net1.target_names, nn.network.target_names)
        self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
        self.assertNotEqual(net1.id, nn.id)

    @db.atomic()
    def test_find_similar_topology_by_id_no_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'hyperparameters': {'hidden_neurons': [10, 10, 10]}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        query = PureNetworkParams.find_similar_topology_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 0)

    @db.atomic()
    def test_find_similar_topology_by_id_multi_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net3 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)

        query = PureNetworkParams.find_similar_topology_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 2)

        for nn in query:
            for quantity in TestPureNetworkPartnerFinding.topo_vars:
                hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
                self.assertEqual(getattr(hyperpar1, quantity),
                                 getattr(nn.hyperparameters.get(), quantity))
            self.assertEqual(net1.target_names, nn.network.target_names)
            self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
            self.assertNotEqual(net1.id, nn.id)

    @db.atomic()
    def test_find_similar_networkpar_by_id(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)

        query = PureNetworkParams.find_similar_networkpar_by_id(net1.pure_network_params.get().id)

        self.assertEqual(query.count(), 1)

        nn = query.get()
        for quantity in TestPureNetworkPartnerFinding.net_vars:
            hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
            self.assertEqual(getattr(hyperpar1, quantity),
                             getattr(nn.hyperparameters.get(), quantity))
        self.assertEqual(net1.target_names, nn.network.target_names)
        self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
        self.assertNotEqual(net1.id, nn.id)

    @db.atomic()
    def test_find_similar_networkpar_by_id_no_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'hyperparameters': {'early_stop_after': 200}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        query = PureNetworkParams.find_similar_networkpar_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 0)

    @db.atomic()
    def test_find_similar_networkpar_by_id_multi_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net3 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)

        query = PureNetworkParams.find_similar_networkpar_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 2)

        for nn in query:
            for quantity in TestPureNetworkPartnerFinding.net_vars:
                hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
                self.assertEqual(getattr(hyperpar1, quantity),
                                 getattr(nn.hyperparameters.get(), quantity))
            self.assertEqual(net1.target_names, nn.network.target_names)
            self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
            self.assertNotEqual(net1.id, nn.id)

    @db.atomic()
    def test_find_partners_by_id(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efiITG_GB']}})

        query = PureNetworkParams.find_partners_by_id(net1.pure_network_params.get().id)

        self.assertEqual(query.count(), 1)
        self.assertNotEqual(net1.target_names, net2.target_names)

        nn = query.get()
        for quantity in TestPureNetworkPartnerFinding.net_vars + TestPureNetworkPartnerFinding.topo_vars:
            hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
            self.assertEqual(getattr(hyperpar1, quantity),
                             getattr(nn.hyperparameters.get(), quantity))
        self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
        self.assertNotEqual(net1.id, nn.id)

    @db.atomic()
    def test_find_similar_networkpar_by_id_no_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'hyperparameters': {'hidden_neurons': [10, 10, 10]}})
        net3 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'hyperparameters': {'early_stop_after': 200}})
        query = PureNetworkParams.find_partners_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 0)

    @db.atomic()
    def test_find_similar_networkpar_by_id_multi_match(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        net3 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efiITG_GB']}})

        query = PureNetworkParams.find_partners_by_id(net1.pure_network_params.get().id)
        self.assertEqual(query.count(), 2)

        for nn in query:
            for quantity in ['goodness', 'cost_l2_scale', 'cost_l1_scale', 'early_stop_measure', 'early_stop_after']:
                hyperpar1 = net1.pure_network_params.get().hyperparameters.get()
                self.assertEqual(getattr(hyperpar1, quantity),
                                 getattr(nn.hyperparameters.get(), quantity))
            self.assertNotEqual(net1.pure_network_params.get().id, nn.id)
            self.assertNotEqual(net1.id, nn.id)

class TestDivsumCreation(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters]

    def setUp(self):
        super().setUp()
        self.filter = Filter.create(**default_dicts['filter'])
        self.train_script = TrainScript.create(**default_dicts['train_script'])

    def test_2D_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efeITG_GB', 'efiITG_GB']}})
        with self.assertRaises(ValueError):
            Network.divsum_from_div_id(net1.id)
        self.assertEqual(Network.select().count(), 1)

    def test_non_div_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script)
        self.assertEqual(len(net1.target_names), 1)
        self.assertNotRegexpMatches(net1.target_names[0], '.*div.*')
        with self.assertRaises(ValueError):
            Network.divsum_from_div_id(net1.id)
        self.assertEqual(Network.select().count(), 1)

    def test_non_existing_recipe_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['foo_div_bar']}})
        with self.assertRaises(NotImplementedError):
            Network.divsum_from_div_id(net1.id)
        self.assertEqual(Network.select().count(), 1)

    def test_efe_div_efi_not_exist_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB_div_efi_GB']}})
        with self.assertRaises(DoesNotExist):
            Network.divsum_from_div_id(net1.id, stop_on_missing=True)

    def test_efe_div_efi_not_exist_continue_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB_div_efi_GB']}})
        Network.divsum_from_div_id(net1.id, stop_on_missing=False)
        self.assertEqual(Network.select().count(), 1)

    @db.atomic()
    def test_efe_div_efi_to_divsum(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB_div_efi_GB']}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efi_GB']}})
        Network.divsum_from_div_id(net1.id)
        self.assertEqual(Network.select().count(), 4)

        combo_net = Network.select().where(Network.target_names == ['efe_GB'])
        self.assertEqual(combo_net.count(), 1)
        combo_net = combo_net.get()
        self.assertListEqual(combo_net.networks, [net1.id, net2.id])

        multi_net = Network.select().where(Network.target_names == ['efe_GB', 'efi_GB'])
        self.assertEqual(multi_net.count(), 1)
        multi_net = multi_net.get()
        self.assertListEqual(multi_net.networks, [combo_net.id, net2.id])

    def test_find_divsum_candidates(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB_div_efi_GB']}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efi_GB']}})

        Network.find_divsum_candidates()
        self.assertEqual(Network.select().count(), 4)


input = pd.DataFrame()
scann = 100
input['Ati'] = np.array(np.linspace(2,13, scann))
input['Ti_Te']  = np.full_like(input['Ati'], 1.)
input['An']  = np.full_like(input['Ati'], 2.)
input['Ate']  = np.full_like(input['Ati'], 5.)
input['qx'] = np.full_like(input['Ati'], 0.660156)
input['smag']  = np.full_like(input['Ati'], 0.399902)
input['x']  = np.full_like(input['Ati'], 0.449951)


class TestComboNetworks(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]

    def setUp(self):
        super().setUp()
        self.filter = Filter.create(**default_dicts['filter'])
        self.train_script = TrainScript.create(**default_dicts['train_script'])

    def test_multiply_create(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB_div_efi_GB']}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efi_GB']}})
        combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='nn0 * nn1')

    def test_multiply_to_QuaLiKizNN(self):
        net1 = PureNetworkParams.from_folder(efi_network_path)
        net2 = PureNetworkParams.from_folder(efi_div_efe_network_path)
        combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='nn0 * nn1')
        nn = combo_net.to_QuaLiKizNN()

    def test_multiply_get_output(self):
        net1 = PureNetworkParams.from_folder(efi_network_path)
        net2 = PureNetworkParams.from_folder(efi_div_efe_network_path)
        combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='nn0 * nn1')
        nn = combo_net.to_QuaLiKizNN()
        nn.get_output(input)

class TestMultiNetworks(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]

    def setUp(self):
        super().setUp()
        self.filter = Filter.create(**default_dicts['filter'])
        self.train_script = TrainScript.create(**default_dicts['train_script'])

    def test_create(self):
        net1 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efe_GB']}})
        net2 = TestPureNetworkParams.create_pure_network(self.filter, self.train_script,
                {'network': {'target_names': ['efi_GB']}})
        multi_net = Network.create(target_names=['efe_GB', 'efi_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='np.hstack(args)')

    def test_to_QuaLiKizNN(self):
        net1 = PureNetworkParams.from_folder(efi_network_path)
        net2 = PureNetworkParams.from_folder(efi_div_efe_network_path)
        combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='nn0 * nn1')
        multi_net = Network.create(target_names=['efe_GB', 'efi_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[combo_net.id, net2.id],
                                  recipe='np.hstack(args)')
        nn = multi_net.to_QuaLiKizNN()

    def test_multiply_get_output(self):
        net1 = PureNetworkParams.from_folder(efi_network_path)
        net2 = PureNetworkParams.from_folder(efi_div_efe_network_path)
        combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[net1.id, net2.id],
                                  recipe='nn0 * nn1')
        multi_net = Network.create(target_names=['efe_GB', 'efi_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[combo_net.id, net2.id],
                                  recipe='np.hstack(args)')
        nn = multi_net.to_QuaLiKizNN()
        nn.get_output(input)

class TestRecursiveAttributes(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]

    def setUp(self):
        super().setUp()
        self.filter = Filter.create(**default_dicts['filter'])
        self.train_script = TrainScript.create(**default_dicts['train_script'])

        self.net1 = PureNetworkParams.from_folder(efi_network_path)
        self.net2 = PureNetworkParams.from_folder(efi_div_efe_network_path)

        # These parameters should be the same for ALL networks:
        hyperpar1 = self.net1.pure_network_params.get().hyperparameters.get()
        for net in [self.net2]:
            hyperpar = net.pure_network_params.get().hyperparameters.get()
            for param in ['hidden_neurons']:
                assert getattr(hyperpar1, param) == getattr(hyperpar, param)

        self.combo_net = Network.create(target_names=['efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[self.net1.id, self.net2.id],
                                  recipe='nn0 * nn1')
        self.multi_net = Network.create(target_names=['efe_GB', 'efi_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[self.combo_net.id, self.net2.id],
                                  recipe='np.hstack(args)')

    def test_flatten_recursive(self):
        arr = np.array([128, 128, 128])

        nested = [arr]
        flat = Network.flatten_recursive(nested)
        self.assertNumpyArrayListEqual(flat, [arr])

        nested = [arr, arr]
        flat = Network.flatten_recursive(nested)
        self.assertNumpyArrayListEqual(flat, [arr, arr])

        nested = [[arr, arr], arr]
        flat = Network.flatten_recursive(nested)
        self.assertNumpyArrayListEqual(flat, [arr, arr, arr])

        nested = [[arr, arr, arr], arr, [arr, arr, arr]]
        flat = Network.flatten_recursive(nested)
        self.assertNumpyArrayListEqual(flat, [arr, arr, arr, arr, arr, arr, arr])

    def test_get_recursive_pure_arrays(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'hidden_neurons'
        param = np.array(getattr(hyperpar, param_name))
        desired = param
        rec_param = self.net1.get_recursive_hyperparameter(param_name)
        self.assertNumpyArrayEqual(desired, rec_param)
        self.assertNumpyArrayEqual(np.array([128, 128, 128]), rec_param)

    def test_get_recursive_combo_arrays(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'hidden_neurons'
        param = np.array(getattr(hyperpar, param_name))
        desired = [param] * 2
        rec_param = self.combo_net.get_recursive_hyperparameter(param_name)
        self.assertNumpyArrayListEqual(desired, rec_param)
        arr = np.array([128, 128, 128])
        manual = [arr, arr]
        self.assertNumpyArrayListEqual(manual, rec_param)

    def test_get_recursive_multi_arrays(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'hidden_neurons'
        param = np.array(getattr(hyperpar, param_name))
        desired = [[param] * 2, param]
        rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
        self.assertNumpyArrayListEqual(desired, rec_param)
        arr = np.array([128, 128, 128])
        manual = [[arr, arr], arr]
        self.assertNumpyArrayListEqual(manual, rec_param)

    def test_get_recursive_multi_arrays_flipped(self):
        self.multi_net = Network.create(target_names=['efi_GB', 'efe_GB'],
                                  feature_names=['Ati'],
                                  filter=self.filter,
                                  train_script=self.train_script,
                                  networks=[self.net2.id, self.combo_net.id],
                                  recipe='np.hstack(args)')
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'hidden_neurons'
        param = np.array(getattr(hyperpar, param_name))
        desired = [param, [param] * 2]
        rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
        self.assertNumpyArrayListEqual(desired, rec_param)
        arr = np.array([128, 128, 128])
        manual = [arr, [arr, arr]]
        self.assertNumpyArrayListEqual(manual, rec_param)

    def test_get_recursive_pure_floats(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'cost_l2_scale'
        param = np.array(getattr(hyperpar, param_name))
        desired = param
        rec_param = self.net1.get_recursive_hyperparameter(param_name)
        self.assertEqual(desired, rec_param)
        self.assertEqual(8e-6, rec_param)

    def test_get_recursive_combo_floats(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'cost_l2_scale'
        param = getattr(hyperpar, param_name)
        desired = [param] * 2
        rec_param = self.combo_net.get_recursive_hyperparameter(param_name)
        self.assertSequenceEqual(desired, rec_param)
        self.assertSequenceEqual([8.e-6, 8e-6], rec_param)

    def test_get_recursive_multi_floats(self):
        hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
        param_name = 'cost_l2_scale'
        param = getattr(hyperpar, param_name)
        desired = [[param] * 2, param]
        rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
        self.assertSequenceEqual(desired, rec_param)
        self.assertSequenceEqual([[8.e-6, 8e-6], 8e-6], rec_param)

if __name__ == '__main__':
    embed()

