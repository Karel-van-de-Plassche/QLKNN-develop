import inspect
import os
import sys
import json
import subprocess
import socket
import re
import traceback
import operator
from warnings import warn
from functools import reduce
from itertools import chain
from collections import OrderedDict

import numpy as np
import scipy.io as io
import pandas as pd
from peewee import (Model,
                    FloatField, FloatField, IntegerField, BooleanField, TextField, ForeignKeyField, DateTimeField,
                    ProgrammingError, AsIs, fn, SQL)
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from playhouse.hybrid import hybrid_property
#from playhouse.shortcuts import RetryOperationalError #peewee==2.10.1
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN, QuaLiKizMultiNN

#class RetryPostgresqlExtDatabase(RetryOperationalError, PostgresqlExtDatabase):
#    pass
#db = RetryPostgresqlExtDatabase(database='nndb', host='gkdb.org')
db = PostgresqlExtDatabase(database='nndb', host='gkdb.org')

class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class TrainScript(BaseModel):
    script = TextField()
    version = TextField()

    @classmethod
    def from_file(cls, pwd):
        with open(pwd, 'r') as script:
            script = script.read()

        train_script_query = TrainScript.select().where(TrainScript.script == script)
        if train_script_query.count() == 0:
            with db.atomic() as txn:
                stdout = subprocess.check_output('git rev-parse HEAD',
                                           shell=True)
                version = stdout.decode('UTF-8').strip()
                train_script = TrainScript(
                    script=script,
                    version=version
                )
                train_script.save()
        elif train_script_query.count() == 1:
            train_script = train_script_query.get()
        else:
            raise Exception('multiple train scripts found. Could not choose')
        return train_script

class Filter(BaseModel):
    script = TextField()
    description = TextField(null=True)
    min = FloatField(null=True)
    max = FloatField(null=True)
    remove_negative = BooleanField(null=True)
    remove_zeros = BooleanField(null=True)
    gam_filter = BooleanField(null=True)
    ck_max = FloatField(null=True)
    diffsep_max = FloatField(null=True)

    @classmethod
    def from_file(cls, pwd):
        with db.atomic() as txn:
            with open(pwd, 'r') as script:
                script = script.read()
            filter = Filter(script=script)
            filter.save()

    @classmethod
    def find_by_path_name(cls, name):
        split = re.split('(?:(unstable)_|)(sane|test|training)_(?:gen(\d+)_|)(\d+)D_nions0_flat_filter(\d+).h5', name)
        try:
            if len(split) != 7:
                raise
            filter_id = int(split[5])
        except:
            raise Exception('Could not find filter ID from name "{!s}"'.format(name))
        return filter_id


class ComboNetwork(BaseModel):
    target_names = ArrayField(TextField)
    recipe = TextField()
    feature_names = ArrayField(TextField)
    networks = ArrayField(IntegerField)

    @hybrid_property
    def hidden_neurons(self):
        return [Network.get_by_id(nn).hyperparameters.get().hidden_neurons for nn in self.networks]

    @hidden_neurons.expression
    def hidden_neurons(cls):
        raise NotImplementedError('Cannot use in SQL query')

    def to_QuaLiKizComboNN(self):
        network_ids = self.networks
        networks = [Network.get_by_id(num).to_QuaLiKizNDNN() for num in network_ids]
        recipe = self.recipe
        for ii in range(len(network_ids)):
            recipe = recipe.replace('nn' + str(ii), 'args[' + str(ii) + ']')
        exec('def combo_func(*args): return ' + recipe, globals())
        return QuaLiKizComboNN(self.target_names, networks, combo_func)

    to_QuaLiKizNN = to_QuaLiKizComboNN

    @classmethod
    def find_divsum_candidates(cls):
        query = (Network
                 .select()
                 .where(Network.target_names[0] % '%_div_%')
                 .where(SQL('array_length(target_names, 1) = 1'))
                 )
        for network in query:
            try:
                cls.divsum_from_div_id(network.id)
            except Exception:
                traceback.print_exc()

    @classmethod
    def divsum_from_div_id(cls, network_id):
        query = (Network
                 .select()
                 .where(Network.id == network_id)
                 )
        nn = query.get()
        if len(nn.target_names) != 1:
            raise Exception('Divsum network needs div network, not {!s}'.format(nn.target_names))
        target_name = nn.target_names[0]
        print('Trying to make combine Network {:d} with target {!s}'.format(nn.id, target_name))
        splitted = re.compile('(.*)_(div|plus)_(.*)').split(target_name)
        if len(splitted) != 5:
            raise Exception('Could not split {!s} in divsum parts'.format(target_name))

        partner_target_sets = []
        formula_sets = []
        if splitted[2] == 'div':
            if splitted[1].startswith('efi') and splitted[3].startswith('efe'):
                # If it is efi / efe
                partner_targets = [[splitted[1] + '_plus_' + splitted[3]]]
                formulas = OrderedDict([(splitted[1], '(nn{0:d} * nn{1:d}) / (nn{0:d} + 1)'),
                                        (splitted[3], '(nn{1:d}) / (nn{0:d} + 1)')])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efe') and splitted[3].startswith('efi'):
                # If it is efe / efi
                partner_targets = [[splitted[3]]]
                formulas = OrderedDict([
                    (splitted[3], 'nn{1:d}'),
                    (splitted[1], '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efi'):
                # If it is pfe / efi
                pfe = splitted[1]
                efi = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                # Triplet style: pfe / efi == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                    (efe, 'nn{1:d} / (1 + nn{0:d} + nn{2:d})'),
                    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
                # Simple style: pfe / efi == nn0, efi == nn1, efe / efi == nn2
                partner_targets = [[efi],
                                   [efe + '_div_' + efi]
                                   ]
                formulas = OrderedDict([
                    (efi, 'nn{1:d}'),
                    (efe, '(nn{1:d} * nn{2:d})'),
                    (pfe, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efi') and splitted[3].startswith('pfe'):
                # If it is efi / pfe
                efi = splitted[1]
                pfe = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                # Triplet style: efi / pfe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efi, '(nn{0:d} * nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                    (efe, '(nn{0:d} * nn{1:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
                # Heatflux style: efi / pfe == nn0, efi + efe == nn1, efi / efe == nn2
                partner_targets = [[efi + '_plus_' + efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} * (1 + nn{2:d}))')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efe'):
                # If it is pfe / efe
                pfe = splitted[1]
                efe = splitted[3]
                split_efe = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efe)
                efi = ''.join(*[[split_efe[0]] + ['i'] + split_efe[2:]])
                # Triplet style: pfe / efe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                    (efe, '(nn{1:d}) / (1 + nn{1:d} + nn{2:d})'),
                    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
                # Heatflux style: pfe / efe == nn0, efi + efe == nn1, efi / efe == nn2
                partner_targets = [[efi + '_plus_' + efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                    (pfe, '(nn{0:d} * nn{1:d} * nn{2:d}) / (1 + nn{2:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            else:
                raise NotImplementedError("Div style network {:d} with target {!s} and first part '{!s}'".format(network_id, target_name, splitted[0]))
        else:
            raise Exception('Divsum network needs div network, not {!s}'.format(nn.target_names))

        for formulas, partner_targets in zip(formula_sets, partner_target_sets):
            nns = [nn]
            skip = False
            for partner_target in partner_targets:
                if len(partner_target) > 1:
                    raise Exception('Multiple partner targets!')
                query = Network.find_similar_topology_by_id(network_id, match_train_dim=False)
                query &= Network.find_similar_networkpar_by_id(network_id, match_train_dim=False)
                query &= (Network
                     .select()
                     .where(Network.target_names == AsIs(partner_target))
                     )
                if query.count() > 1:
                    print('Found {:d} matches for {!s}'.format(query.count(), partner_target))
                    try:
                        sort = sorted([(el.postprocess.get().rms, el.id) for el in query])
                    except Postprocess.DoesNotExist as ee:
                        net_id = re.search('PARAMS: \[(.*)\]', ee.args[0])[1]
                        table_field = re.search('WHERE \("t1"."(.*)"', ee.args[0])[1]
                        raise Exception('{!s} {!s} does not exist! Run postprocess.py'.format(table_field, net_id))
                    print('Selected {1:d} with RMS val {0:.2f}'.format(*sort[0]))
                    query = (Network
                             .select()
                             .where(Network.id == sort[0][1])
                    )
                elif query.count() == 0:
                    print('No match for {!s}! Skipping..'.format(partner_target))
                    skip = True

                if query.count() > 0:
                    nns.append(query.get())


            if skip is not True:
                recipes = OrderedDict()
                network_ids = [nn.id for nn in nns]
                for target, formula in formulas.items():
                    recipes[target] = formula.format(*list(range(len(nns))))

                combonets = []
                purenets = []
                for target, recipe in recipes.items():
                    if all([el not in recipe for el in ['+', '-', '/', '*']]):
                        net_num = int(recipe.replace('nn', ''))
                        net_id = network_ids[net_num]
                        purenets.append(Network.get_by_id(net_id))
                    else:
                        query = (ComboNetwork.select()
                                 .where((ComboNetwork.recipe == recipe) &
                                        (ComboNetwork.networks == AsIs(network_ids)))
                                 )
                        if query.count() == 0:
                            combonet = cls(target_names=[target],
                                           feature_names=nn.feature_names,
                                           recipe=recipe,
                                           networks=network_ids)
                            #raise Exception(combonet.recipe + ' ' + str(combonet.networks))
                            combonet.save()
                            print('Created ComboNetwork {:d} with recipe {!s} and networks {!s}'.format(combonet.id, recipe, network_ids))
                        elif query.count() == 1:
                            combonet = query.get()
                            print('Network with recipe {!s} and networks {!s} already exists! Skipping!'.format(recipe, network_ids))
                        else:
                            raise NotImplementedError('Duplicate recipies! How could this happen..?')

                        combonets.append(combonet)

                flatten = lambda l: [item for sublist in l for item in sublist]

                if len(combonets) > 1:
                    combo_network_partners = AsIs([combonet.id for combonet in combonets[1:]])
                else:
                    combo_network_partners = None

                if len(purenets) > 0:
                    network_partners = AsIs([purenet.id for purenet in purenets])
                else:
                    network_partners = None
                try:
                    net = MultiNetwork.get(MultiNetwork.combo_network          == combonets[0],
                                           MultiNetwork.combo_network_partners == combo_network_partners,
                                           MultiNetwork.network_partners       == network_partners,
                                           MultiNetwork.target_names           == AsIs(list(recipes.keys())),
                                           MultiNetwork.feature_names          == AsIs(nn.feature_names)
                    )
                except MultiNetwork.DoesNotExist:
                    net = MultiNetwork(combo_network          = combonets[0],
                                       combo_network_partners = combo_network_partners,
                                       network_partners       = network_partners,
                                       target_names           = AsIs(list(recipes.keys())),
                                       feature_names          = AsIs(nn.feature_names)
                    )
                    net.save()
                    print('Created MultiNetwork with id: {:d}'.format(net.id))
                else:
                    print('MultiNetwork with ComboNetworks {!s} already exists with id: {:d}'.format([combonet.id for combonet in combonets], net.id))

    @classmethod
    def calc_op(cls, column):
        query = (cls.select(
                            ComboNetwork,
                            ComboNetwork.id.alias('combo_id'),
                            fn.ARRAY_AGG(getattr(Hyperparameters, column), coerce=False).alias(column))
                 .join(Network, on=(Network.id == fn.ANY(ComboNetwork.networks)))
                 .join(Hyperparameters, on=(Network.id == Hyperparameters.network_id))
                 .group_by(cls.id)
        )
        return query


class Network(BaseModel):
    filter = ForeignKeyField(Filter, related_name='filter', null=True)
    train_script = ForeignKeyField(TrainScript, related_name='train_script')
    feature_prescale_bias = HStoreField()
    feature_prescale_factor = HStoreField()
    target_prescale_bias = HStoreField()
    target_prescale_factor = HStoreField()
    feature_names = ArrayField(TextField)
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_names = ArrayField(TextField)
    target_min = HStoreField()
    target_max = HStoreField()
    timestamp = DateTimeField(constraints=[SQL('DEFAULT now()')])

    @classmethod
    def find_partners_by_id(cls, network_id):
        q1 = Network.find_similar_topology_by_id(network_id, match_train_dim=False)
        q2 = Network.find_similar_networkpar_by_id(network_id, match_train_dim=False)
        return q1 & q2

    @classmethod
    def find_similar_topology_by_settings(cls, settings_path):
        with open(settings_path) as file_:
            json_dict = json.load(file_)
            cls.find_similar_topology_by_values(
                                                json_dict['hidden_neurons'],
                                                json_dict['hidden_activation'],
                                                json_dict['output_activation'],
                                                train_dim=json_dict['train_dim'])
        return query

    @classmethod
    def find_similar_topology_by_id(cls, network_id, match_train_dim=True):
        query = (Network
                 .select(
                         Hyperparameters.hidden_neurons,
                         Hyperparameters.hidden_activation,
                         Hyperparameters.output_activation)
                 .where(Network.id == network_id)
                 .join(Hyperparameters)
        )

        train_dim, = (Network
                 .select(
                         Network.target_names)
                 .where(Network.id == network_id)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None
        query = cls.find_similar_topology_by_values(*query.tuples().get(), train_dim=train_dim)
        query = query.where(Network.id != network_id)
        return query

    @classmethod
    def find_similar_topology_by_values(cls, hidden_neurons, hidden_activation, output_activation, train_dim=None):
        query = (Network.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.hidden_neurons ==
                        AsIs(hidden_neurons))
                 .where(Hyperparameters.hidden_activation ==
                        AsIs(hidden_activation))
                 .where(Hyperparameters.output_activation ==
                        AsIs(output_activation)))

        if train_dim is not None:
            query = query.where(Network.target_names ==
                        AsIs(train_dim))
        return query

    @classmethod
    def find_similar_networkpar_by_settings(cls, settings_path):
        with open(settings_path) as file_:
            json_dict = json.load(file_)

        query = cls.find_similar_networkpar_by_values(json_dict['train_dim'],
                                                    json_dict['goodness'],
                                                    json_dict['cost_l2_scale'],
                                                    json_dict['cost_l1_scale'],
                                                    json_dict['early_stop_measure'])
        return query

    @classmethod
    def find_similar_networkpar_by_id(cls, network_id, match_train_dim=True):
        query = (Network
                 .select(
                         Hyperparameters.goodness,
                         Hyperparameters.cost_l2_scale,
                         Hyperparameters.cost_l1_scale,
                         Hyperparameters.early_stop_measure)
                 .where(Network.id == network_id)
                 .join(Hyperparameters)
        )

        filter_id, train_dim = (Network
                 .select(Network.filter_id,
                         Network.target_names)
                 .where(Network.id == network_id)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None

        query = cls.find_similar_networkpar_by_values(*query.tuples().get(), filter_id=filter_id, train_dim=train_dim)
        query = query.where(Network.id != network_id)
        return query

    @classmethod
    def find_similar_networkpar_by_values(cls, goodness, cost_l2_scale, cost_l1_scale, early_stop_measure, filter_id=None, train_dim=None):
        # TODO: Add new hyperparameters here?
        query = (Network.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.goodness ==
                        goodness)
                 .where(Hyperparameters.cost_l2_scale.cast('numeric') ==
                        AsIs(cost_l2_scale))
                 .where(Hyperparameters.cost_l1_scale.cast('numeric') ==
                        AsIs(cost_l1_scale))
                 .where(Hyperparameters.early_stop_measure ==
                        early_stop_measure)
                 )
        if train_dim is not None:
            query = query.where(Network.target_names ==
                        AsIs(train_dim))

        if filter_id is not None:
                 query = query.where(Network.filter_id ==
                                     AsIs(filter_id))
        else:
            print('Warning! Not filtering on filter_id')
        return query

    #@classmethod
    #def find_similar_networkpar_by_settings(cls, settings_path):
    #    with open(settings_path) as file_:
    #        json_dict = json.load(file_)

    #    query = cls.find_similar_networkpar_by_values(json_dict['train_dim'],
    #                                                json_dict['goodness'],
    #                                                json_dict['cost_l2_scale'],
    #                                                json_dict['cost_l1_scale'],
    #                                                json_dict['early_stop_measure'])
    #    return query

    @classmethod
    def find_similar_trainingpar_by_id(cls, network_id):
        query = (Network
                 .select(Network.target_names,
                         Hyperparameters.minibatches,
                         Hyperparameters.optimizer,
                         Hyperparameters.standardization,
                         Hyperparameters.early_stop_after)
                 .where(Network.id == network_id)
                 .join(Hyperparameters)
        )

        filter_id = (Network
                 .select(Network.filter_id)
                 .where(Network.id == network_id)
                 ).tuples().get()[0]
        query = cls.find_similar_trainingpar_by_values(*query.tuples().get())
        query = query.where(Network.id != network_id)
        return query

    @classmethod
    def find_similar_trainingpar_by_values(cls, train_dim, minibatches, optimizer, standardization, early_stop_after):
        query = (Network.select()
                 .where(Network.target_names == AsIs(train_dim))
                 .join(Hyperparameters)
                 .where(Hyperparameters.minibatches == minibatches)
                 .where(Hyperparameters.optimizer == optimizer)
                 .where(Hyperparameters.standardization == standardization)
                 .where(Hyperparameters.early_stop_after == early_stop_after)
                 )
        return query


    @classmethod
    def from_folders(cls, pwd, **kwargs):
        for path_ in os.listdir(pwd):
            path_ = os.path.join(pwd, path_)
            if os.path.isdir(path_):
                try:
                    Network.from_folder(path_, **kwargs)
                except IOError:
                    print('Could not parse', path_, 'is training done?')

    @classmethod
    def from_folder(cls, pwd):
        with db.atomic() as txn:
            script_file = os.path.join(pwd, 'train_NDNN.py')
            #with open(script_file, 'r') as script:
            #    script = script.read()
            train_script = TrainScript.from_file(script_file)

            json_path = os.path.join(pwd, 'nn.json')
            nn = QuaLiKizNDNN.from_json(json_path)
            with open(json_path) as file_:
                json_dict = json.load(file_)
                dict_ = {}
                for name in ['feature_prescale_bias', 'feature_prescale_factor',
                             'target_prescale_bias', 'target_prescale_factor',
                             'feature_names', 'feature_min', 'feature_max',
                             'target_names', 'target_min', 'target_max']:
                    attr = getattr(nn, '_' + name)
                    if 'names' in name:
                        dict_[name] = list(attr)
                    else:
                        dict_[name] = {str(key): str(val) for key, val in attr.items()}

            dict_['train_script'] = train_script

            with open(os.path.join(pwd, 'settings.json')) as file_:
                settings = json.load(file_)

            dict_['filter_id'] = Filter.find_by_path_name(settings['dataset_path'])
            network = Network(**dict_)
            network.save()
            hyperpar = Hyperparameters.from_settings(network, settings)
            hyperpar.save()
            if settings['optimizer'] == 'lbfgs':
                optimizer = LbfgsOptimizer(hyperparameters=hyperpar,
                                           maxfun=settings['lbfgs_maxfun'],
                                           maxiter=settings['lbfgs_maxiter'],
                                           maxls=settings['lbfgs_maxls'])
            elif settings['optimizer'] == 'adam':
                optimizer = AdamOptimizer(hyperparameters=hyperpar,
                                          learning_rate=settings['learning_rate'],
                                          beta1=settings['adam_beta1'],
                                          beta2=settings['adam_beta2'])
            elif settings['optimizer'] == 'adadelta':
                optimizer = AdadeltaOptimizer(hyperparameters=hyperpar,
                                              learning_rate=settings['learning_rate'],
                                              rho=settings['adadelta_rho'])
            elif settings['optimizer'] == 'rmsprop':
                optimizer = RmspropOptimizer(hyperparameters=hyperpar,
                                              learning_rate=settings['learning_rate'],
                                              decay=settings['rmsprop_decay'],
                                              momentum=settings['rmsprop_momentum'])
            optimizer.save()

            activations = settings['hidden_activation'] + [settings['output_activation']]
            for ii, layer in enumerate(nn.layers):
                nwlayer = NetworkLayer(network = network,
                                       weights = np.float32(layer._weights).tolist(),
                                       biases = np.float32(layer._biases).tolist(),
                                       activation = activations[ii])
                nwlayer.save()

            NetworkMetadata.from_dict(json_dict['_metadata'], network)
            TrainMetadata.from_folder(pwd, network)

            network_json = NetworkJSON(network=network, network_json=json_dict, settings_json=settings)
            network_json.save()
            return network

    def to_QuaLiKizNDNN(self):
        json_dict = self.network_json.get().network_json
        nn = QuaLiKizNDNN(json_dict)
        return nn

    to_QuaLiKizNN = to_QuaLiKizNDNN

    def to_matlab(self):
        js = self.network_json.get().network_json
        newjs = {}
        for key, val in js.items():
            newjs[key.replace('/', '_').replace(':', '_')] = val
        io.savemat('nn' + str(self.id) + '.mat', newjs)

    def summarize(self):
        net = self.select().get()
        print({'target_names':     net.target_names,
               'rms_test':         net.network_metadata.get().rms_test,
               'rms_train':        net.network_metadata.get().rms_train,
               'rms_validation':   net.network_metadata.get().rms_validation,
               'epoch':            net.network_metadata.get().epoch,
               'train_time':       net.train_metadata.get().walltime[-1],
               'hidden_neurons':   net.hyperparameters.get().hidden_neurons,
               'standardization':  net.hyperparameters.get().standardization,
               'cost_l2_scale':    net.hyperparameters.get().cost_l2_scale,
               'early_stop_after': net.hyperparameters.get().early_stop_after}
        )

class MultiNetwork(BaseModel):
    network                     = ForeignKeyField(Network, related_name='pair_network', null=True)
    combo_network               = ForeignKeyField(ComboNetwork, related_name='pair_network', null=True)
    network_partners            = ArrayField(IntegerField, null=True)
    combo_network_partners      = ArrayField(IntegerField, null=True)
    target_names                = ArrayField(TextField)
    feature_names               = ArrayField(TextField)

    def to_QuaLiKizMultiNN(self):
        nns = []
        if self.combo_network is not None:
            nns.append(self.combo_network.to_QuaLiKizComboNN())
        if self.combo_network_partners is not None:
            for nn_id in self.combo_network_partners:
                nn = ComboNetwork.get_by_id(nn_id).to_QuaLiKizComboNN()
                nns.append(nn)
        if self.network is not None:
            nns.append(self.network.to_QuaLiKizNDNN())
        if self.network_partners is not None:
            for nn_id in self.network_partners:
                nn = Network.get_by_id(nn_id).to_QuaLiKizNDNN()
                nns.append(nn)

        return QuaLiKizMultiNN(nns)

    to_QuaLiKizNN = to_QuaLiKizMultiNN

    @classmethod
    def from_candidates(cls):
        #subquery = (Network.select(Network.id.alias('id'),
        #                           fn.unnest(Network.target_names).alias('unnested_tags'))
        #            .alias('subquery'))
        tags = ["div", "plus"]
        #tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
        #tags_filter = reduce(operator.or_, tags_filters)
        query = no_elements_in_list(Network, 'target_names', tags)
        query &= (Network.select()
                  .where(SQL("array_length(target_names, 1) = 1"))
                  .where(Network.target_names != AsIs(['efeETG_GB']))
                  )
        #query = (Network.select()
        #         .join(subquery, on=subquery.c.id == Network.id)
        #         .where(SQL("array_length(target_names, 1) = 1"))
        #         .where(~tags_filter)
        #         .where(Network.target_names != AsIs(['efeETG_GB']))
        #         # gets rid of duplicates
        #         .group_by(Network.id)
        #)
        #combo_query = (ComboNetwork.select())

        #for nn in chain(query, combo_query):
        for nn in query:
            splitted = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(nn.target_names[0])
            if splitted[1] == 'i':
                partner_target = ''.join(splitted[:1] + ['e'] + splitted[2:])
            else:
                print('Skipping, prefer to have ion network first')
                continue

            query = nn.__class__.find_partners_by_id(nn.id)
            query &= (nn.__class__.select()
                     .where(nn.__class__.target_names == AsIs([partner_target]))
                     )
            if query.count() == 0:
                print('No partners found for {!s}, id {!s}, target {!s}'.format(nn, nn.id, nn.target_names))
                continue
            elif query.count() > 1:
                print('Found {:d} matches for {!s}'.format(query.count(), partner_target))
                try:
                    sort = sorted([(el.postprocess.get().rms, el.id) for el in query])
                except Postprocess.DoesNotExist as ee:
                    net_id = re.search('PARAMS: \[(.*)\]', ee.args[0])[1]
                    table_field = re.search('WHERE \("t1"."(.*)"', ee.args[0])[1]
                    print('{!s} {!s} does not exist! Run postprocess.py'.format(table_field, net_id))
                    continue
                print('Selected {1:d} with RMS val {0:.2f}'.format(*sort[0]))
                query = (nn.__class__
                         .select()
                         .where(nn.__class__.id == sort[0][1])
                )
            partner = query.get()

            if isinstance(nn, Network):
                duplicate_check = (MultiNetwork.select()
                                   .where((MultiNetwork.network_id == partner.id)
                                    | (MultiNetwork.network_id == nn.id)
                                    | MultiNetwork.network_partners.contains(partner.id)
                                    | MultiNetwork.network_partners.contains(nn.id))
                                    )
                if duplicate_check.count() == 0:
                    net = cls(network=nn,
                              network_partners=[partner.id],
                              target_names=nn.target_names + partner.target_names,
                              feature_names=nn.feature_names
                    )
                    net.save()
                    print('Created MultiNetwork with id: {:d}'.format(net.id))
                else:
                    print('{!s}, id {!s} already in {!s}'.format(nn, nn.id, cls))

    @classmethod
    def calc_op(cls, column):
        subquery = ComboNetwork.calc_op(column).alias('sub1')
        query = (MultiNetwork.select(MultiNetwork.id.alias('multi_id'),
                                     fn.ARRAY_AGG(getattr(subquery.c, column), coerce=False).alias(column))
                 .join(subquery,
                       on=(cls.combo_network_id == subquery.c.combo_id) |
                       (subquery.c.combo_id == fn.ANY(cls.combo_network_partners))
                 ).alias('sub2')
                 .group_by(cls.id)
        )
        return query

            #else:
            #    duplicate_check = (MultiNetwork.select()
            #                       .where((MultiNetwork.combo_network_id == partner.id)
            #                        | (MultiNetwork.combo_network_id == nn.id)
            #                        | MultiNetwork.combo_network_partners.contains(partner.id)
            #                        | MultiNetwork.combo_network_partners.contains(nn.id))
            #                        )
            #    if duplicate_check.count() == 0:
            #        net = cls(combo_network=nn,
            #                  combo_network_partners=[partner.id],
            #                  target_names=nn.target_names + partner.target_names,
            #                  feature_names=nn.feature_names
            #        )
            #        net.save()
            #        print('Created MultiNetwork with id: {:d}'.format(net.id))
            #    else:
            #        print('{!s}, id {!s} already in {!s}'.format(nn, nn.id, cls))

class NetworkJSON(BaseModel):
    network = ForeignKeyField(Network, related_name='network_json')
    network_json = JSONField()
    settings_json = JSONField()

class NetworkLayer(BaseModel):
    network = ForeignKeyField(Network, related_name='network_layer')
    weights = ArrayField(FloatField)
    biases = ArrayField(FloatField)
    activation = TextField()

class NetworkMetadata(BaseModel):
    network = ForeignKeyField(Network, related_name='network_metadata')
    epoch = IntegerField()
    best_epoch = IntegerField()
    rms_test = FloatField(null=True)
    rms_train = FloatField(null=True)
    rms_validation = FloatField()
    rms_validation_descaled = FloatField(null=True)
    loss_test = FloatField(null=True)
    loss_train = FloatField(null=True)
    loss_validation = FloatField()
    metadata = HStoreField()

    @classmethod
    def from_dict(cls, json_dict, network):
        with db.atomic() as txn:
            stringified = {str(key): str(val) for key, val in json_dict.items()}
            try:
                rms_train = json_dict['rms_train']
                loss_train = json_dict['loss_train']
            except KeyError:
                loss_train = rms_train = None
            try:
                loss_test = json_dict['loss_test']
                rms_test = json_dict['loss_test']
            except KeyError:
                rms_test = loss_test = None
            try:
                rms_validation_descaled = json_dict['rms_validation_descaled']
            except KeyError:
                rms_validation_descaled = None
            network_metadata = NetworkMetadata(
                network=network,
                epoch=json_dict['epoch'],
                best_epoch=json_dict['best_epoch'],
                rms_train=rms_train,
                rms_validation=json_dict['rms_validation'],
                rms_validation_descaled=rms_validation_descaled,
                rms_test=rms_test,
                loss_train=loss_train,
                loss_validation=json_dict['loss_validation'],
                loss_test=loss_test,
                metadata=stringified
            )
            network_metadata.save()
            return network_metadata

class TrainMetadata(BaseModel):
    network = ForeignKeyField(Network, related_name='train_metadata')
    set = TextField(choices=['train', 'test', 'validation'])
    step =         ArrayField(IntegerField)
    epoch =        ArrayField(IntegerField)
    walltime =     ArrayField(FloatField)
    loss =         ArrayField(FloatField)
    mse =          ArrayField(FloatField)
    mabse =        ArrayField(FloatField, null=True)
    l1_norm =      ArrayField(FloatField, null=True)
    l2_norm =      ArrayField(FloatField, null=True)
    hostname = TextField()

    @classmethod
    def from_folder(cls, pwd, network):
        train_metadatas = None
        with db.atomic() as txn:
            for name in cls.set.choices:
                train_metadatas = []
                try:
                    with open(os.path.join(pwd, name + '_log.csv')) as file_:
                        df = pd.DataFrame.from_csv(file_)
                except IOError:
                    pass
                else:
                    try:
                        # TODO: Only works on debian-like
                        train_metadata = TrainMetadata(
                            network=network,
                            set=name,
                            step=[int(x) for x in df.index],
                            epoch=[int(x) for x in df['epoch']],
                            walltime=df['walltime'],
                            loss=df['loss'],
                            mse=df['mse'],
                            mabse=df['mabse'],
                            l1_norm=df['l1_norm'],
                            l2_norm=df['l2_norm'],
                            hostname=socket.gethostname()
                        )
                    except KeyError:
                        print('Legacy file.. Fallback')
                        # TODO: Only works on debian-like
                        train_metadata = TrainMetadata(
                            network=network,
                            set=name,
                            step=[int(x) for x in df.index],
                            epoch=[int(x) for x in df['epoch']],
                            walltime=df['walltime'],
                            loss=df['loss'],
                            mse=df['mse'],
                            hostname=socket.gethostname()
                        )
                    train_metadata.save()
                    train_metadatas.append(train_metadata)
        return train_metadatas


class Hyperparameters(BaseModel):
    network = ForeignKeyField(Network, related_name='hyperparameters')
    hidden_neurons = ArrayField(IntegerField)
    hidden_activation = ArrayField(TextField)
    output_activation = TextField()
    standardization = TextField()
    goodness = TextField()
    drop_chance = FloatField()
    optimizer = TextField()
    cost_l2_scale = FloatField()
    cost_l1_scale = FloatField()
    early_stop_after = FloatField()
    early_stop_measure = TextField()
    minibatches = IntegerField()
    drop_outlier_above = FloatField()
    drop_outlier_below = FloatField()
    validation_fraction = FloatField()
    dtype = TextField()

    @classmethod
    def from_settings(cls, network, settings):
        hyperpar = cls(network=network,
                       hidden_neurons=settings['hidden_neurons'],
                       hidden_activation=settings['hidden_activation'],
                       output_activation=settings['output_activation'],
                       standardization=settings['standardization'],
                       goodness=settings['goodness'],
                       drop_chance=settings['drop_chance'],
                       optimizer=settings['optimizer'],
                       cost_l2_scale=settings['cost_l2_scale'],
                       cost_l1_scale=settings['cost_l1_scale'],
                       early_stop_after=settings['early_stop_after'],
                       early_stop_measure=settings['early_stop_measure'],
                       minibatches=settings['minibatches'],
                       drop_outlier_above=settings['drop_outlier_above'],
                       drop_outlier_below=settings['drop_outlier_below'],
                       validation_fraction=settings['validation_fraction'],
                       dtype=settings['dtype'])
        return hyperpar


class LbfgsOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters, related_name='lbfgs_optimizer')
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

class AdamOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters, related_name='adam_optimizer')
    learning_rate = FloatField()
    beta1 = FloatField()
    beta2 = FloatField()

class AdadeltaOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters, related_name='adadelta_optimizer')
    learning_rate = FloatField()
    rho = FloatField()

class RmspropOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters, related_name='rmsprop_optimizer')
    learning_rate = FloatField()
    decay = FloatField()
    momentum = FloatField()

class Postprocess(BaseModel):
    network         = ForeignKeyField(Network, related_name='postprocess', null=True)
    combo_network   = ForeignKeyField(ComboNetwork, related_name='postprocess', null=True)
    multi_network   = ForeignKeyField(MultiNetwork, related_name='postprocess', null=True)
    filter          = ForeignKeyField(Filter, related_name='postprocess')
    rms             = FloatField()
    leq_bound       = FloatField()
    less_bound      = FloatField()

class PostprocessSlice(BaseModel):
    network = ForeignKeyField(Network, related_name='postprocess_slice', null=True)
    combo_network = ForeignKeyField(ComboNetwork, related_name='postprocess_slice', null=True)
    multi_network = ForeignKeyField(MultiNetwork, related_name='postprocess_slice', null=True)
    thresh_rel_mis_median       = ArrayField(FloatField)
    thresh_rel_mis_95width      = ArrayField(FloatField)
    no_thresh_frac              = ArrayField(FloatField)
    pop_abs_mis_median          = ArrayField(FloatField)
    pop_abs_mis_95width         = ArrayField(FloatField)
    no_pop_frac                 = ArrayField(FloatField)
    wobble_tot                  = ArrayField(FloatField)
    wobble_unstab               = ArrayField(FloatField)
    frac                        = FloatField()
    dual_thresh_mismatch_median = FloatField(null=True)
    dual_thresh_mismatch_95width= FloatField(null=True)
    no_dual_thresh_frac         = FloatField(null=True)

def create_schema():
    db.execute_sql('SET ROLE developer')
    db.execute_sql('CREATE SCHEMA develop AUTHORIZATION developer')

def create_tables():
    db.execute_sql('SET ROLE developer')
    db.create_tables([Filter, Network, NetworkJSON, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer, AdamOptimizer, AdadeltaOptimizer, RmspropOptimizer, TrainScript, PostprocessSlice, Postprocess, ComboNetwork])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()

def any_element_in_list(cls, column, tags):
    subquery = (cls.select(cls.id.alias('id'),
                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
                .alias('subquery'))
    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
    tags_filter = reduce(operator.or_, tags_filters)
    query = (cls.select()
             .join(subquery, on=subquery.c.id == cls.id)
             .where(tags_filter)
             # gets rid of duplicates
             .group_by(cls.id)
    )
    return query

def no_elements_in_list(cls, column, tags, fields=None):
    subquery = (cls.select(cls.id.alias('id'),
                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
                .alias('subquery'))
    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
    tags_filter = reduce(operator.or_, tags_filters)
    query = (cls.select(fields)
             .join(subquery, on=subquery.c.id == cls.id)
             .where(~tags_filter)
             # gets rid of duplicates
             .group_by(cls.id)
    )
    return query

def create_views():
    """
    CREATE VIEW
    SUMMARY AS
    SELECT A.id, target_names, hidden_neurons, standardization, cost_l2_scale, early_stop_after, best_rms_test, best_rms_validation, best_rms_train, final_rms_validation, final_rms_train, walltime, hostname FROM
    (
    SELECT network.id, network.target_names, hyperparameters.hidden_neurons, hyperparameters.standardization, hyperparameters.cost_l2_scale, hyperparameters.early_stop_after, networkmetadata.rms_test as best_rms_test, networkmetadata.rms_validation as best_rms_validation, networkmetadata.rms_train as best_rms_train
    FROM network
    INNER JOIN hyperparameters
    ON network.id = hyperparameters.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    ) A
    INNER JOIN
    (
    SELECT network.id AS id_B, sqrt(trainmetadata.mse[array_length(trainmetadata.mse, 1)]) as final_rms_validation
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    WHERE trainmetadata.set = 'validation'
    ) B
    ON A.id = B.id_B
    INNER JOIN
    (
    SELECT network.id AS id_C, sqrt(trainmetadata.mse[array_length(trainmetadata.mse, 1)]) as final_rms_train, trainmetadata.walltime[array_length(trainmetadata.walltime, 1)], trainmetadata.hostname
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    WHERE trainmetadata.set = 'train'
    ) C
    ON A.id = C.id_C
    """
    """
     DROP VIEW SUMMARY_LOSS;
CREATE VIEW
    SUMMARY_LOSS AS
    SELECT A.id, target_names, hidden_neurons, standardization, cost_l2_scale, early_stop_after, best_rms_test,  best_rms_validation, l2_norm_validation, walltime, hostname FROM
    (
    SELECT network.id, network.target_names, hyperparameters.hidden_neurons, hyperparameters.standardization, hyperparameters.cost_l2_scale, hyperparameters.early_stop_after, networkmetadata.rms_test as best_rms_test, networkmetadata.rms_validation as best_rms_validation
    FROM network
    INNER JOIN hyperparameters
    ON network.id = hyperparameters.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    WHERE hyperparameters.early_stop_measure = 'loss'
    ) A
    INNER JOIN
    (
    SELECT network.id AS id_C, trainmetadata.l2_norm[networkmetadata.best_epoch + 1] as l2_norm_validation, trainmetadata.walltime[array_length(trainmetadata.walltime, 1)], trainmetadata.hostname
    FROM network
    INNER JOIN trainmetadata
    ON network.id = trainmetadata.network_id
    INNER JOIN networkmetadata
    ON network.id = networkmetadata.network_id
    WHERE trainmetadata.set = 'validation'
    ) C
    ON A.id = C.id_C
"""

"""
Avg l2 multinetwork:
SELECT multinetwork.id as multi_id, multinetwork.target_names, AVG(cost_l2_scale) AS cost_l2_scale
FROM "multinetwork"
JOIN combonetwork ON (combo_network_id = combonetwork.id) OR (combonetwork.id = ANY (combo_network_partners))
JOIN network ON (network.id = ANY (combonetwork.networks))
JOIN hyperparameters ON (network.id = hyperparameters.network_id)
GROUP BY multinetwork.id
ORDER BY multi_id
"""

if __name__ == '__main__':
    from IPython import embed
    embed()
#purge_tables()
#create_tables()
#create_views()
#Network.from_folder('finished_nns_filter2/efiITG_GB_filter2', filter_id=3)
