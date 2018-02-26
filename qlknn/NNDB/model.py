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
                    ProgrammingError, AsIs, fn, SQL, DoesNotExist)
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from playhouse.hybrid import hybrid_property
#from playhouse.shortcuts import RetryOperationalError #peewee==2.10.1
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN

#class RetryPostgresqlExtDatabase(RetryOperationalError, PostgresqlExtDatabase):
#    pass
#db = RetryPostgresqlExtDatabase(database='nndb', host='gkdb.org')
db = PostgresqlExtDatabase(database='nndb', host='gkdb.org', register_hstore=True)
#db.execute_sql('CREATE SCHEMA IF NOT EXISTS develop;')
def flatten(l):
    return [item for sublist in l for item in sublist]

class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class TrainScript(BaseModel):
    script = TextField()
    version = TextField()

    @classmethod
    @db.atomic()
    def from_file(cls, pwd):
        with open(pwd, 'r') as script:
            script = script.read()

        train_script_query = TrainScript.select().where(TrainScript.script == script)
        if train_script_query.count() == 0:
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
    hypercube_script = TextField(null=True)
    description = TextField(null=True)
    min = FloatField(null=True)
    max = FloatField(null=True)
    remove_negative = BooleanField(null=True)
    remove_zeros = BooleanField(null=True)
    gam_filter = BooleanField(null=True)
    ck_max = FloatField(null=True)
    diffsep_max = FloatField(null=True)

    @classmethod
    @db.atomic()
    def from_file(cls, filter_file, hyper_file):
        with open(filter_file, 'r') as script:
            filter_script = script.read()

        with open(hyper_file, 'r') as script:
            hypercube_script = script.read()

        filter = Filter(script=filter_script, hypercube_script=hypercube_script)
        filter.save()

    @classmethod
    def find_by_path_name(cls, name):
        split = re.split('(?:(unstable)_|)(sane|test|training)_(?:gen(\d+)_|)(\d+)D_nions0_flat_filter(\d+).h5', name)
        try:
            if len(split) != 7:
                raise Exception
            filter_id = int(split[5])
        except:
            raise Exception('Could not find filter ID from name "{!s}"'.format(name))
        return filter_id


class Network(BaseModel):
    feature_names = ArrayField(TextField)
    target_names = ArrayField(TextField)
    recipe = TextField(null=True)
    networks = ArrayField(IntegerField, null=True)

    def get_recursive_hyperparameter(self, property):
        if self.networks is not None:
            prop_list = []
            for net_id in self.networks:
                net = Network.get_by_id(net_id)
                inner_prop_list = net.get_recursive_hyperparameter(property)
                prop_list.append(inner_prop_list)
        elif self.pure_network_params.count() == 1:
            query = (Hyperparameters.select(getattr(Hyperparameters, property))
                     .join(PureNetworkParams)
                     .join(Network)
                     .where(Network.id == self.id))
            value = query.tuples().get()[0]
            if isinstance(value, list):
                prop_list = np.array(value)
            else:
                prop_list = value
            #return [self.pure_network_params.get().hyperparametes.get().hidden_neurons]
        else:
            raise Exception
        return prop_list

    def flat_recursive_property(self, property):
        prop_list = self.get_recursive_hyperparameter(property)
        flat_list = prop_list
        while isinstance(prop_list, list) and (any([isinstance(el, list) for el in prop_list])):
            flat_list = []
            for ii in range(len(prop_list)):
                el = prop_list[ii]
                try:
                    if isinstance(el, list):
                        flattened = flatten(el)
                        flat_list.append(flattened)
                    else:
                        flat_list.extend([el])
                except TypeError:
                    try:
                        if isinstance(el, list):
                            flat_list.extend(el)
                        else:
                            flat_list.extend([el])
                    except TypeError:
                        flat_list.append(el)
            prop_list = flat_list
        if isinstance(prop_list, list):
            if prop_list[1:] == prop_list[:-1]:
                return prop_list[0]
            else:
                raise Exception('Unequal values for {!s}={!s}'.format(property, prop_list))
        else:
            return prop_list

    @hybrid_property
    def hidden_neurons(self):
        return self.flat_recursive_property('hidden_neurons')

    @hybrid_property
    def cost_l2_scale(self):
        return self.flat_recursive_property('cost_l2_scale')
    #@hybrid_property
    #def hidden_neurons(self):
    #    return [Network.get_by_id(nn).hyperparameters.get().hidden_neurons for nn in self.networks]

    #@hidden_neurons.expression
    #def hidden_neurons(cls):
    #    raise NotImplementedError('Cannot use in SQL query')

    #def to_QuaLiKizComboNN(self):
    #    network_ids = self.networks
    #    networks = [Network.get_by_id(num).to_QuaLiKizNDNN() for num in network_ids]
    #    recipe = self.recipe
    #    for ii in range(len(network_ids)):
    #        recipe = recipe.replace('nn' + str(ii), 'args[' + str(ii) + ']')
    #    exec('def combo_func(*args): return ' + recipe, globals())
    #    return QuaLiKizComboNN(self.target_names, networks, combo_func)

    #to_QuaLiKizNN = to_QuaLiKizComboNN

    @classmethod
    def find_divsum_candidates(cls):
        query = (cls
                 .select()
                 .where(cls.target_names[0] % '%_div_%')
                 .where(fn.array_length(cls.target_names, 1) == 1)
                 )
        for pure_network_params in query:
            try:
                cls.divsum_from_div_id(pure_network_params.id)
            except Exception:
                traceback.print_exc()

    @classmethod
    def divsum_from_div_id(cls, network_id, stop_on_missing=False):
        nn = cls.get_by_id(network_id)
        if len(nn.target_names) != 1:
            raise ValueError('Divsum network needs div network, not {!s}'.format(nn.target_names))
        target_name = nn.target_names[0]
        print('Trying to make combine Network {:d} with target {!s}'.format(nn.id, target_name))
        splitted = re.compile('(.*)_(div|plus)_(.*)').split(target_name)
        if len(splitted) != 5:
            raise ValueError('Could not split {!s} in divsum parts'.format(target_name))

        partner_target_sets = []
        formula_sets = []
        if splitted[2] == 'div':
            if splitted[1].startswith('efi') and splitted[3].startswith('efe'):
                # If it is efi / efe
                # Old style: efi / efe == nn0, efi + efe == nn1
                #partner_targets = [[splitted[1] + '_plus_' + splitted[3]]]
                #formulas = OrderedDict([(splitted[1], '(nn{0:d} * nn{1:d}) / (nn{0:d} + 1)'),
                #                        (splitted[3], '(nn{1:d}) / (nn{0:d} + 1)')])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: efi / efe == nn0, efe == nn1
                efe = splitted[3]
                efi = splitted[1]
                partner_targets = [[efe]]
                formulas = OrderedDict([
                    (efe, '(nn{0:d} * nn{1:d})'),
                    (efi, 'nn{1:d}')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efe') and splitted[3].startswith('efi'):
                # If it is efe / efi
                # Simple style: efe / efi == nn0, efi == nn1
                efe = splitted[1]
                efi = splitted[3]
                partner_targets = [[efi]]
                formulas = OrderedDict([
                    (efe, 'nn{1:d}'),
                    (efi, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efi'):
                # If it is pfe / efi
                pfe = splitted[1]
                efi = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                ## Triplet style: pfe / efi == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                #    (efe, 'nn{1:d} / (1 + nn{0:d} + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: pfe / efi == nn0, efi == nn1, efe / efi == nn2
                partner_targets = [[efi],
                                   [efe + '_div_' + efi]
                                   ]
                formulas = OrderedDict([
                    (efe, '(nn{1:d} * nn{2:d})'),
                    (efi, 'nn{1:d}'),
                    (pfe, '(nn{0:d} * nn{1:d})')
                ])
                partner_target_sets.append(partner_targets)
                formula_sets.append(formulas)
            elif splitted[1].startswith('efi') and splitted[3].startswith('pfe'):
                raise NotImplementedError('Should look at those again..')
                # If it is efi / pfe
                efi = splitted[1]
                pfe = splitted[3]
                split_efi = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efi)
                efe = ''.join(*[[split_efi[0]] + ['e'] + split_efi[2:]])
                ## Triplet style: efi / pfe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{0:d} * nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                #    (efe, '(nn{0:d} * nn{1:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})'),
                #    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} + nn{2:d} + nn{0:d} * nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                ## Heatflux style: efi / pfe == nn0, efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                #    (pfe, '(nn{1:d} * nn{2:d}) / (nn{0:d} * (1 + nn{2:d}))')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
            elif splitted[1].startswith('pfe') and splitted[3].startswith('efe'):
                # If it is pfe / efe
                pfe = splitted[1]
                efe = splitted[3]
                split_efe = re.compile('(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)').split(efe)
                efi = ''.join(*[[split_efe[0]] + ['i'] + split_efe[2:]])
                ## Triplet style: pfe / efe == nn0, pfe + efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[pfe + '_plus_' + efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{0:d} + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{1:d} + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d}) / (1 + nn{0:d} + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                ## Heatflux style: pfe / efe == nn0, efi + efe == nn1, efi / efe == nn2
                #partner_targets = [[efi + '_plus_' + efe],
                #                   [efi + '_div_' + efe]
                #                   ]
                #formulas = OrderedDict([
                #    (efi, '(nn{1:d} * nn{2:d}) / (1 + nn{2:d})'),
                #    (efe, '(nn{1:d}) / (1 + nn{2:d})'),
                #    (pfe, '(nn{0:d} * nn{1:d} * nn{2:d}) / (1 + nn{2:d})')
                #])
                #partner_target_sets.append(partner_targets)
                #formula_sets.append(formulas)
                # Simple style: pfe/efe == nn0, efe == nn1, efi / efe == nn2
                partner_targets = [[efe],
                                   [efi + '_div_' + efe]
                                   ]
                formulas = OrderedDict([
                    (efe, '(nn{1:d} * nn{2:d})'),
                    (efi, 'nn{1:d}'),
                    (pfe, '(nn{0:d} * nn{1:d})')
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
                query = PureNetworkParams.find_similar_topology_by_id(nn.pure_network_params.get().id, match_train_dim=False)
                query &= PureNetworkParams.find_similar_networkpar_by_id(nn.pure_network_params.get().id, match_train_dim=False)
                query &= (PureNetworkParams
                     .select()
                     .where(Network.target_names == partner_target)
                     .join(Network)
                     )
                if query.count() > 1:
                    print('Found {:d} matches for {!s}'.format(query.count(), partner_target))
                    try:
                        candidates = [(el.network.postprocess.get().rms, el.id) for el in query]
                    except Postprocess.DoesNotExist as ee:
                        net_id = re.search('Params: \[(.*)\]', ee.args[0])[1]
                        table_field = re.search('WHERE \("t1"."(.*)"', ee.args[0])[1]
                        raise Exception('{!s} {!s} does not exist! Run postprocess.py'.format(table_field, net_id))
                    sort = []
                    for rms, pure_id in candidates:
                        assert len(rms) == 1
                        sort.append([rms[0], pure_id])
                    sort = sorted(sort)
                    print('Selected {1:d} with RMS val {0:.2f}'.format(*sort[0]))
                    query = (PureNetworkParams
                             .select()
                             .where(PureNetworkParams.id == sort[0][1])
                    )
                elif query.count() == 0:
                    if stop_on_missing:
                        raise DoesNotExist('No {!s} with target {!s}!'.format(cls, partner_target))
                    print('No {!s} with target {!s}! Skipping..'.format(cls, partner_target))
                    skip = True

                if query.count() == 1:
                    purenet = query.get()
                    nns.append(purenet.network)
                    # Sanity check, something weird happening here..
                    if nns[-1].target_names != partner_target:
                        print('Insanety! Wrong partner found {!s} != {!s}'.format(nns[-1].target_names, partner_target))
                        embed()

            # TODO: change after PureNetworkParams split
            if skip is not True:
                recipes = OrderedDict()
                network_ids = [nn.id for nn in nns]
                for target, formula in formulas.items():
                    recipes[target] = formula.format(*list(range(len(nns))))

                nets = []
                for target, recipe in recipes.items():
                    if all([el not in recipe for el in ['+', '-', '/', '*']]):
                        net_num = int(recipe.replace('nn', ''))
                        net_id = network_ids[net_num]
                        nets.append(Network.get_by_id(net_id))
                    else:
                        query = (Network.select()
                                 .where((Network.recipe == recipe) &
                                        (Network.networks == network_ids))
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

                        nets.append(combonet)

                flatten = lambda l: [item for sublist in l for item in sublist]

                try:
                    net = cls.get(
                        cls.recipe == 'np.hstack(args)',
                        cls.networks == [net.id for net in nets],
                        cls.target_names == list(recipes.keys()),
                        cls.feature_names == nn.feature_names
                    )
                except Network.DoesNotExist:
                    net = cls.create(
                        recipe = 'np.hstack(args)',
                        networks = [net.id for net in nets],
                        target_names = list(recipes.keys()),
                        feature_names = nn.feature_names
                    )
                    print('Created MultiNetwork with id: {:d}'.format(net.id))
                else:
                    print('MultiNetwork with Networks {!s} already exists with id: {:d}'.format([net.id for net in nets], net.id))

    def to_QuaLiKizNDNN(self):
        return self.pure_network_params.get().to_QuaLiKizNDNN()

    def to_QuaLiKizComboNN(self):
        network_ids = self.networks
        networks = [Network.get_by_id(num).to_QuaLiKizNN() for num in network_ids]
        recipe = self.recipe
        for ii in range(len(network_ids)):
            recipe = recipe.replace('nn' + str(ii), 'args[' + str(ii) + ']')
        exec('def combo_func(*args): return ' + recipe, globals())
        return QuaLiKizComboNN(self.target_names, networks, combo_func)

    def to_QuaLiKizNN(self):
        if self.networks is None:
            net = self.to_QuaLiKizNDNN()
        else:
            net = self.to_QuaLiKizComboNN()
        return net

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


class PureNetworkParams(BaseModel):
    network = ForeignKeyField(Network, related_name='pure_network_params', unique=True)
    filter = ForeignKeyField(Filter, related_name='pure_network_params')
    train_script = ForeignKeyField(TrainScript, related_name='pure_network_params')
    feature_prescale_bias = HStoreField()
    feature_prescale_factor = HStoreField()
    target_prescale_bias = HStoreField()
    target_prescale_factor = HStoreField()
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_min = HStoreField()
    target_max = HStoreField()
    timestamp = DateTimeField(constraints=[SQL('DEFAULT now()')])

    def download_raw(self):
        root_dir = 'Network_' + str(self.network_id)
        os.mkdir(root_dir)
        network_json = self.network_json.get()
        with open(os.path.join(root_dir, 'settings.json'), 'w') as settings_file:
            json.dump(network_json.settings_json, settings_file,
                       sort_keys=True, indent=4)
        with open(os.path.join(root_dir, 'nn.json'), 'w') as network_file:
            json.dump(network_json.network_json, network_file,
                       sort_keys=True, indent=4)
        with open(os.path.join(root_dir, 'train_NDNN.py'), 'w') as train_file:
            train_file.writelines(self.train_script.get().script)

    @classmethod
    def find_partners_by_id(cls, pure_network_params_id):
        q1 = cls.find_similar_topology_by_id(pure_network_params_id, match_train_dim=False)
        q2 = cls.find_similar_networkpar_by_id(pure_network_params_id, match_train_dim=False)
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
    def find_similar_topology_by_id(cls, pure_network_params_id, match_train_dim=True):
        query = (cls
                 .select(
                         Hyperparameters.hidden_neurons,
                         Hyperparameters.hidden_activation,
                         Hyperparameters.output_activation)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
        )

        train_dim, = (cls
                 .select(
                         Network.target_names)
                 .where(cls.id == pure_network_params_id)
                 .join(Network)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None
        query = cls.find_similar_topology_by_values(*query.tuples().get(), train_dim=train_dim)
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_topology_by_values(cls, hidden_neurons, hidden_activation, output_activation, train_dim=None):
        query = (cls.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.hidden_neurons ==
                        hidden_neurons)
                 .where(Hyperparameters.hidden_activation ==
                        hidden_activation)
                 .where(Hyperparameters.output_activation ==
                        output_activation))

        if train_dim is not None:
            query = (query.where(Network.target_names == train_dim)
                     .switch(cls)
                     .join(Network))
        return query

    @classmethod
    def find_similar_networkpar_by_settings(cls, settings_path):
        with open(settings_path) as file_:
            json_dict = json.load(file_)

        query = cls.find_similar_networkpar_by_values(json_dict['train_dim'],
                                                    json_dict['goodness'],
                                                    json_dict['cost_l2_scale'],
                                                    json_dict['cost_l1_scale'],
                                                    json_dict['early_stop_after'],
                                                    json_dict['early_stop_measure'])
        return query

    @classmethod
    def find_similar_networkpar_by_id(cls, pure_network_params_id, match_train_dim=True):
        query = (cls
                 .select(
                         Hyperparameters.goodness,
                         Hyperparameters.cost_l2_scale,
                         Hyperparameters.cost_l1_scale,
                         Hyperparameters.early_stop_measure,
                         Hyperparameters.early_stop_after)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
        )

        filter_id, train_dim = (cls
                 .select(cls.filter_id,
                         Network.target_names)
                 .where(cls.id == pure_network_params_id)
                 .join(Network)
                 ).tuples().get()
        if match_train_dim is not True:
            train_dim = None

        query = cls.find_similar_networkpar_by_values(*query.tuples().get(), filter_id=filter_id, train_dim=train_dim)
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_networkpar_by_values(cls, goodness, cost_l2_scale, cost_l1_scale, early_stop_measure, early_stop_after, filter_id=None, train_dim=None):
        # TODO: Add new hyperparameters here?
        query = (cls.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.goodness ==
                        goodness)
                 .where(Hyperparameters.cost_l2_scale.cast('numeric') ==
                        cost_l2_scale)
                 .where(Hyperparameters.cost_l1_scale.cast('numeric') ==
                        cost_l1_scale)
                 .where(Hyperparameters.early_stop_measure ==
                        early_stop_measure)
                 .where(Hyperparameters.early_stop_after ==
                        early_stop_after)
                 )
        if train_dim is not None:
            query = (query.where(Network.target_names ==
                                 train_dim)
                     .switch(cls)
                     .join(Network)
            )

        if filter_id is not None:
            query = query.where(cls.filter_id ==
                                filter_id)
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
    def find_similar_trainingpar_by_id(cls, pure_network_params_id):
        query = (cls
                 .select(Network.target_names,
                         Hyperparameters.minibatches,
                         Hyperparameters.optimizer,
                         Hyperparameters.standardization,
                         Hyperparameters.early_stop_after)
                 .where(cls.id == pure_network_params_id)
                 .join(Hyperparameters)
                 .join(Network)
        )

        filter_id = (cls
                 .select(cls.filter_id)
                 .where(cls.id == cls.network_id)
                 ).tuples().get()[0]
        query = cls.find_similar_trainingpar_by_values(*query.tuples().get())
        query = query.where(cls.id != pure_network_params_id)
        return query

    @classmethod
    def find_similar_trainingpar_by_values(cls, train_dim, minibatches, optimizer, standardization, early_stop_after):
        query = (cls.select()
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
    @db.atomic()
    def from_folder(cls, pwd):
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
        net_dict = {'feature_names': dict_.pop('feature_names'),
                    'target_names': dict_.pop('target_names')}

        with open(os.path.join(pwd, 'settings.json')) as file_:
            settings = json.load(file_)

        dict_['filter_id'] = Filter.find_by_path_name(settings['dataset_path'])
        network = Network.create(**net_dict)
        dict_['network'] = network
        pure_network_params = PureNetworkParams.create(**dict_)
        pure_network_params.save()
        hyperpar = Hyperparameters.from_settings(pure_network_params, settings)
        hyperpar.save()
        if settings['optimizer'] == 'lbfgs':
            optimizer = LbfgsOptimizer(
                pure_network_params=pure_network_params,
                maxfun=settings['lbfgs_maxfun'],
                maxiter=settings['lbfgs_maxiter'],
                maxls=settings['lbfgs_maxls'])
        elif settings['optimizer'] == 'adam':
            optimizer = AdamOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                beta1=settings['adam_beta1'],
                beta2=settings['adam_beta2'])
        elif settings['optimizer'] == 'adadelta':
            optimizer = AdadeltaOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                rho=settings['adadelta_rho'])
        elif settings['optimizer'] == 'rmsprop':
            optimizer = RmspropOptimizer(
                pure_network_params=pure_network_params,
                learning_rate=settings['learning_rate'],
                decay=settings['rmsprop_decay'],
                momentum=settings['rmsprop_momentum'])
        optimizer.save()

        activations = settings['hidden_activation'] + [settings['output_activation']]
        for ii, layer in enumerate(nn.layers):
            nwlayer = NetworkLayer.create(
                pure_network_params=pure_network_params,
                weights = np.float32(layer._weights).tolist(),
                biases = np.float32(layer._biases).tolist(),
                activation = activations[ii])

        NetworkMetadata.from_dict(json_dict['_metadata'], pure_network_params)
        TrainMetadata.from_folder(pwd, pure_network_params)

        network_json = NetworkJSON.create(
            pure_network_params=pure_network_params,
            network_json=json_dict,
            settings_json=settings)
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


class NetworkJSON(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_json', unique=True)
    network_json = JSONField()
    settings_json = JSONField()

class NetworkLayer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_layer')
    weights = ArrayField(FloatField)
    biases = ArrayField(FloatField)
    activation = TextField()

class NetworkMetadata(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='network_metadata', unique=True)
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
    @db.atomic()
    def from_dict(cls, json_dict, pure_network_params):
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
            pure_network_params=pure_network_params,
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
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='train_metadata')
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
    @db.atomic()
    def from_folder(cls, pwd, pure_network_params):
        train_metadatas = None
        for name in cls.set.choices:
            train_metadatas = []
            try:
                with open(os.path.join(pwd, name + '_log.csv')) as file_:
                    df = pd.DataFrame.from_csv(file_)
            except IOError:
                pass
            else:
                # TODO: Only works on debian-like
                train_metadata = TrainMetadata(
                    pure_network_params=pure_network_params,
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
                train_metadata.save()
                train_metadatas.append(train_metadata)
        return train_metadatas


class Hyperparameters(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='hyperparameters', unique=True)
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
    def from_settings(cls, pure_network_params, settings):
        hyperpar = cls(pure_network_params=pure_network_params,
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
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='lbfgs_optimizer', unique=True)
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

class AdamOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='adam_optimizer', unique=True)
    learning_rate = FloatField()
    beta1 = FloatField()
    beta2 = FloatField()

class AdadeltaOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='adadelta_optimizer', unique=True)
    learning_rate = FloatField()
    rho = FloatField()

class RmspropOptimizer(BaseModel):
    pure_network_params = ForeignKeyField(PureNetworkParams, related_name='rmsprop_optimizer', unique=True)
    learning_rate = FloatField()
    decay = FloatField()
    momentum = FloatField()

class Postprocess(BaseModel):
    network         = ForeignKeyField(Network, related_name='postprocess')
    filter          = ForeignKeyField(Filter, related_name='postprocess')
    rms             = ArrayField(FloatField)
    leq_bound       = FloatField()
    less_bound      = FloatField()

class PostprocessSlice(BaseModel):
    network = ForeignKeyField(Network, related_name='postprocess_slice', null=True)
    thresh_rel_mis_median       = ArrayField(FloatField)
    thresh_rel_mis_95width      = ArrayField(FloatField)
    no_thresh_frac              = ArrayField(FloatField)
    pop_abs_mis_median          = ArrayField(FloatField)
    pop_abs_mis_95width         = ArrayField(FloatField)
    no_pop_frac                 = ArrayField(FloatField)
    wobble_tot                  = ArrayField(FloatField)
    wobble_unstab               = ArrayField(FloatField)
    wobble_qlkunstab            = ArrayField(FloatField)
    frac                        = FloatField()
    dual_thresh_mismatch_median = FloatField(null=True)
    dual_thresh_mismatch_95width= FloatField(null=True)
    no_dual_thresh_frac         = FloatField(null=True)

def create_schema():
    db.execute_sql('SET ROLE developer;')
    db.execute_sql('CREATE SCHEMA develop AUTHORIZATION developer;')
    db.execute_sql('ALTER DEFAULT PRIVILEGES IN SCHEMA develop GRANT ALL ON TABLES TO developer;')

def create_tables():
    db.execute_sql('SET ROLE developer;')
    db.create_tables([Filter, Network, PureNetworkParams, NetworkJSON, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer, AdamOptimizer, AdadeltaOptimizer, RmspropOptimizer, TrainScript, PostprocessSlice, Postprocess])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()

#def any_element_in_list(cls, column, tags):
#    subquery = (cls.select(cls.id.alias('id'),
#                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
#                .alias('subquery'))
#    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
#    tags_filter = reduce(operator.or_, tags_filters)
#    query = (cls.select()
#             .join(subquery, on=subquery.c.id == cls.id)
#             .where(tags_filter)
#             # gets rid of duplicates
#             .group_by(cls.id)
#    )
#    return query
#
#def no_elements_in_list(cls, column, tags, fields=None):
#    subquery = (cls.select(cls.id.alias('id'),
#                               fn.unnest(getattr(cls, column)).alias('unnested_tags'))
#                .alias('subquery'))
#    tags_filters = [subquery.c.unnested_tags.contains(tag) for tag in tags]
#    tags_filter = reduce(operator.or_, tags_filters)
#    if not fields:
#        fields = Network._meta.sorted_fields
#    query = (cls.select(fields)
#             .join(subquery, on=subquery.c.id == cls.id)
#             .where(~tags_filter)
#             # gets rid of duplicates
#             .group_by(cls.id)
#    )
#    return query

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
