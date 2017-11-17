from peewee import *
from peewee import (FloatField, FloatField, ProgrammingError, IntegerField, BooleanField,
                    Param, Passthrough)
import numpy as np
import inspect
import sys
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from playhouse.shortcuts import RetryOperationalError
from IPython import embed
from warnings import warn
import os
networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), '../../networks'))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN, QuaLiKizComboNN
import json
import pandas as pd
import subprocess
import socket
import re

class RetryPostgresqlExtDatabase(RetryOperationalError, PostgresqlExtDatabase):
    pass
db = RetryPostgresqlExtDatabase(database='nndb', host='gkdb.org')

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
                sp_result = subprocess.run('git rev-parse HEAD',
                                           stdout=subprocess.PIPE,
                                           shell=True,
                                           check=True)
                version = sp_result.stdout.decode('UTF-8').strip()
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
        split = re.split('filtered_(\d)D_nions0_flat_filter(\d).h5', name)
        try:
            if len(split) != 4:
                raise
            filter_id = int(split[2])
        except:
            raise Exception('Could not find filter ID from name "{!s}"'.format(name))
        return filter_id

class ComboNetwork(BaseModel):
    target_name = ArrayField(TextField)
    recipe = TextField()

    def to_QuaLiKizComboNN(self):
        target_name = self.target_name.get()
        recipe = self.recipe.get()
        #94 div
        #85 sum
        embed()

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
        splitted = re.compile('(.*)_(div|plus)_(.*)').split(target_name)
        if len(splitted) != 5:
            raise Exception('Could not split {!s} in divsum parts'.format(target_name))

        if splitted[2] == 'div':
            partner_op = '_plus_'
        else:
            raise Exception('Divsum network needs div network, not {!s}'.format(nn.target_names))

        query = Network.find_similar_topology_by_id(network_id, match_train_dim=False)
        query &= Network.find_similar_networkpar_by_id(network_id, match_train_dim=False)
        query &= (Network
                 .select()
                 .where(Network.target_names == Param([splitted[1] + partner_op + splitted[3]]))
                 )
        if query.count() != 1:
            raise Exception('Found {:d} matches.'.format(query.count()))

        nn_sum = query.get()
        target_1 = splitted[1]
        recipe_target_1 = '({{{1:d}}} * {{{0:d}}})/ ({{{1:d}}} + 1)'.format(nn_sum.id, nn.id)
        target_2 = splitted[3]
        recipe_target_2 = '{{{0:d}}} / ({{{1:d}}} + 1)'.format(nn_sum.id, nn.id)

        for target, recipe_target in [(target_1, recipe_target_1), (target_2, recipe_target_2)]:
            if ComboNetwork.select().where(ComboNetwork.recipe == recipe_target).count() == 0:
                ComboNetwork(target_name=[target], recipe=recipe_target).save()
            else:
                print('Network with recipe {!s} already exists! Skipping!'.format(recipe_target_1))


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
    def find_partner_by_id(cls, network_id):
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
                        Param(hidden_neurons))
                 .where(Hyperparameters.hidden_activation ==
                        Param(hidden_activation))
                 .where(Hyperparameters.output_activation ==
                        Param(output_activation)))

        if train_dim is not None:
            query = query.where(Network.target_names ==
                        Param(train_dim))
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
        query = (Network.select()
                 .join(Hyperparameters)
                 .where(Hyperparameters.goodness ==
                        goodness)
                 .where(Hyperparameters.cost_l2_scale ==
                        Passthrough(str(cost_l2_scale)))
                 .where(Hyperparameters.cost_l1_scale ==
                        Passthrough(str(cost_l1_scale)))
                 .where(Hyperparameters.early_stop_measure ==
                        early_stop_measure)
                 )
        if train_dim is not None:
            query = query.where(Network.target_names ==
                        Param(train_dim))

        if filter_id is not None:
                 query = query.where(Network.filter_id ==
                                     Param(filter_id))
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
                 .where(Network.target_names == Param(train_dim))
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
                except FileNotFoundError:
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
            try:
                hyperpar = Hyperparameters(network=network,
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
                                           minibatches=settings['minibatches']
                )
            except KeyError:
                print('Legacy file.. Fallback')
                hyperpar = Hyperparameters(network=network,
                                           hidden_neurons=settings['hidden_neurons'],
                                           hidden_activation=settings['hidden_activation'],
                                           output_activation=settings['output_activation'],
                                           standardization=settings['standardization'],
                                           goodness=settings['goodness'],
                                           optimizer=settings['optimizer'],
                                           cost_l2_scale=settings['cost_l2_scale'],
                                           cost_l1_scale=settings['cost_l1_scale'],
                                           early_stop_after=settings['early_stop_after'],
                )
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
    rms_test = FloatField()
    rms_train = FloatField(null=True)
    rms_validation = FloatField()
    loss_test = FloatField()
    loss_train = FloatField(null=True)
    loss_validation = FloatField()
    metadata = HStoreField()

    @classmethod
    def from_dict(cls, json_dict, network):
        with db.atomic() as txn:
            stringified = {str(key): str(val) for key, val in json_dict.items()}
            try:
                rms_train = json_dict['rms_train']
            except KeyError:
                rms_train = None
            try:
                loss_train = json_dict['loss_train']
            except KeyError:
                loss_train = None
            try:
                loss_validation = json_dict['loss_validation']
                loss_test = json_dict['loss_test']
            except KeyError:
                loss_validation = loss_test = None
            network_metadata = NetworkMetadata(
                network=network,
                epoch=json_dict['epoch'],
                best_epoch=json_dict['best_epoch'],
                rms_test=json_dict['rms_test'],
                rms_train=rms_train,
                rms_validation=json_dict['rms_validation'],
                loss_test=loss_test,
                loss_train=loss_train,
                loss_validation=loss_validation,
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
                except FileNotFoundError:
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

class Postprocessing(BaseModel):
    network = ForeignKeyField(Network, related_name='postprocessing')
    filtered_rms = FloatField()
    rel_filtered_rms = FloatField()
    l2_norm = FloatField()
    filtered_loss = FloatField()
    filtered_real_loss = FloatField()
    filtered_real_loss_function = TextField()

class PostprocessSlice(BaseModel):
    network = ForeignKeyField(Network, related_name='postprocess_slice')
    thresh_rel_mis_median       = ArrayField(FloatField)
    thresh_rel_mis_95width      = ArrayField(FloatField)
    no_thresh_frac              = ArrayField(FloatField)
    pop_abs_mis_median          = ArrayField(FloatField)
    pop_abs_mis_95width         = ArrayField(FloatField)
    no_pop_frac                 = ArrayField(FloatField)
    dual_thresh_mismatch_median = FloatField(null=True)
    no_dual_thresh_frac         = FloatField(null=True)

def create_schema():
    db.execute_sql('SET ROLE developer')
    db.execute_sql('CREATE SCHEMA develop AUTHORIZATION developer')

def create_tables():
    db.execute_sql('SET ROLE developer')
    db.create_tables([Filter, Network, NetworkJSON, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer, AdamOptimizer, AdadeltaOptimizer, RmspropOptimizer, TrainScript, PostprocessSlice, Postprocessing, ComboNetwork])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()

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



if __name__ == '__main__':
    from IPython import embed
    embed()
#purge_tables()
#create_tables()
#create_views()
#Network.from_folder('finished_nns_filter2/efiITG_GB_filter2', filter_id=3)
