from peewee import *
from peewee import FloatField, FloatField, ProgrammingError, IntegerField, BooleanField
import numpy as np
import inspect
import sys
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from IPython import embed
import os
from run_model import QuaLiKizNDNN
import json
import pandas as pd
import subprocess
import socket

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
        with db.atomic() as txn:
            sp_result = subprocess.run('git rev-parse HEAD',
                                       stdout=subprocess.PIPE,
                                       shell=True,
                                       check=True)
            version = sp_result.stdout.decode('UTF-8').strip()
            with open(pwd, 'r') as script:
                script = script.read()

            train_script = TrainScript(
                script=script,
                version=version
            )
            train_script.save()
            return train_script

class Filter(BaseModel):
    script = TextField()
    description = TextField(null=True)
    min = FloatField(null=True)
    max = FloatField(null=True)
    remove_negative = BooleanField(null=True)
    remove_zeros = BooleanField(null=True)
    gam_filter = BooleanField(null=True)

    @classmethod
    def from_file(cls, pwd):
        with db.atomic() as txn:
            with open(pwd, 'r') as script:
                script = script.read()
            filter = Filter(script=script)
            filter.save()


class Network(BaseModel):
    filter = ForeignKeyField(Filter, related_name='filter', null=True)
    train_script = ForeignKeyField(TrainScript, related_name='train_script')
    prescale_bias = HStoreField()
    prescale_factor = HStoreField()
    feature_names = ArrayField(TextField)
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_names = ArrayField(TextField)
    target_min = HStoreField()
    target_max = HStoreField()

    @classmethod
    def from_folders(cls, pwd, **kwargs):
        for path_ in os.listdir(pwd):
            path_ = os.path.join(pwd, path_)
            if os.path.isdir(path_):
                Network.from_folder(path_, **kwargs)

    @classmethod
    def from_folder(cls, pwd, filter_id=None):
        with db.atomic() as txn:
            script_file = os.path.join(pwd, 'train_NDNN.py')
            with open(script_file, 'r') as script:
                script = script.read()
            train_script_query = TrainScript.select().where(TrainScript.script == script)
            if train_script_query.count() == 0:
                train_script = TrainScript.from_file(script_file)
            elif train_script_query.count() == 1:
                train_script = train_script_query.get()
            else:
                raise Exception('multiple train scripts found. Could not choose')

            json_path = os.path.join(pwd, 'nn.json')
            nn = QuaLiKizNDNN.from_json(json_path)
            with open(json_path) as file_:
                json_dict = json.load(file_)
                dict_ = {}
                for name in ['prescale_bias', 'prescale_factor',
                             'feature_names', 'feature_min', 'feature_max',
                             'target_names', 'target_min', 'target_max']:
                    attr = getattr(nn, name)
                    if 'names' in name:
                        dict_[name] = list(attr)
                    else:
                        dict_[name] = {str(key): str(val) for key, val in attr.items()}

        dict_['train_script'] = train_script
        dict_['filter_id'] = filter_id
        network = Network(**dict_)
        network.save()

        with open(os.path.join(pwd, 'settings.json')) as file_:
            settings = json.load(file_)
            hyperpar = Hyperparameters(network=network,
                                       hidden_neurons=settings['hidden_neurons'],
                                       standardization=settings['standardization'],
                                       cost_l2_scale=settings['cost_l2_scale'],
                                       early_stop_after=settings['early_stop_after'])
            hyperpar.save()
            if settings['optimizer'] == 'lbfgs':
                optimizer = LbfgsOptimizer(hyperparameters=hyperpar,
                                           maxfun=settings['lbfgs_maxfun'],
                                           maxiter=settings['lbfgs_maxiter'],
                                           maxls=settings['lbfgs_maxls'])
                optimizer.save()
            if settings['optimizer'] == 'adam':
                optimizer = AdamOptimizer(hyperparameters=hyperpar,
                                          learning_rate=settings['learning_rate'],
                                          beta1=settings['adam_beta1'],
                                          beta2=settings['adam_beta2'])
                optimizer.save()

        activations = settings['hidden_activation'] + [settings['output_activation']]
        for ii, layer in enumerate(nn.layers):
            nwlayer = NetworkLayer(network = network,
                                   weights = layer.weight.tolist(),
                                   biases = layer.bias.tolist(),
                                   activation = activations[ii])
            nwlayer.save()

        NetworkMetadata.from_dict(json_dict['_metadata'], network)
        TrainMetadata.from_folder(pwd, network)

        network_json = NetworkJSON(network=network, network_json=json_dict, settings_json=settings)
        network_json.save()
        return network

class NetworkJSON(BaseModel):
    network = ForeignKeyField(Network)
    network_json = JSONField()
    settings_json = JSONField()

class NetworkLayer(BaseModel):
    network = ForeignKeyField(Network)
    weights = ArrayField(FloatField)
    biases = ArrayField(FloatField)
    activation = TextField()

class NetworkMetadata(BaseModel):
    network = ForeignKeyField(Network)
    nn_develop_version = TextField()
    epoch = IntegerField()
    best_epoch = IntegerField()
    rms_test = FloatField()
    rms_train = FloatField()
    rms_validation = FloatField()
    metadata = HStoreField()

    @classmethod
    def from_dict(cls, json_dict, network):
        with db.atomic() as txn:
            stringified = {str(key): str(val) for key, val in json_dict.items()}
            network_metadata = NetworkMetadata(
                network=network,
                nn_develop_version=json_dict['nn_develop_version'],
                epoch=json_dict['epoch'],
                best_epoch=json_dict['best_epoch'],
                rms_test=json_dict['rms_test'],
                rms_train=json_dict['rms_train'],
                rms_validation=json_dict['rms_validation'],
                metadata=stringified
            )
            network_metadata.save()
            return network_metadata

class TrainMetadata(BaseModel):
    network = ForeignKeyField(Network)
    set = TextField(choices=['train', 'test', 'validation'])
    step =         ArrayField(IntegerField)
    epoch =        ArrayField(IntegerField)
    walltime =     ArrayField(FloatField)
    loss =         ArrayField(FloatField)
    mse =          ArrayField(FloatField)
    mabse =        ArrayField(FloatField)
    l1_loss =      ArrayField(FloatField)
    l2_loss =      ArrayField(FloatField)
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
                    train_metadata = TrainMetadata(
                        network=network,
                        set=name,
                        step=[int(x) for x in df.index],
                        epoch=[int(x) for x in df['epoch']],
                        walltime=df['walltime'],
                        loss=df['loss'],
                        mse=df['mse'],
                        mabse=df['mabse'],
                        l1_loss=df['l1_loss'],
                        l2_loss=df['l2_loss'],
                        hostname=socket.gethostname()
                    )
                    # TODO: Only works on debian-like
                    train_metadata.save()
                    train_metadatas.append(train_metadata)
        return train_metadatas


class Hyperparameters(BaseModel):
    network = ForeignKeyField(Network, related_name='hyperparameters')
    hidden_neurons = ArrayField(IntegerField)
    standardization = TextField()
    cost_l2_scale = FloatField()
    early_stop_after = FloatField()

class LbfgsOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters)
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

class AdamOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters)
    learning_rate = FloatField()
    beta1 = FloatField()
    beta2 = FloatField()

def create_tables():
    db.execute_sql('SET ROLE developer')
    db.create_tables([Filter, Network, NetworkJSON, NetworkLayer, NetworkMetadata, TrainMetadata, Hyperparameters, LbfgsOptimizer, AdamOptimizer, TrainScript])

def purge_tables():
    clsmembers = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__)
    for name, cls in clsmembers:
        if name != BaseModel:
            try:
                db.drop_table(cls, cascade=True)
            except ProgrammingError:
                db.rollback()


#purge_tables()
#create_tables()
#Network.from_folder('finished_nns_filter2/efiITG_GB_filter2', filter_id=3)
