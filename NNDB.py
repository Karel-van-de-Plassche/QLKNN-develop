from peewee import *
from peewee import FloatField, FloatField, ProgrammingError, IntegerField
import numpy as np
import inspect
import sys
from playhouse.postgres_ext import PostgresqlExtDatabase, ArrayField, BinaryJSONField, JSONField, HStoreField
from IPython import embed
import scipy as sc
from scipy import io
import os
from run_model import QuaLiKizNDNN
import json

db = PostgresqlExtDatabase(database='nndb', host='gkdb.com')
class BaseModel(Model):
    """A base model that will use our Postgresql database"""
    class Meta:
        database = db
        schema = 'develop'

class Filter(BaseModel):
    script = TextField()
    description = TextField()

class Network(BaseModel):
    filter = ForeignKeyField(Filter, related_name='filter', null=True)
    prescale_bias = HStoreField()
    prescale_factor = HStoreField()
    feature_names = ArrayField()
    feature_min = HStoreField()
    feature_max = HStoreField()
    target_names = ArrayField(TextField)
    target_min = HStoreField()
    target_max = HStoreField()
    weights = ArrayField()
    biases = ArrayField()
    activation = TextField()
    json = JSONField()

    @classmethod
    def from_folder(cls, pwd):
        json_path = os.path.join(pwd, 'nn.json')
        nn = QuaLiKizNDNN.from_json(json_path)
        with open(json_path) as file_:
            json_dict = json.load(file_)
        dict_ = {'json': json_dict}
        for name in ['prescale_bias', 'prescale_factor',
                     'feature_names', 'feature_min', 'feature_max',
                     'target_names', 'target_min', 'target_max']:
            dict_[name] = getattr(nn, name)

        weights = []
        biases = []
        for layer in nn.layers:
            weights.append(layer.weight)
            biases.append(layer.bias)
        dict_['weights'] = weights
        dict_['biases'] = biases

        network = Network(**dict_)

        embed()

class NetworkLayer(BaseModel):
    network = ForeignKeyField(Network)
    weights = ArrayField()
    biases = ArrayField()


class NetworkMetadata(BaseModel):
    network = ForeignKeyField(Network)
    metadata =  HStoreField()

class TrainMetadata(BaseModel):
    network = ForeignKeyField(Network)
    set = TextField(choices=['train', 'test', 'validation'])
    step = IntegerField()
    epoch = IntegerField()
    walltime = FloatField()
    loss = FloatField()
    mse = FloatField()

class Hyperparameters(BaseModel):
    hidden_neurons = ArrayField(IntegerField)
    scaling = TextField()
    cost_l2_scale = FloatField()
    early_stop_after = FloatField()

class LbfgsOptimizer(BaseModel):
    hyperparameters = ForeignKeyField(Hyperparameters)
    maxfun = IntegerField()
    maxiter = IntegerField()
    maxls = IntegerField()

Network.from_folder('nns/efeITG_GB_3')
