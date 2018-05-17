import pandas as pd
import numpy as np
from tensorflow.python.ops.random_ops import random_shuffle

try:
    profile
except NameError:
    from qlknn.misc.tools import profile

class Dataset():
    def __init__(self, features, target):
        from IPython import embed
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if not isinstance(target, np.ndarray):
            target = np.array(target)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data = np.hstack([features, target])
        self._num_features = features.shape[1]
        self._num_examples = self._data.shape[0]

    @property
    def _features(self):
        return self._data[:, :self._num_features]

    @property
    def _target(self):
        return self._data[:, self._num_features:]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if batch_size == -1:
            batch_size = self._num_examples
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                #from IPython import embed
                #embed()
                #print(self._data[:10, :])
                perm = np.arange(self._num_examples)
                #print('C', self._data.flags['C_CONTIGUOUS'])
                #print('F', self._data.flags['F_CONTIGUOUS'])
                np.random.shuffle(perm)
                #self._data = self._data[np.random.permutation(self._num_examples), :]
                self._data = np.take(self._data, perm, axis=0)
                #print(self._data[:10, :])
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, \
                'Batch size asked bigger than number of samples'
        end = self._index_in_epoch
        batch = (self._data[start:end, :self._num_features], self._data[start:end, self._num_features:])
        return batch

    def to_hdf(self, file, key):
        with pd.HDFStore(file) as store:
            store.put(key + '/features', self._features)
            store.put(key + '/target', self._target)

    @classmethod
    def read_hdf(cls, file, key):
        with pd.HDFStore(file) as store:
            dataset = Dataset(store.get(key + '/features'),
                              store.get(key + '/target'))
        return dataset

    def astype(self, dtype):
        self._features = self._features.astype(dtype)
        self._target = self._target.astype(dtype)
        return self


class Datasets():
    _fields = ['train', 'validation', 'test']

    def __init__(self, **kwargs):
        for name in self._fields:
            setattr(self, name, kwargs.pop(name))
        assert ~bool(kwargs)

    def to_hdf(self, file):
        for name in self._fields:
            getattr(self, name).to_hdf(file, name)

    @classmethod
    def read_hdf(cls, file):
        datasets = {}
        for name in cls._fields:
            datasets[name] = Dataset.read_hdf(file, name)
        return Datasets(**datasets)

    def astype(self, dtype):
        for name in self._fields:
            setattr(self, name, getattr(self, name).astype(dtype))
        return self

@profile
def convert_panda(data_df, feature_names, target_names, frac_validation, frac_test, shuffle=True):
    #data_df = pd.concat(input_df, target_df, axis=1)
    total_size = len(data_df)
    # Dataset might be ordered. Shuffle to be sure
    if shuffle:
        data_df = shuffle_panda(data_df)
    validation_size = int(frac_validation * total_size)
    test_size = int(frac_test * total_size)
    train_size = total_size - validation_size - test_size

    datasets = []
    for slice_ in [data_df.iloc[:train_size],
                   data_df.iloc[train_size:train_size + validation_size],
                   data_df.iloc[train_size + validation_size:]]:
        datasets.append(Dataset(slice_[feature_names],
                                slice_[target_names]))

    return Datasets(train=datasets[0],
                    validation=datasets[1],
                    test=datasets[2])

def split_panda(panda, frac=0.1):
    panda1 = panda.sample(frac=frac)
    panda2_i = panda.index ^ panda1.index
    panda2 = panda.loc[panda2_i]
    return (panda1, panda2)


def shuffle_panda(panda):
    return panda.iloc[np.random.permutation(np.arange(len(panda)))]
