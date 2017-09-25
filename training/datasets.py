import pandas as pd
import numpy as np
class Dataset():
    def __init__(self, features, target):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._features = features
        self._target = target
        assert self._features.shape[0] == self._target.shape[0]
        self._num_examples = features.shape[0]

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
            # TODO: Use panda_shuffle function
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self._features.iloc[perm]
                self._target = self._target.iloc[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, \
                'Batch size asked bigger than number of samples'
        end = self._index_in_epoch
        batch = (self._features.iloc[start:end], self._target.iloc[start:end])
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

def convert_panda(features_df, targets_df, frac_validation, frac_test, shuffle=True):
    panda = pd.concat([features_df, targets_df], axis=1)
    feature_names = features_df.columns
    target_names = targets_df.columns
    total_size = features_df.shape[0]
    # Dataset might be ordered. Shuffle to be sure
    if shuffle:
        panda = shuffle_panda(panda)
    validation_size = int(frac_validation * total_size)
    test_size = int(frac_test * total_size)
    train_size = total_size - validation_size - test_size

    datasets = []
    for slice_ in [panda.iloc[:train_size],
                   panda.iloc[train_size:train_size + validation_size],
                   panda.iloc[train_size + validation_size:]]:
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
