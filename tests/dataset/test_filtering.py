import os

from unittest import TestCase, skip
from IPython import embed

from qlknn.dataset.filtering import *

test_files_dir = os.path.abspath(os.path.join(__file__, '../../gen2_test_files'))
class DataSetTestCase(TestCase):

    def setUp(self):
        dim = 4
        store_name = ''.join([str(dim), 'D_nions0_flat'])
        store_path = os.path.join(test_files_dir, store_name + '.h5')
        self.store = pd.HDFStore(store_path, 'r')
        self.input = self.store['/megarun1/input']
        self.data = self.store['/megarun1/flattened']
        self.const = self.store['/megarun1/constants']

        super(DataSetTestCase, self).setUp()

    def tearDown(self):
        self.store.close()
        super(DataSetTestCase, self).tearDown()

class TestFilters(DataSetTestCase):
    def test_ck_filter(self):
        ck_filter(self.data, 50)

    def test_totsep_filter(self):
        totsep_filter(self.data, 1.5)

    def test_ambipolar_filter(self):
        ambipolar_filter(self.data, 1.5)

    def test_femtoflux_filter(self):
        femtoflux_filter(self.data, 1e-4)

    def test_sanity_filter(self):
        data = sanity_filter(self.data, 50, 1.5, 1.5, 1e-4)

    def test_regime_filter(self):
        data = regime_filter(self.data, 0, 100)

    def test_stability_filter(self):
        stability_filter(self.data)

    def test_div_filter(self):
        div_filter(self.data)
