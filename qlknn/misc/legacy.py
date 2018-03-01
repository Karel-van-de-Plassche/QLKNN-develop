
class Network(BaseModel):
    def get_recursive_hyperparameter(self, property):
        prop_list = []
        if self.networks is not None:
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
                prop_list.append(np.array(value))
            else:
                prop_list.append(value)
            #return [self.pure_network_params.get().hyperparametes.get().hidden_neurons]
        else:
            raise Exception
        return prop_list

class TestRecursiveAttributes(ModelTestCase):
    requires = require_lists['pure_network_params'] + [Hyperparameters, AdamOptimizer, LbfgsOptimizer, AdadeltaOptimizer, RmspropOptimizer, NetworkLayer, NetworkMetadata, TrainMetadata, NetworkJSON]
    @staticmethod
    def consume(iterable, break_strings=True, break_arrays=False):
        iterable = iter(iterable)

        while 1:
            try:
                item = next(iterable)
            except StopIteration:
                break

            if isinstance(item, str):
                if break_strings:
                    for char in item:
                        yield char
                else:
                    yield item
                continue

            if isinstance(item, np.ndarray):
                if not break_arrays:
                    yield item
                    continue

            try:
                data = iter(item)
                iterable = itertools.chain(data, iterable)
            except TypeError:
                yield item

    def flatten_recursive(iterable):
        return list(Network.consume(iterable))

    #def test_flatten_recursive(self):
    #    arr = np.array([128, 128, 128])

    #    nested = [arr]
    #    flat = Network.flatten_recursive(nested)
    #    self.assertNumpyArrayListEqual(flat, [arr])

    #    nested = [arr, arr]
    #    flat = Network.flatten_recursive(nested)
    #    self.assertNumpyArrayListEqual(flat, [arr, arr])

    #    nested = [[arr, arr], arr]
    #    flat = Network.flatten_recursive(nested)
    #    self.assertNumpyArrayListEqual(flat, [arr, arr, arr])

    #    nested = [[arr, arr, arr], arr, [arr, arr, arr]]
    #    flat = Network.flatten_recursive(nested)
    #    self.assertNumpyArrayListEqual(flat, [arr, arr, arr, arr, arr, arr, arr])

    #def test_get_recursive_pure_arrays(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'hidden_neurons'
    #    param = [np.array(getattr(hyperpar, param_name))]
    #    desired = param
    #    rec_param = self.net1.get_recursive_hyperparameter(param_name)
    #    self.assertNumpyArrayEqual(desired, rec_param)
    #    self.assertNumpyArrayEqual([np.array([128, 128, 128])], rec_param)

    #def test_get_recursive_combo_arrays(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'hidden_neurons'
    #    param = [np.array(getattr(hyperpar, param_name))]
    #    desired = [param, param]
    #    rec_param = self.combo_net.get_recursive_hyperparameter(param_name)
    #    self.assertNumpyArrayListEqual(desired, rec_param)
    #    arr = [np.array([128, 128, 128])]
    #    manual = [arr, arr]
    #    self.assertNumpyArrayListEqual(manual, rec_param)

    #def test_get_recursive_multi_arrays(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'hidden_neurons'
    #    param = [np.array(getattr(hyperpar, param_name))]
    #    desired = [[param] * 2, param]
    #    rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
    #    self.assertNumpyArrayListEqual(desired, rec_param)
    #    arr = [np.array([128, 128, 128])]
    #    manual = [[arr, arr], arr]
    #    self.assertNumpyArrayListEqual(manual, rec_param)

    #def test_get_recursive_multi_arrays_flipped(self):
    #    self.multi_net = Network.create(target_names=['efi_GB', 'efe_GB'],
    #                              feature_names=['Ati'],
    #                              filter=self.filter,
    #                              train_script=self.train_script,
    #                              networks=[self.net2.id, self.combo_net.id],
    #                              recipe='np.hstack(args)')
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'hidden_neurons'
    #    param = [np.array(getattr(hyperpar, param_name))]
    #    desired = [param, [param] * 2]
    #    rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
    #    self.assertNumpyArrayListEqual(desired, rec_param)
    #    arr = [np.array([128, 128, 128])]
    #    manual = [arr, [arr, arr]]
    #    self.assertNumpyArrayListEqual(manual, rec_param)

    #def test_get_recursive_pure_floats(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'cost_l2_scale'
    #    param = [np.array(getattr(hyperpar, param_name))]
    #    desired = param
    #    rec_param = self.net1.get_recursive_hyperparameter(param_name)
    #    self.assertEqual(desired, rec_param)
    #    self.assertEqual([8e-6], rec_param)

    #def test_get_recursive_combo_floats(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'cost_l2_scale'
    #    param = [getattr(hyperpar, param_name)]
    #    desired = [param] * 2
    #    rec_param = self.combo_net.get_recursive_hyperparameter(param_name)
    #    self.assertSequenceEqual(desired, rec_param)
    #    self.assertSequenceEqual([[8.e-6], [8e-6]], rec_param)

    #def test_get_recursive_multi_floats(self):
    #    hyperpar = self.net1.pure_network_params.get().hyperparameters.get()
    #    param_name = 'cost_l2_scale'
    #    param = [getattr(hyperpar, param_name)]
    #    desired = [[param] * 2, param]
    #    rec_param = self.multi_net.get_recursive_hyperparameter(param_name)
    #    self.assertSequenceEqual(desired, rec_param)
    #    self.assertSequenceEqual([[[8.e-6], [8e-6]], [8e-6]], rec_param)
