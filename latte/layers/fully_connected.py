import numpy as np
from ..neuron import WeightedNeuron, BiasNeuron
from ..ensemble import Ensemble, EnsembleGroup
import latte.core

class FCNeuron(WeightedNeuron):
    pass
    # def forward(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.value += self.inputs[i] * self.weights[i]

    # def backward(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.grad_inputs[i] += self.grad * self.weights[i]

    # def update_internal(self):
    #     for i in range_dim(self.inputs, 0):
    #         self.grad_weights[i] += self.grad * self.inputs[i]

def FullyConnectedLayer(net, input_ensemble, num_outputs):
    fan_in = len(input_ensemble)
    scale = np.sqrt(3.0 / fan_in)
    weights = np.random.rand(num_outputs, *input_ensemble.shape).astype(np.float32) * (2 * scale) - scale
    weights_grad = np.zeros_like(weights)

    neurons = np.array([FCNeuron(weights[i], weights_grad[i]) for i in range(num_outputs)])

    ens = net.init_ensemble(neurons)

    input_shape = input_ensemble.shape
    flattened = np.prod(input_shape)
    mapping_range = tuple(range(d) for d in input_ensemble.shape)

    def mapping(x):
        return mapping_range

    net.add_connections(input_ensemble, ens, mapping)

    bias = np.zeros((num_outputs, 1), dtype=np.float32)
    grad_bias = np.zeros_like(bias)

    bias_neurons = np.array([BiasNeuron(bias[i], grad_bias[i]) for i in range(num_outputs)])

    bias_ens = net.init_activation_ensemble(bias_neurons, ens)

    input_ensemble.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
    if "value" in input_ensemble.tiling_info:
        input_ensemble.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
        input_ensemble.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
        ens.tile('inputs', dim=0, factor=latte.config.SIMDWIDTH)
        ens.tile('grad_inputs', dim=0, factor=latte.config.SIMDWIDTH)
    ens.privatize('grad_weights')
    bias_ens.privatize('grad_bias')

    ens.tile('weights', dim=0, factor=latte.config.SIMDWIDTH)
    ens.tile('grad_weights', dim=0, factor=latte.config.SIMDWIDTH)
    if "value" in input_ensemble.tiling_info:
        ens.tile('weights', dim=1, factor=latte.config.SIMDWIDTH)
        ens.tile('grad_weights', dim=1, factor=latte.config.SIMDWIDTH)
    bias_ens.tile('bias', dim=0, factor=latte.config.SIMDWIDTH)
    bias_ens.tile('grad_bias', dim=0, factor=latte.config.SIMDWIDTH)
    ens.tile('value', dim=0, factor=latte.config.SIMDWIDTH)
    ens.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)

    ens.vectorize(direction="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
    factor = 8
    while net.batch_size % factor != 0:
        factor -= 1
    ens.unroll(direction="forward", loop_var="_neuron_index_0", factor=factor)
    ens.parallelize(direction="forward", loop_var="_neuron_index_0")
    ens.parallelize(direction="forward", loop_var="_neuron_index_1_outer")
    ens.parallelize(direction="backward", loop_var="_neuron_index_0")
    ens.parallelize(direction="backward", loop_var="__unique_loopvar0_outer")

    # ens.vectorize(direction="backward", loop_var="__unique_loopvar0", factor=latte.config.SIMDWIDTH)
    # ens.unroll(direction="backward", loop_var="_neuron_index_0", factor=8)
    # ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1", "__unique_loopvar0"))
    # ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "_neuron_index_1"))
    for i in range(1, input_ensemble.ndim):
        ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "__unique_loopvar{}".format(i)))
    if "value" in input_ensemble.tiling_info:
        ens.vectorize(direction="backward", loop_var="__unique_loopvar0_inner", factor=latte.config.SIMDWIDTH)
    ens.parallelize(direction="backward", loop_var="__unique_loopvar0")
    factor = 8
    while net.batch_size % factor != 0:
        factor -= 1
    ens.unroll(direction="backward", loop_var="_neuron_index_0", factor=factor)
    # bias_ens.vectorize(direction="forward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
    # bias_ens.parallelize(direction="forward", loop_var="_neuron_index_1")
    # bias_ens.unroll(direction="forward", loop_var="_neuron_index_0", factor=8)

    # ens.vectorize(direction="backward", loop_var="_neuron_index_1_inner", factor=latte.config.SIMDWIDTH)
    # if not "grad" in input_ensemble.tiling_info:
    #     input_ensemble.tile('grad', dim=0, factor=latte.config.SIMDWIDTH)
    # ens.vectorize(direction="backward", loop_var="__unique_loopvar0_inner", factor=latte.config.SIMDWIDTH)
    # ens.unroll(direction="backward", loop_var="_neuron_index_1", factor=4)
    # ens.parallelize(direction="backward", loop_var="__unique_loopvar0")
    # for i in range(1, input_ensemble.ndim):
    #     ens.swap_loops(direction="backward", loop_vars=("_neuron_index_1_inner", "__unique_loopvar{}".format(i)))
    # bias_ens.parallelize(direction="backward", loop_var="_neuron_index_1")
    # bias_ens.unroll(direction="backward", loop_var="_neuron_index_0", factor=8)
    bias_ens.parallelize(direction="forward", loop_var="_neuron_index_0")
    bias_ens.parallelize(direction="forward", loop_var="_neuron_index_1_outer")
    bias_ens.parallelize(direction="backward", loop_var="_neuron_index_0")
    bias_ens.parallelize(direction="backward", loop_var="__unique_loopvar0_outer")

    return EnsembleGroup(ens, bias_ens)
