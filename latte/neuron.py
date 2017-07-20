'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np

class Neuron:
    # Fields that store a value for each item in the batch (default behavior is
    # one value per neuron)
    batch_fields     = ["value", "grad", "inputs", "grad_inputs"]

    # A list of neuron fields that are initialized as 0, this improves
    # performance of initialization (latte can allocate the entire array of
    # zeros at once instead of getting the value of the field for each neuron)
    zero_init_fields = ["value", "grad"]

    def __init__(self):
        self.value = 0.0
        self.grad = 0.0

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def update_internal(self):
        raise NotImplementedError

class DataNeuron(Neuron):
    def forward(self):
        pass

    def backward(self):
        pass

class WeightedNeuron(Neuron):
    def __init__(self, weights, grad_weights):
        super().__init__()
        self.inputs = []
        self.grad_inputs = []

        self.weights = weights
        self.grad_weights = grad_weights

    def forward(self):
        for i in eachindex(self.inputs):
            self.value += self.inputs[i] * self.weights[i]

    def backward(self):
        for i in eachindex(self.inputs):
            self.grad_inputs[i] += self.grad * self.weights[i]

    def update_internal(self):
        for i in eachindex(self.inputs):
            self.grad_weights[i] += self.grad * self.inputs[i]


class BiasNeuron(Neuron):
    def __init__(self, bias, grad_bias):
        self.inputs = []
        self.grad_inputs = []

        self.bias = bias
        self.grad_bias = grad_bias

    def forward(self):
        self.value = self.input + self.bias[0]

    def backward(self):
        pass

    def update_internal(self):
        self.grad_bias[0] += self.grad

