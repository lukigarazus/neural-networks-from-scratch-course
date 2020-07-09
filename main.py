import random
import functools
import numpy as np

np.random.seed(0)


class Neuron:
    def __init__(self, weights, bias=0):
        self.bias = bias
        self.weights = 0.1 * np.random.randn(weights)

    def __hash__(self):
        return "Neuron " + str(self.bias) + " " + str(self.weights)

    def __str__(self):
        return "Neuron " + str(self.bias) + " " + str(self.weights)


class Layer:
    def __init__(self, neuron_count, neuron_weights):
        if neuron_count > 0:
            self.neurons = [Neuron(neuron_weights)
                            for i in range(0, neuron_count)]
            # for n in self.neurons:
            #     print(n)
        else:
            raise Exception("Layer needs at least one neuron")

    @property
    def weights(self):
        return [n.weights for n in self.neurons]

    @property
    def biases(self):
        return [n.bias for n in self.neurons]


class Network:
    def __init__(self, layer_neuron_counts):
        if len(layer_neuron_counts) == 0:
            raise Exception("Need at least one layer in a network")

        # Omit input layer
        self.layers = [Layer(lc, layer_neuron_counts[i])
                       for i, lc in enumerate(layer_neuron_counts[1:])]

    @property
    def weights(self):
        return [l.weights for l in self.layers]

    @property
    def biases(self):
        return [l.biases for l in self.layers]

    def fire_layer(self, inputs, layer):
        weights, biases = layer
        return np.dot(weights,
                      inputs) + biases

    def fire_layer_batch(self, inputs_batch, layer):
        weights, biases = layer
        return np.dot(inputs_batch, np.array(weights).T) + biases

    def fire(self, inputs):
        return functools.reduce(lambda acc, el: self.fire_layer(acc, el), zip(self.weights, self.biases), inputs)

    def fire_batch(self, inputs_batch):
        return functools.reduce(lambda acc, el: self.fire_layer_batch(acc, el), zip(self.weights, self.biases), inputs_batch)


net1 = Network([4, 5, 2])

X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

print(net1.fire_batch(X))
