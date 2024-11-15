import numpy as np
import math
from numpy.random import default_rng

input_values = np.array([1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y = np.array([1.0, 0.0, 0.0, 0.0])

class Neural_Network:
    def __init__(self, definition_array, activation_function):
        self.definition = definition_array
        self.activation_function = activation_function
        self.layers = list()
        layer_number = len(definition_array)
        for i in range(len(definition_array)):
            try:
                prev_layer = self.layers[i-1]
            except IndexError:
                prev_layer = "null"
            try:
                prev_layer_Neuron_count = self.layers[i-1].Neuron_count
            except IndexError:
                prev_layer_Neuron_count = 0
            self.layers.append(Layer(prev_layer, definition_array[i], prev_layer_Neuron_count, layer_number, self.activation_function, self))
            layer_number -= 1

    def feedforward(self):
        for i in self.layers:
            print("layer_number:", i.layer_number, " prev_layer_number:", i.prev_Layer, " Neuron_count:", i.Neuron_count, " prev_Neuron_count:", i.prev_Layer_Neuron_count)
            i.feedforward()
            print(i.activation_values)
        self.output_values = np.array(softmax(self.layers[len(self.layers) - 1].activation_values))
        print(self.output_values * 100)

    def backpropagate(self):
        for i in reversed(self.layers):
            print("layer_number:", i.layer_number, " prev_layer_number:", i.prev_Layer, " Neuron_count:", i.Neuron_count, " prev_Neuron_count:", i.prev_Layer_Neuron_count)
            i.backpropagate(y)
            print(i.weights)

class Layer:
    def __init__(self, prev_Layer, Neuron_count, prev_Layer_Neuron_count, layer_number, activation_function, nn):
        self.nn = nn
        self.prev_Layer = prev_Layer
        self.Neuron_count = Neuron_count
        self.layer_number = layer_number
        self.activation_function = activation_function
        self.prev_Layer_Neuron_count = prev_Layer_Neuron_count
        self.inputs = np.zeros(self.prev_Layer_Neuron_count)
        self.weights = default_rng(42).random((self.prev_Layer_Neuron_count, self.Neuron_count))
        self.biases = np.zeros(self.Neuron_count)
        self.activation_values = np.zeros(self.Neuron_count)
        if self.prev_Layer == "null":
            self.activation_values = input_values
    def __str__(self):
        return f"{self.layer_number}"
    def feedforward(self):
        if self.prev_Layer != "null":
            for i in range(self.Neuron_count):
                self.inputs = self.prev_Layer.activation_values
                self.neuron_weights = self.weights[: , i]
                if(self.activation_function == "sigmoid"): self.activation_values[i] = sigmoid(np.sum(self.inputs * self.neuron_weights) + self.biases[i])
                elif(self.activation_function == "relu"): self.activation_values[i] = relu(np.sum(self.inputs * self.neuron_weights) + self.biases[i])
                elif(self.activation_function == "tanh"): self.activation_values[i] = tanh(np.sum(self.inputs * self.neuron_weights) + self.biases[i])
        else:
            self.activation_values = input_values
    def backpropagate(self, y):
        if(self.layer_number == 1):
            for i in range(self.Neuron_count):
                output_error = (self.activation_values[i] * (1 - self.activation_values[i])) * (y[i] - self.activation_values[i])
                for v in range(self.prev_Layer_Neuron_count):
                    print("i:", i, " v:", v)
                    self.weights[v,i] = np.add(self.weights[v, i], 1 * (output_error) * self.inputs[v])
        else:
            if(self.prev_layer != "null"):
                for i in range(self.Neuron_count):
                    hidden_error = (self.activation_values[i] * (1 - self.activation_values[i])) * (self.weights) 
                    for i in range(self.nn.layers[i+1].Neuron_count):
                        



def sigmoid(x):
    return(1/(1+math.e**(x*-1)))

def relu(x):
    if x < 0: x = 0
    return(x)

def tanh(x):
    return(math.tanh(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def deriv_sigmoid(x):
    x = sigmoid(x)
    return(x * (1 - x))
        
NeN = Neural_Network([8,6,6,4], "sigmoid")

NeN.feedforward()

NeN.backpropagate()