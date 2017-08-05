
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, rate, 
            test_data=None):
        n_test = len(test_data)
        n = len(training_data)
        ret = []
        for j in range(epochs):
            if (j > 10 and j % rate == 0):
                eta = eta / 2
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                d = self.evaluate(test_data)
                print("Epoch {0}: error = {1}, eta = {2}, j = {3}".format(
                   j, d[1], eta, j))
                ret.append(d[1])
           # else:
                #print("Epoch {0} complete".format(j))
        return ret

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # self.backprop(x[0:self.sizes[0]], x[self.sizes[0] : len(x)])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activation = np.reshape(activation, (len(activation), -1))
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)
            z = z + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        results = []
        error = []
        for x, y in test_data:
            r = self.feedforward(np.reshape(x, (len(x), -1))) 
            e = np.abs(r - np.reshape(y, (len(y), -1)))
            e = np.square(e)
            e = np.sum(e) / len(e)
            results.append(r)
            error.append(e)

        return [results, np.sum(error) / len(error)]

    def predict(self, data):
        results = []
        for x in test_data:
            r = self.feedforward(np.reshape(x, (len(x), -1))) 
            results.append(r)

        return results


    def cost_derivative(self, output_activations, y):
        return (output_activations - np.reshape(y, (len(y), -1)))

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))