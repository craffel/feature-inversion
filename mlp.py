import numpy as np
import theano.tensor as T
import theano


def relu(x):
    '''
    Rectifier function.
    '''
    return T.maximum(0.0, x)


class Layer(object):
    def __init__(self, n_input, n_output, W=None, b=None, activation=relu):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a
        nonlinearity and x is the input vector.

        Input:
            n_input - number of input nodes
            n_output - number of output nodes
            W - Mixing matrix, default None which means initialize randomly
            b - Bias vector, default None which means initialize to ones
            activation - nonlinear activation function, default relu
        '''
        # Randomly initialize W
        if W is None:
            # Tanh is best initialized to values between +/- sqrt(6/(n_nodes))
            W_values = np.asarray(np.random.uniform(-np.sqrt(6./(n_input +
                                                                 n_output)),
                                                    np.sqrt(6./(n_input +
                                                                n_output)),
                                                    (n_output, n_input)),
                                  dtype=theano.config.floatX)
            # Sigmoid activation uses +/- 4*sqrt(6/(n_nodes))
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            # Create theano shared variable for W
            W = theano.shared(value=W_values, name='W', borrow=True)
        # Initialize b to zeros
        if b is None:
            b = theano.shared(value=np.ones((n_output, 1),
                                            dtype=theano.config.floatX),
                              name='b',
                              borrow=True,
                              broadcastable=(False, True))

        self.W = W
        self.b = b
        self.activation = activation

        # Easy-to-access parameter list
        self.params = [self.W, self.b]

    def output(self, x):
        '''
        Compute this layer's output given an input

        Input:
            x - Theano symbolic variable for layer input
        Output:
            output - Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        return (lin_output if self.activation is None else
                self.activation(lin_output))


class MLP(object):
    def __init__(self, layer_sizes=None, Ws=None, bs=None, activations=None):
        '''
        Multi-layer perceptron

        Input:
            layer_sizes - List-like of layer sizes, len n_layers + 1, includes
            input and output dimensionality
                Default None, which means retrieve layer sizes from W
            Ws - List-like of weight matrices, len n_layers, where Ws[n] is
            layer_sizes[n + 1] x layer_sizes[n]
                Default None, which means initialize randomly
            bs - List-like of biases, len n_layers, where bs[n] is
            layer_sizes[n + 1]
                Default None, which means initialize randomly
            activations - List of length n_layers of activation function for
            each layer
                Default None, which means all layers are tanh
        '''
        # Check that we received layer sizes or weight matrices + bias vectors
        if layer_sizes is None and Ws is None:
            raise ValueError('Either layer_sizes or Ws must not be None')

        # Initialize lists of layers
        self.layers = []

        # Populate layer sizes if none was provided
        if layer_sizes is None:
            layer_sizes = []
            # Each layer size is the input size of each mixing matrix
            for W in Ws:
                layer_sizes.append(W.shape[1])
            # plus the output size of the last layer
            layer_sizes.append(Ws[-1].shape[0])

        # Make a list of Nones if Ws and bs are None
        if Ws is None:
            Ws = [None]*(len(layer_sizes) - 1)
        if bs is None:
            bs = [None]*(len(layer_sizes) - 1)

        # All activations are tanh if none was provided
        if activations is None:
            activations = [T.tanh]*(len(layer_sizes) - 1)

        # Construct the layers
        for n, (n_input, n_output,
                W, b, activation) in enumerate(zip(layer_sizes[:-1],
                                                   layer_sizes[1:], Ws, bs,
                                                   activations)):
            self.layers.append(Layer(n_input, n_output, W, b,
                                     activation=activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        '''
        Compute the MLP's output given an input

        Input:
            x - Theano symbolic variable for MLP input
        Output:
            output - x passed through the net
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x


def gradient_updates(cost, params, learning_rate):
    '''
    Compute updates for gradient descent over some parameters.

    Input:
        cost - Theano cost function to minimize
        params - Parameters to compute gradient against
        learning_rate - GD learning rate
    Output:
        updates - list of updates, per-parameter
    '''
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        updates.append((param, param - learning_rate*T.grad(cost, param)))
    return updates


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    Input:
        cost - Theano cost function to minimize
        params - Parameters to compute gradient against
        learning_rate - GD learning rate
        momentum - GD momentum
    Output:
        updates - list of updates, per-parameter
    '''
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        mparam = theano.shared(param.get_value()*0.,
                               broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*mparam))
        updates.append((mparam,
                        mparam*momentum + (1. - momentum)*T.grad(cost, param)))
    return updates
