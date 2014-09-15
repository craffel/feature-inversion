import numpy as np
import theano.tensor as T
import theano
import load_data
import glob
import librosa
import pretty_midi
import random
import nntools

# Where do the training .mid files live
train_glob = glob.glob('data/midi_train/*/*.mid')
# Where do the validate .mid files live
validate_glob = glob.glob('data/midi_validate/*/*.mid')
# SGD learning rate and momentum
learning_rate = .000001
momentum = .99
# How often should we generate example output?
check_frequency = 100
# Training batch size
batch_size = 100

# Create generater objects over the datasets
train_generator = load_data.midi_stft_generator(train_glob,
                                                batch_size=batch_size,
                                                n_queue=100000)

# Load in example data point
X, Y = train_generator.next()

# Hidden layer size = nextpow2(input dimensionality)
hidden_size = 2**np.ceil(np.log2(X.shape[0]))
# Construct neural network, 2 hidden layers with dropout
relu = nntools.nonlinearities.rectify
l_in = nntools.layers.InputLayer(num_features=X.shape[0],
                                 batch_size=batch_size)
l_hidden1 = nntools.layers.DenseLayer(l_in, num_units=hidden_size,
                                      nonlinearity=relu)
l_hidden1_dropout = nntools.layers.DropoutLayer(l_hidden1, p=0.5)
l_hidden2 = nntools.layers.DenseLayer(l_hidden1_dropout,
                                      num_units=hidden_size,
                                      nonlinearity=relu)
l_hidden2_dropout = nntools.layers.DropoutLayer(l_hidden2, p=0.5)
l_out = nntools.layers.DenseLayer(l_hidden2_dropout, num_units=Y.shape[0],
                                  nonlinearity=relu)

# Symbolic variables for network input and target output
input = T.matrix('input')
target_output = T.matrix('target_output')

# Squared error cost function
error = T.sum((l_out.get_output(input) - target_output)**2)
all_params = nntools.layers.get_all_params(l_out)
updates = nntools.updates.nesterov_momentum(error,
                                            all_params,
                                            learning_rate)
# Function for optimizing the neural net parameters, by minimizing cost
train = theano.function([input, target_output],
                        error,
                        updates=updates)
# Function for computing the network output
network_output = theano.function([input],
                                 l_out.get_output(input, deterministic=True))
# Compute cost without training
cost = theano.function([input, target_output], error)

try:
    for n, (X, Y) in enumerate(train_generator):
        train_cost = train(X.T, Y.T)
        # Write out a wav file from the validation set
        if n and (not n % check_frequency):
            midi = pretty_midi.PrettyMIDI(random.choice(validate_glob))
            X, Y = load_data.midi_to_features(midi)
            print "Iteration {}, cost {}".format(n, cost(X.T, Y.T))
            Y_pred = network_output(X.T).T
            y_pred = librosa.istft(load_data.flatten_mag_phase(Y_pred))
            librosa.output.write_wav('{}.wav'.format(n), y_pred, 8000)
except KeyboardInterrupt:
    print "Ran {} iterations, final training cost {}".format(n, train_cost)
