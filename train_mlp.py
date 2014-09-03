import numpy as np
import theano.tensor as T
import theano
import load_data
import mlp
import glob
import librosa
import pretty_midi
import random

# Where do the training .mid files live
train_glob = glob.glob('data/midi_train/*/*.mid')
# Where do the validate .mid files live
validate_glob = glob.glob('data/midi_validate/*/*.mid')
# How many hidden layers?
n_hidden = 3
# SGD learning rate and momentum
initial_learning_rate = .01
momentum = .99
# How often should we generate example output?
check_frequency = 1000

# Create generater objects over the datasets
train_generator = load_data.midi_stft_generator(train_glob, batch_size=1)

# Load in example data point
X, Y = train_generator.next()

# Create Theano symbolic functions/variables
input = T.matrix('input')
target_output = T.matrix('target_output')
# Set the hidden layer size to nextpow2(input_size)
hidden_size = 2**np.ceil(np.log2(X.shape[0]))
layer_sizes = [X.shape[0]] + [hidden_size]*n_hidden + [Y.shape[0]]

# Create multi-layer perceptron
inverter = mlp.MLP(layer_sizes)

# Squared error cost function
error = T.sum((inverter.output(input) - target_output)**2)

iteration = T.scalar('iteration')
learning_rate = initial_learning_rate/(iteration + 1)

# Function for optimizing the neural net parameters, by minimizing cost
train = theano.function([input, target_output, iteration],
                        error,
                        updates=mlp.gradient_updates_momentum(error,
                                                              inverter.params,
                                                              learning_rate,
                                                              momentum))
# Function for computing the network output
network_output = theano.function([input], inverter.output(input))
# Compute cost without training
cost = theano.function([input, target_output], error)

try:
    for n, (X, Y) in enumerate(train_generator):
        train_cost = train(X, Y, n)
        # Write out a wav file from the validation set
        if n and (not n % check_frequency):
            midi = pretty_midi.PrettyMIDI(random.choice(validate_glob))
            X, Y = load_data.midi_to_features(midi)
            print "Iteration {}, cost {}".format(n, cost(X, Y))
            Y_pred = network_output(X)
            y_pred = librosa.istft(load_data.flatten_mag_phase(Y_pred))
            librosa.output.write_wav('{}.wav'.format(n), y_pred, 8000)
except KeyboardInterrupt:
    print "Ran {} iterations, final training cost {}".format(n, train_cost)
