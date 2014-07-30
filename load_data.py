'''
Functions for loading in data
'''

import multiprocessing as mp
import time
import Queue
import numpy as np
import librosa
import pretty_midi
import random


def midi_to_stft(midi, fs=8000, **kwargs):
    '''
    Convert MIDI data to an STFT of its synthesized audio data.
    kwargs will be passed to librosa.stft

    :parameters:
        - midi : pretty_midi.PrettyMIDI
            MIDI data object
        - fs : int
            Sampling rate, default 22050

    :returns:
        - stft : np.ndarray
            STFT matrix
    '''
    audio_data = midi.synthesize(fs=fs)
    return librosa.stft(audio_data, n_fft=1024, **kwargs)


def midi_to_piano_roll(midi, hop_seconds=1024/4/8000., min_note=20,
                       max_note=100):
    '''
    Converts MIDI data into a piano roll flattened across instruments.
    Drum instruments and special effects are ignored.

    :parameters:
        - midi : pretty_midi.PrettyMIDI
            MIDI data object
        - hop_seconds : float
            Time between each column in the piano roll matrix
        - min_note : int
            Lowest note in the piano roll to consider.
        - max_note : int
            Highest note in the piano roll to consider.

    :returns:
        - piano_roll : np.ndarray
            Piano roll representation
    '''
    # Return the flattened piano roll at the specified hop rate over the
    # specified note range
    return midi.get_piano_roll(fs=1./hop_seconds)[min_note:max_note, :]


def midi_to_stacked_piano_roll(midi, hop_seconds=1024/4/8000., min_note=20,
                               max_note=100):
    '''
    Converts MIDI data into matrix of stacked piano rolls, one for each
    instrument class.
    Drum instruments and special effects are ignored.

    :parameters:
        - midi : pretty_midi.PrettyMIDI
            MIDI data object
        - hop_seconds : float
            Time between each column in the piano roll matrix
        - min_note : int
            Lowest note in the piano roll to consider.
        - max_note : int
            Highest note in the piano roll to consider.

    :returns:
        - stacked_piano_roll : np.ndarray
            Stacked piano roll representation
    '''
    n_notes = max_note - min_note
    # Start index in the stacked piano roll of each instrument class
    stacked_index = {'Piano' : 0, 'Chromatic Percussion' : n_notes,
                     'Organ' : 2*n_notes, 'Guitar' : 3*n_notes,
                     'Bass' : 4*n_notes, 'Strings' : 5*n_notes,
                     'Ensemble' : 6*n_notes, 'Brass' : 7*n_notes,
                     'Reed' : 8*n_notes, 'Pipe' : 9*n_notes,
                     'Synth Lead' : 10*n_notes, 'Synth Pad' : 11*n_notes,
                     'Ethnic' : 12*n_notes, 'Percussive' : 13*n_notes}
    # Initialize stacked piano roll
    times = np.arange(0, midi.get_end_time(), hop_seconds)
    stacked_piano_roll = np.zeros((14*n_notes, times.shape[0]))
    # This will map program number to the stacked piano roll
    for instrument in midi.instruments:
        ins_class = pretty_midi.program_to_instrument_class(instrument.program)
        # Skip drum and effects instruments
        if instrument.is_drum or \
           'Effects' in ins_class:
            continue
        # Get the piano roll for this instrument
        piano_roll = instrument.get_piano_roll(fs=1./hop_seconds)
        # Determine row and column indices to add in piano roll
        index = stacked_index[ins_class]
        note_range = np.r_[index:index + n_notes]
        n_col = piano_roll.shape[1]
        stacked_piano_roll[note_range, :n_col] += piano_roll[min_note:max_note]
    return stacked_piano_roll


def split_mag_phase(complex_matrix):
    '''
    Given a complex-valued matrix, create a new matrix with twice the number of
    rows where the absolute value and angle of the complex values are in either
    half.

    :parameters:
        - complex_matrix : np.ndarray, dtype=np.complex
            Complex-valued matrix

    :returns:
        - split_matrix : np.ndarray, dtype=np.float
            Matrix with magnitude and phase split into two parts
            shape=(2*complex_matrix.shape[0], complex_matrix.shape[1])
    '''
    return np.vstack([np.abs(complex_matrix), np.angle(complex_matrix)])


def shingle(x, stacks):
    ''' Shingles a matrix column-wise

    :parameters:
        - x : np.ndarray
            Matrix to shingle
        - stacks : int
            Number of copies of each column to stack

    :returns:
        - x_shingled : np.ndarray
            X with columns stacked
    '''
    return np.vstack([x[:, n:(x.shape[1] - stacks + n)]
                      for n in xrange(stacks)])


def symmetric_shingle(x, stacks):
    ''' Shingles a matrix column-wise symmetrically

    :parameters:
        - x : np.ndarray
            Matrix to shingle
        - stacks : int
            Number of copies of each column to stack

    :returns:
        - x_shingled : np.ndarray
            X with columns stacked symmetrically
    '''
    return np.vstack([np.pad(x, ((0, 0), (n, 0)), mode='constant')[:, :-n]
                      for n in xrange(1, stacks + 1)] + [x] +
                     [np.pad(x, ((0, 0), (0, n)), mode='constant')[:, n:]
                      for n in xrange(1, stacks + 1)])


def standardize(X):
    ''' Return column vectors to standardize X, via (X - X_mean)/X_std

    :parameters:
        - X : np.ndarray, shape=(n_features, n_examples)
            Data matrix

    :returns:
        - X_mean : np.ndarray, shape=(n_features, 1)
            Mean column vector
        - X_std : np.ndarray, shape=(n_features, 1)
            Standard deviation column vector
    '''
    std = np.std(X, axis=1).reshape(-1, 1)
    return np.mean(X, axis=1).reshape(-1, 1), std + (std == 0)


def midi_stft_generator(file_list, shuffle_indices=True):
    '''
    Given an iterable of MIDI files, generate STFTs and piano rolls of each.
    Shuffles both the file list and the columns of the STFTs/piano rolls.

    :parameters:
        - file_list : list
            Iterable list of MIDI files
        - shuffle_indices : bool
            Should the columns loaded by each file be shuffled?
    '''
    # Randomize the order of training
    shuffled_file_list = list(file_list)
    random.shuffle(shuffled_file_list)
    # Iterate over the list indefinitely
    while True:
        for filename in shuffled_file_list:
            print 'Loading {}'.format(filename)
            # Load in MIDI data
            midi = pretty_midi.PrettyMIDI(filename)
            # Synthesize and compute STFT
            stft = midi_to_stft(midi)
            # Create stacked matrix of STFT phase and magnitude
            Y = split_mag_phase(stft)
            # Create piano roll
            X = midi_to_piano_roll(midi)
            # Standardize the input features
            X_mean, X_std = standardize(X)
            X = (X - X_mean)/X_std
            # Shingle every 2 columns
            X = symmetric_shingle(X, 2)
            # Trim the smaller representation
            if Y.shape[1] > X.shape[1]:
                Y = Y[:, :X.shape[1]]
            else:
                X = X[:, :Y.shape[1]]
            if shuffle_indices:
                # Randomly shuffle the order of training examples
                shuffled_indices = np.random.permutation(X.shape[1])
                X = X[:, shuffled_indices]
                Y = Y[:, shuffled_indices]
            yield X, Y


def buffered_gen_mp(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate process.

    From Sander Dieleman:
        https://github.com/benanne/kaggle-galaxies

    :parameters:
        - source_gen : function
            Generator function which yields data samples
        - buffer_size : int
            Maximum number of samples to pre-generate
        - sleep_time : float
            When the buffer is full, wait this long to generate more data
    """
    buffer = mp.Queue(maxsize=buffer_size)

    def _buffered_generation_process(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in
            # generating more data
            # when the buffer is full, it only causes extra memory usage and
            # effectively
            # increases the buffer size by one.
            while buffer.full():
                # print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                # print "DEBUG: OUT OF DATA, CLOSING BUFFER"
                # signal that we're done putting data in the buffer
                buffer.close()
                break

            buffer.put(data)

    process = mp.Process(target=_buffered_generation_process,
                         args=(source_gen, buffer))
    process.start()

    while True:
        try:
            # yield buffer.get()
            # just blocking on buffer.get() here creates a problem: when get()
            # is called and the buffer is empty, this blocks. Subsequently
            # closing the buffer does NOT stop this block.  so the only
            # solution is to periodically time out and try again. That way
            # we'll pick up on the 'close' signal.
            try:
                yield buffer.get(True, timeout=sleep_time)
            except Queue.Empty:
                if not process.is_alive():
                    # no more data is going to come. This is a workaround
                    # because the buffer.close() signal does not seem to be
                    # reliable.
                    break

                # print "DEBUG: queue is empty, waiting..."
                # ignore this, just try again.
                pass
        # if the buffer has been closed, calling get() on it will raise
        # IOError.
        except IOError:
            # this means that we're done iterating.
            # print "DEBUG: buffer closed, stopping."
            break
