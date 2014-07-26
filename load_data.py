'''
Functions for loading in data
'''

import multiprocessing as mp
import time
import Queue
import numpy as np
import librosa


def midi_to_stft(midi, fs=22050, **kwargs):
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
    audio_data = midi.fluidsynth(fs=22050)
    return librosa.stft(audio_data, **kwargs)


def midi_to_stacked_piano_roll(midi, hop_seconds=2048/4/22050.):
    '''
    Converts MIDI data into matrix of stacked piano rolls, one for each
    instrument class.
    Drum instruments and special effects are ignored.

    :parameters:
        - midi : pretty_midi.PrettyMIDI
            MIDI data object
        - hop_seconds : float
            Time between each column in the piano roll matrix

    :returns:
        - stacked_piano_roll : np.ndarray
            Stacked piano roll representation
    '''


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


def midi_stft_generator(file_list):
    '''
    Given an iterable of MIDI files, generate STFTs and piano rolls of each.
    Shuffles both the file list and the columns of the STFTs/piano rolls.

    :parameters:
        - file_list : iterable
            Iterable list of MIDI files
    '''


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
