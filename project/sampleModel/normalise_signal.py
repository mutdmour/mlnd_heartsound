# function [normalised_signal] = normalise_signal(signal)
#
# This function subtracts the mean and divides by the standard deviation of
# a (1D) signal in order to normalise it for machine learning applications.
#
## Inputs:
# signal: the original signal
#
## Outputs:
# normalised_signal: the original signal, minus the mean and divided by
# the standard deviation.
#
# Developed by David Springer for the paper:
# D. Springer et al., ?Logistic Regression-HSMM-based Heart Sound
# Segmentation,? IEEE Trans. Biomed. Eng., In Press, 2015.

import numpy as np

def normalise_signal(signal):
	mean_of_signal = np.mean(signal)
	standard_deviation = np.std(signal)
	normalised_signal = (signal - mean_of_signal)/float(standard_deviation)
	return normalised_signal

if __name__ == '__main__':
    import scipy.io
    signal = scipy.io.loadmat('./test_data/normalise_signal/signal.mat',struct_as_record=False)
    signal = signal['signal']
    signal = np.reshape(signal,np.shape(signal)[0])

    actual = normalise_signal(signal)

    desired = scipy.io.loadmat('./test_data/normalise_signal/normalised_signal.mat',struct_as_record=False)
    desired = desired['normalized_signal']
    desired = np.reshape(desired,np.shape(desired)[0])

    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-4)
    print "normalize_signal.py has been tested successfully"