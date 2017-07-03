# function [hilbert_envelope] = Hilbert_Envelope(input_signal, sampling_frequency,figures)
#
# This function finds the Hilbert envelope of a signal. This is taken from:
#
# Choi et al, Comparison of envelope extraction algorithms for cardiac sound
# signal segmentation, Expert Systems with Applications, 2008
#
## Inputs:
# input_signal: the original signal
# samplingFrequency: the signal's sampling frequency
# figures: (optional) boolean variable to display a figure of both the
# original and normalised signal
#
## Outputs:
# hilbert_envelope is the hilbert envelope of the original signal

#function hilbert_envelope = 
import numpy as np
from scipy.signal import hilbert

def Hilbert_Envelope(input_signal, sampling_frequency,figures=False):
	hilbert_envelope = np.abs(hilbert(input_signal)); #find the envelope of the signal using the Hilbert transform
	return hilbert_envelope
	# if(figures)
	#     figure('Name', 'Hilbert Envelope');
	#     plot(input_signal');
	#     hold on;
	#     plot(hilbert_envelope,'r');
	#     legend('Original Signal','Hilbert Envelope');
	#     pause();
	# end

if __name__ == '__main__':
    import scipy.io
    input_signal = scipy.io.loadmat('./test_data/Hilbert_Envelope/input_signal.mat',struct_as_record=False)
    input_signal = input_signal['input_signal']
    input_signal = np.reshape(input_signal,np.shape(input_signal)[0])

    actual = Hilbert_Envelope(input_signal, 1000)

    desired = scipy.io.loadmat('./test_data/Hilbert_Envelope/hilbert_envelope.mat',struct_as_record=False)
    desired = desired['hilbert_envelope']
    # print np.shape(desired), desired
    desired = np.reshape(desired,np.shape(desired)[0]) 
    np.testing.assert_allclose(actual, desired, rtol=1e-07, atol=1e-7)
    print "Hilbert_Envelope.py has been tested successfully"