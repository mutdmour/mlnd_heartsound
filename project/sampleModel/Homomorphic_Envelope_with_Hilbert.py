# function homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency,lpf_frequency,figures)
#
# This function finds the homomorphic envelope of a signal, using the method
# described in the following publications:
#
# S. E. Schmidt et al., ?Segmentation of heart sound recordings by a 
# duration-dependent hidden Markov model.,? Physiol. Meas., vol. 31, no. 4,
# pp. 513?29, Apr. 2010.
# 
# C. Gupta et al., ?Neural network classification of homomorphic segmented
# heart sounds,? Appl. Soft Comput., vol. 7, no. 1, pp. 286?297, Jan. 2007.
#
# D. Gill et al., ?Detection and identification of heart sounds using 
# homomorphic envelogram and self-organizing probabilistic model,? in 
# Computers in Cardiology, 2005, pp. 957?960.
# (However, these researchers found the homomorphic envelope of shannon
# energy.)
#
# In I. Rezek and S. Roberts, ?Envelope Extraction via Complex Homomorphic
# Filtering. Technical Report TR-98-9,? London, 1998, the researchers state
# that the singularity at 0 when using the natural logarithm (resulting in
# values of -inf) can be fixed by using a complex valued signal. They
# motivate the use of the Hilbert transform to find the analytic signal,
# which is a converstion of a real-valued signal to a complex-valued
# signal, which is unaffected by the singularity. 
#
# A zero-phase low-pass Butterworth filter is used to extract the envelope.
## Inputs:
# input_signal: the original signal (1D) signal
# samplingFrequency: the signal's sampling frequency (Hz)
# lpf_frequency: the frequency cut-off of the low-pass filter to be used in
# the envelope extraciton (Default = 8 Hz as in Schmidt's publication).
# figures: (optional) boolean variable dictating the display of a figure of
# both the original signal and the extracted envelope:
#
## Outputs:
# homomorphic_envelope: The homomorphic envelope of the original
# signal (not normalised).

from scipy.signal import butter, filtfilt, hilbert
import numpy as np

#function homomorphic_envelope = 
def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency,lpf_frequency=8,figures=False):

	#8Hz, 1st order, Butterworth LPF
	B_low, A_low = butter(1,2*lpf_frequency/float(sampling_frequency),'low')
	homomorphic_envelope = np.exp(filtfilt(B_low,A_low,np.log(np.abs(hilbert(input_signal)))))
	return homomorphic_envelope
	# print homomorphic_envelope
	# Remove spurious spikes in first sample:
	# homomorphic_envelope[1] = [homomorphic_envelope[2]]

	# if(figures)
	#     figure('Name', 'Homomorphic Envelope')
	#     plot(input_signal)
	#     hold on
	#     plot(homomorphic_envelope,'r')
	#     legend('Original Signal','Homomorphic Envelope')

if __name__ == '__main__':
    import scipy.io
    input_signal = scipy.io.loadmat('./test_data/Homomorphic_Envelope_with_Hilbert/input_signal.mat',struct_as_record=False)
    input_signal = input_signal['input_signal']
    input_signal = np.reshape(input_signal,np.shape(input_signal)[0])

    actual = Homomorphic_Envelope_with_Hilbert(input_signal, 1000)

    desired = scipy.io.loadmat('./test_data/Homomorphic_Envelope_with_Hilbert/homomorphic_envelope.mat',struct_as_record=False)
    desired = desired['homomorphic_envelope']
    desired = np.reshape(desired,np.shape(desired)[0]) 
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=.5)
    print "Homomorphic_Envelope_with_Hilbert.py has been tested successfully"

