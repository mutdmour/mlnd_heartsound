# function high_pass_filtered_signal = butterworth_high_pass_filter(original_signal,order,cutoff,sampling_frequency)
#
# High-pass filter a given signal using a forward-backward, zero-phase
# butterworth filter.
#
## INPUTS:
# original_signal: The 1D signal to be filtered
# order: The order of the filter (1,2,3,4 etc). NOTE: This order is
# effectively doubled as this function uses a forward-backward filter that
# ensures zero phase distortion
# cutoff: The frequency cutoff for the high-pass filter (in Hz)
# sampling_frequency: The sampling frequency of the signal being filtered
# (in Hz).
# figures (optional): boolean variable dictating the display of figures
#
## OUTPUTS:
# high_pass_filtered_signal: the high-pass filtered signal.
#
# This code is derived from the paper:
# S. E. Schmidt et al., "Segmentation of heart sound recordings by a
# duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
# no. 4, pp. 513-29, Apr. 2010.

#function high_pass_filtered_signal = 
from scipy.signal import butter, filtfilt
import numpy as np

def butterworth_highpass_filter(original_signal,order,cutoff,sampling_frequency,figures=False):
	return butterworth_filter(original_signal,order,cutoff,sampling_frequency,'highpass',figures)

def butterworth_lowpass_filter(original_signal,order,cutoff,sampling_frequency,figures=False):
	return butterworth_filter(original_signal,order,cutoff,sampling_frequency,'lowpass',figures)

def butterworth_filter(original_signal,order,cutoff,sampling_frequency,ftype,figures=False):
	#Get the butterworth filter coefficients
	B,A = butter(order,2.*cutoff/sampling_frequency,ftype)

	#Forward-backward filter the original signal using the butterworth
	#coefficients, ensuring zero phase distortion
	filtered_signal = filtfilt(B,A,original_signal)
	# print B, A
	print filtered_signal[0:10]

	return filtered_signal

	# if(figures)
	    
	#     figure('Name','High-pass filter frequency response')
	#     [sos,g] = zp2sos(B_high,A_high,1)	     # Convert to SOS form
	#     Hd = dfilt.df2tsos(sos,g)   # Create a dfilt object
	#     h = fvtool(Hd)	             # Plot magnitude response
	#     set(h,'Analysis','freq')	     # Display frequency response
	    
	#     figure('Name','Original vs. high-pass filtered signal')
	#     plot(original_signal)
	#     hold on
	#     plot(high_pass_filtered_signal,'r')
	#     legend('Original Signal', 'High-pass filtered signal')
	#     pause()
	# end

if __name__ == '__main__':
	import scipy.io
	signal = scipy.io.loadmat('./test_data/butterworth_filter/audio_data.mat',struct_as_record=False)
	signal = signal['audio_data']
	signal = np.reshape(signal,np.shape(signal)[0])
	# print np.shape(signal), signal
	Fs = 1000

	actual_lowpass = butterworth_lowpass_filter(signal,2,400,Fs)
	desired_lowpass = scipy.io.loadmat('./test_data/butterworth_filter/low_pass_filtered_audio_data.mat',struct_as_record=False)
	desired_lowpass = desired_lowpass['low_pass_filtered_audio_data']
	desired_lowpass = np.reshape(desired_lowpass,np.shape(desired_lowpass)[0])
	np.testing.assert_allclose(actual_lowpass, desired_lowpass, rtol=1e-02)

	actual_highpass = butterworth_highpass_filter(signal,2,25,Fs)
	desired_highpass = scipy.io.loadmat('./test_data/butterworth_filter/high_pass_filtered_audio_data.mat',struct_as_record=False)
	desired_highpass = desired_highpass['high_pass_filtered_audio_data']
	desired_highpass = np.reshape(desired_highpass,np.shape(desired_highpass)[0])
	np.testing.assert_allclose(actual_highpass, desired_highpass, rtol=1e-02)
	# print signal
    # state_observation_values = np.transpose(state_observation_values['state_observation_values'])
    # print np.shape(state_observation_values)
    # print state_observation_values[0:10][0]

	print "butterworth_filter.py has been tested successfully"
