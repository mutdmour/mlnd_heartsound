# function [heartRate systolicTimeInterval] = getHeartRateSchmidt(audio_data, Fs, figures)
#
# Derive the heart rate and the sytolic time interval from a PCG recording.
# This is used in the duration-dependant HMM-based segmentation of the PCG
# recording.
#
# This method is based on analysis of the autocorrelation function, and the
# positions of the peaks therein.
#
# This code is derived from the paper:
# S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
# duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
# no. 4, pp. 513-29, Apr. 2010.
#
# Developed by David Springer for comparison purposes in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
## INPUTS:
# audio_data: The raw audio data from the PCG recording
# Fs: the sampling frequency of the audio recording
# figures: optional boolean to display figures
#
## OUTPUTS:
# heartRate: the heart rate of the PCG in beats per minute
# systolicTimeInterval: the duration of systole, as derived from the
# autocorrelation function, in seconds

import butterworth_filter as bf
import schmidt_spike_removal as ssr
import Homomorphic_Envelope_with_Hilbert as heh
import numpy as np

# import normalise_signal as ns
# import Hilbert_Envelope as he
# from scipy.signal import resample_poly
# import get_PSD_feature_Springer_HMM as getPSD
# import getDWT as gdwt

#function [heartRate, systolicTimeInterval] = 
def getHeartRateSchmidt(audio_data, Fs, figures):
	## Get heatrate:
	# From Schmidt:
	# "The duration of the heart cycle is estimated as the time from lag zero
	# to the highest peaks between 500 and 2000 ms in the resulting
	# autocorrelation"
	# This is performed after filtering and spike removal:
	Fs = float(Fs)

	## 25-400Hz 4th order Butterworth band pass
	audio_data = bf.butterworth_low_pass_filter(audio_data,2,400,Fs, false)
	audio_data = bf.butterworth_high_pass_filter(audio_data,2,25,Fs)

	## Spike removal from the original paper:
	audio_data = ssr.schmidt_spike_removal(audio_data,Fs)

	## Find the homomorphic envelope
	homomorphic_envelope = heh.Homomorphic_Envelope_with_Hilbert(audio_data, Fs)

	## Find the autocorrelation:
	y=homomorphic_envelope-np.mean(homomorphic_envelope)
	[c] = xcorr(y,'coeff') #xxx ?
	signal_autocorrelation = c[len(homomorphic_envelope)+1:end]

	min_index = 0.5*Fs
	max_index = 2*Fs

	index = np.argmax(signal_autocorrelation[min_index:max_index])
	true_index = index+min_index-1

	heartRate = 60/(true_index/Fs)

	## Find the systolic time interval:
	# From Schmidt: "The systolic duration is defined as the time from lag zero
	# to the highest peak in the interval between 200 ms and half of the heart
	# cycle duration"
	max_sys_duration = int(np.round(((60/heartRate)*Fs)/2))
	min_sys_duration = np.round(0.2*Fs)

	pos = np.argmax(signal_autocorrelation[min_sys_duration:max_sys_duration])
	systolicTimeInterval = (min_sys_duration+pos)/Fs

	return heartRate, systolicTimeInterval

	# if(figures)
	#     figure('Name', 'Heart rate calculation figure')
	#     plot(signal_autocorrelation)
	#     hold on
	#     plot(true_index, signal_autocorrelation(true_index),'ro')
	#     plot((min_sys_duration+pos), signal_autocorrelation((min_sys_duration+pos)), 'mo')
	#     xlabel('Samples')
	#     legend('Autocorrelation', 'Position of max peak used to calculate HR', 'Position of max peak within systolic interval')