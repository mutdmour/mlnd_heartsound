#function [psd] = get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high, figures)
#
# PSD-based feature extraction for heart sound segmentation.
#
## INPUTS:
# data: this is the audio waveform
# sampling_frequency is self-explanatory
# frequency_limit_low is the lower-bound on the frequency range you want to
# analyse
# frequency_limit_high is the upper-bound on the frequency range
# figures: (optional) boolean variable to display figures
#
## OUTPUTS:
# psd is the array of maximum PSD values between the max and min limits,
# resampled to the same size as the original data.

#function [psd] = 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

######## xxx wrong
def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high, figures=False):

	# Find the spectrogram of the signal:
	# [~,F,T,P] = spectrogram(data,sampling_frequency/40,round(sampling_frequency/80),1:1:round(sampling_frequency/2),sampling_frequency)
	nfft = int(round(sampling_frequency/40.))
	window = np.hamming(nfft)
	# print len(data), nfft, len(window)
	Pxx, freqs, bins = matplotlib.mlab.specgram(data, 
							noverlap=round(sampling_frequency/80.), 
							NFFT=nfft,
							pad_to=sampling_frequency,
							window=window,
							Fs=sampling_frequency, 
							mode='psd',
							scale_by_freq=False) 
	#F(cyclical frequencies) = 1:1:round(sampling_frequency/2), window=sampling_frequency/40., 
	# print len(freqs), freqs
	# print len(bins), bins
	# print Pxx


	# if(figures)
	#     figure()
	#     surf(T,F,10*log(P),'edgecolor','none') axis tight
	#     view(0,90)
	#     xlabel('Time (Seconds)') ylabel('Hz')
	#     pause()
	# end

	F = freqs
	low_limit_position = np.argmin(np.abs(F - frequency_limit_low))
	high_limit_position = np.argmin(np.abs(F - frequency_limit_high))

	# Find the mean PSD over the frequency range of interest:
	P = Pxx/100
	psd = np.mean(P[low_limit_position:high_limit_position][:],axis=0)
	# print len(psd), psd
	return psd

	# if(figures)
	#     t4  = (1:length(psd))./sampling_frequency
	#     t3  = (1:length(data))./sampling_frequency
	#     figure('Name', 'PSD Feature')
	    
	#     plot(t3,(data - mean(data))./std(data),'c')
	#     hold on
	    
	#     plot(t4, (psd - mean(psd))./std(psd),'k')
	    
	#     pause()
	# end

if __name__ == '__main__':
	import scipy.io
	data = scipy.io.loadmat('./test_data/get_PSD_feature_Springer_HMM/data.mat',struct_as_record=False)
	data = data['data']
	data = np.reshape(data,np.shape(data)[0])

	actual = get_PSD_feature_Springer_HMM(data,1000,40,60)

	desired = scipy.io.loadmat('./test_data/get_PSD_feature_Springer_HMM/psd.mat',struct_as_record=False)
	desired = desired['psd']
	desired = np.reshape(desired,np.shape(desired)[1])
	np.testing.assert_allclose(actual, desired, rtol=1e-07, atol=1e-1) #xxx increase accuracy

	print "get_PSD_feature_Springer_HMM.py has been tested successfully"
