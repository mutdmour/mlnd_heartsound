# function [PCG_Features, featuresFs] = getSpringerPCGFeatures(audio_data, Fs, figures)
#
# Get the features used in the Springer segmentation algorithm. These 
# features include:
# -The homomorphic envelope (as performed in Schmidt et al's paper)
# -The Hilbert envelope
# -A wavelet-based feature
# -A PSD-based feature
# This function was developed for use in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
## INPUTS:
# audio_data: array of data from which to extract features
# Fs: the sampling frequency of the audio data
# figures (optional): boolean variable dictating the display of figures
#
## OUTPUTS:
# PCG_Features: array of derived features
# featuresFs: the sampling frequency of the derived features. This is set
# in default_Springer_HSMM_options.m

#[PCG_Features, featuresFs] = 
import butterworth_filter as bf
import schmidt_spike_removal as ssr
import Homomorphic_Envelope_with_Hilbert as heh
import normalise_signal as ns
import Hilbert_Envelope as he
from scipy.signal import resample_poly
import get_PSD_feature_Springer_HMM as getPSD
import numpy as np
import getDWT as gdwt
# from scipy.signal import resample_poly
# from sp.multirate import resample

def getSpringerPCGFeatures(audio_data, springer_options, figures=False):
    # function PCG_Features = getSpringerPCGFeatures(audio, Fs)
    # Get the features used in the Springer segmentation algorithm.

    # Check to see if the Wavelet toolbox is available on the machine:
    Fs = springer_options['audio_Fs']
    include_wavelet = springer_options['include_wavelet_feature']
    featuresFs = springer_options['audio_segmentation_Fs'] # Downsampled feature sampling frequency
    print "orig", np.shape(audio_data), audio_data

    ## 25-400Hz 4th order Butterworth band pass
    audio_data = bf.butterworth_lowpass_filter(audio_data,2,400,Fs,'lowpass')
    print "lowpass", np.shape(audio_data), audio_data

    audio_data = bf.butterworth_highpass_filter(audio_data,2,25,Fs,'highpass')
    print "highpass", np.shape(audio_data), audio_data
    ## Spike removal from the original paper:
    audio_data = ssr.schmidt_spike_removal(audio_data,Fs)
    # print np.shape(audio_data), audio_data

    # Find the homomorphic envelope
    homomorphic_envelope = heh.Homomorphic_Envelope_with_Hilbert(audio_data, Fs)
    # print np.shape(homomorphic_envelope), homomorphic_envelope
    # Downsample the envelope:
    # print featuresFs, Fs
    # print homomorphic_envelope
    downsampled_homomorphic_envelope = resample_poly(homomorphic_envelope,featuresFs, Fs)
    # normalise the envelope:
    downsampled_homomorphic_envelope = ns.normalise_signal(downsampled_homomorphic_envelope)

    ## Hilbert Envelope
    hilbert_envelope = he.Hilbert_Envelope(audio_data, Fs)
    downsampled_hilbert_envelope = resample_poly(hilbert_envelope, featuresFs, Fs)
    downsampled_hilbert_envelope = ns.normalise_signal(downsampled_hilbert_envelope)

    ## Power spectral density feature:

    psd = np.transpose(getPSD.get_PSD_feature_Springer_HMM(audio_data, Fs, 40,60))
    psd = resample_poly(psd, len(downsampled_homomorphic_envelope), len(psd))
    psd = ns.normalise_signal(psd)

    ## Wavelet features:

    if (include_wavelet):
        wavelet_level = 3
        wavelet_name = 'rbio3.9'
        
        # Audio needs to be longer than 1 second for getDWT to work:
        if (len(audio_data)< Fs*1.025):
            audio_data = np.concatenate(audio_data,np.zeros(round(0.025*Fs)))
        
        cD = gdwt.getDWT(audio_data,wavelet_level,wavelet_name) #cA not used
        
        wavelet_feature = abs(cD[wavelet_level-1][:])
        wavelet_feature = wavelet_feature[0:len(homomorphic_envelope)]
        downsampled_wavelet = resample_poly(wavelet_feature, featuresFs, Fs)
        downsampled_wavelet =  np.transpose(ns.normalise_signal(downsampled_wavelet))

    if(include_wavelet):
        PCG_Features = [downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd, downsampled_wavelet]
    else:
        PCG_Features = [downsampled_homomorphic_envelope, downsampled_hilbert_envelope, psd]

    ## Plotting figures
    # if(figures)
    #     figure('Name', 'PCG features')
    #     t1 = (1:length(audio_data))./Fs
    #     plot(t1,audio_data)
    #     hold on
    #     t2 = (1:length(PCG_Features))./featuresFs
    #     plot(t2,PCG_Features)
    #     pause()
    return PCG_Features, featuresFs

if __name__ == '__main__':
    import scipy.io
    input_signal = scipy.io.loadmat('./test_data/getSpringerPCGFeatures/audio_data.mat',struct_as_record=False)
    input_signal = input_signal['audio_data']
    input_signal = np.reshape(input_signal,np.shape(input_signal)[0])

    import default_Springer_HSMM_options
    springer_options = default_Springer_HSMM_options.default_Springer_HSMM_options()
    
    actual = getSpringerPCGFeatures(input_signal, springer_options)

    desired = scipy.io.loadmat('./test_data/getSpringerPCGFeatures/PCG_Features.mat',struct_as_record=False)
    desired = desired['PCG_Features']
    desired = np.transpose(desired)

    np.testing.assert_allclose(desired, actual, rtol=1e-07, atol=1e-7)
    print "getSpringerPCGFeatures.py has been tested successfully"
