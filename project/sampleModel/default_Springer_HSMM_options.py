# function springer_options = default_Springer_HSMM_options()
#
# The default options to be used with the Springer segmentation algorithm.
# USAGE: springer_options = default_Springer_HSMM_options
#
# Developed for use in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.

def default_Springer_HSMM_options():

	springer_options = {}

	## The sampling frequency at which to extract signal features:
	springer_options['audio_Fs'] = 1000

	## The downsampled frequency
	#Set to 50 in Springer paper
	springer_options['audio_segmentation_Fs'] = 50

	## Tolerance for S1 and S2 localization
	springer_options['segmentation_tolerance'] = 0.1 #seconds

	## Whether to use the mex code or not:
	springer_options['use_mex'] = True #XXX

	## Whether to use the wavelet function or not:
	springer_options['include_wavelet_feature'] = True

	return springer_options

if __name__ == '__main__':
	desired = {'audio_Fs': 1000, 'segmentation_tolerance': 0.1, 'use_mex': True, 'audio_segmentation_Fs': 50, 'include_wavelet_feature': True}
	actual = default_Springer_HSMM_options()
	assert(actual == desired)
    print "default_Springer_HSMM_options.py has been tested successfully"
