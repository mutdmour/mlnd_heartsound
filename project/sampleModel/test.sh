echo
echo testing sample model...
python default_Springer_HSMM_options.py
python butterworth_filter.py
python get_PSD_feature_Springer_HMM.py 
python schmidt_spike_removal.py
python Hilbert_Envelope.py
python Homomorphic_Envelope_with_Hilbert.py
python getDWT.py
python labelPCGStates.py
python normalise_signal.py
python get_duration_distributions.py
python expand_qt.py
python mnrfit.py
python trainBandPiMatricesSpringer.py

#-W ignore to turn off warnings
# python getSpringerPCGFeatures.py
# python trainSpringerSegmentationAlgorithm.py