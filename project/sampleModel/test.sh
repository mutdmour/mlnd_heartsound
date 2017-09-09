#!/usr/bin/env bash
echo
echo testing sample model...
python butterworth_filter.py
python default_Springer_HSMM_options.py
python expand_qt.py
python get_duration_distributions.py
python get_PSD_feature_Springer_HMM.py
python getDWT.py
python getHeartRateSchmidt.py
python getSpringerPCGFeatures.py
python Hilbert_Envelope.py
python Homomorphic_Envelope_with_Hilbert.py
python labelPCGStates.py
python mnrfit.py
python mnrval.py
python mvnpdf.py
python normalise_signal.py
python schmidt_spike_removal.py
python trainBandPiMatricesSpringer.py
python trainSpringerSegmentationAlgorithm.py
python viterbiDecodePCG_Springer
python xcorr.py

#TODO -W ignore to turn off warnings
# python runSpringerSegmentationAlgorithm.py