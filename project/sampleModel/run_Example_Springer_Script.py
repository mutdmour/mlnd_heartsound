## Example Springer script
# A script to demonstrate the use of the Springer segmentation algorithm

## Copyright (C) 2016  David Springer
# dave.springer@gmail.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Remastered in Python - Mutasem Aldmour

import default_Springer_HSMM_options
import trainSpringerSegmentationAlgorithm
import scipy.io
import numpy as np

def runExampleScript():
	## Load the default options:
	# These options control options such as the original sampling frequency of
	# the data, the sampling frequency for the derived features and whether the
	# mex code should be used for the Viterbi decoding:
	springer_options = default_Springer_HSMM_options.default_Springer_HSMM_options()

	# print springer_options
	## Load the audio data and the annotations:
	# These are 6 example PCG recordings, downsampled to 1000 Hz, with
	# annotations of the R-peak and end-T-wave positions.
	example_data = scipy.io.loadmat('example_data.mat',struct_as_record=False) ## numpy.ndarray
	example_data = example_data['example_data'][0][0]

	## extract variables
	example_audio_data = example_data.example_audio_data
	example_annotations = example_data.example_annotations
	# there's also patient_number and binary_diagnosis

	## Split the data into train and test sets:
	# Select the 5 recordings for training and a sixth for testing:
	training_indices = [0, 46, 360, 401, 571]
	train_recordings = [example_audio_data[0][i] for i in training_indices]
	train_annotations = [example_annotations[i] for i in training_indices]

	test_index = 663
	test_recordings = example_audio_data[0][test_index] 
	test_annotations = example_annotations[test_index]


	## Train the HMM:
	#xxx B_matrix, pi_vector, total_obs_distribution = trainSpringerSegmentationAlgorithm.trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options['audio_Fs'], False)
	x = trainSpringerSegmentationAlgorithm.trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options, False)

	# ## Run the HMM on an unseen test recording:
	# # And display the resulting segmentation
	# numPCGs = len(test_recordings)

	# for PCGi in range(1,numPCGs):
	#     [assigned_states] = runSpringerSegmentationAlgorithm(test_recordings{PCGi}, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, true) #XXX

if __name__ == '__main__':
	runExampleScript()
