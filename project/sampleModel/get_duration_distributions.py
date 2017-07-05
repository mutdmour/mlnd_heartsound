# function [d_distributions max_S1 min_S1 max_S2 min_S2 max_systole min_systole max_diastole min_diastole] = get_duration_distributions(heartrate,systolic_time)
#
# This function calculates the duration distributions for each heart cycle
# state, and the minimum and maximum times for each state.
#
## Inputs:
# heartrate is the calculated average heart rate over the entire recording
# systolic_time is the systolic time interval
#
## Outputs:
# d_distributions is a 4 (the number of states) dimensional vector of
# gaussian mixture models (one dimensional in this case), representing the
# mean and std deviation of the duration in each state.
#
# The max and min values are self-explanatory.
#
# This code is implemented as outlined in the paper:
# S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
# duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
# no. 4, pp. 513-29, Apr. 2010.


#function [d_distributions max_S1 min_S1 max_S2 min_S2 max_systole min_systole max_diastole min_diastole] = 
import default_Springer_HSMM_options
import numpy as np
def get_duration_distributions(heartrate,systolic_time):
	springer_options = default_Springer_HSMM_options.default_Springer_HSMM_options()
	fs = float(springer_options['audio_segmentation_Fs'])

	mean_S1 = np.round(0.122*fs)
	std_S1 = np.round(0.022*fs)
	mean_S2 = np.round(0.094*fs)
	std_S2 = np.round(0.022*fs)
	# print mean_S1, std_S1, mean_S2, std_S2 #6 1 6 1 all good

	mean_systole = np.round(systolic_time*fs) - mean_S1
	std_systole = (25./1000)*fs

	mean_diastole = ((60./heartrate) - systolic_time - 0.094)*fs
	std_diastole = 0.07*mean_diastole + (6./1000)*fs

	## Cell array for the mean and covariance of the duration distributions:
	d_distributions = np.zeros((4,2))

	## Assign mean and covariance values to d_distributions:
	d_distributions[0,0] = mean_S1
	d_distributions[0,1] = (std_S1)**2

	d_distributions[1,0] = mean_systole
	d_distributions[1,1] = (std_systole)**2

	d_distributions[2,0] = mean_S2
	d_distributions[2,1] = (std_S2)**2

	d_distributions[3,0] = mean_diastole
	d_distributions[3,1] = (std_diastole)**2


	#Min systole and diastole times
	min_systole = mean_systole - 3*(std_systole+std_S1)
	max_systole = mean_systole + 3*(std_systole+std_S1)

	min_diastole = mean_diastole-3*std_diastole
	max_diastole = mean_diastole + 3*std_diastole



	#Setting the Min and Max values for the S1 and S2 sounds:
	#If the minimum lengths are less than a 50th of the sampling frequency, set
	#to a 50th of the sampling frequency:
	min_S1 = (mean_S1 - 3*(std_S1))
	if(min_S1<(fs/50)):
	    min_S1 = (fs/50)

	min_S2 = (mean_S2 - 3*(std_S2))
	if(min_S2<(fs/50)):
	    min_S2 = (fs/50)
	max_S1 = (mean_S1 + 3*(std_S1))
	max_S2 = (mean_S2 + 3*(std_S2))


	return d_distributions,max_S1,min_S1,max_S2,min_S2,max_systole,min_systole,max_diastole,min_diastole

if __name__ == '__main__':
	heartrate = 51.4139
	systolic_time = 0.3910

	actual = get_duration_distributions(heartrate, systolic_time)
	actual_d_distributions, actual_max_S1, actual_min_S1, actual_max_S2, actual_min_S2, actual_max_systole, actual_min_systole,actual_max_diastole,actual_min_diastole = actual

	desired_d_distributions = np.array([[6.,1.],[14.,1.5625],[5.,1.],[34.1000,7.2200]])
	np.testing.assert_allclose(desired_d_distributions, actual_d_distributions, atol=1e-1, rtol=0)

	def test(a, b):
		np.testing.assert_almost_equal(a,b,decimal=4)

	desired_max_S1 = 9
	test(desired_max_S1, actual_max_S1)
	desired_min_S1 = 3
	test(desired_min_S1, actual_min_S1)

	desired_max_S2 = 8
	test(desired_max_S2, actual_max_S2)
	desired_min_S2 = 2
	test(desired_min_S2, actual_min_S2)

	desired_max_systole = 20.7500
	test(desired_max_systole, actual_max_systole)
	desired_min_systole = 7.2500 
	test(desired_min_systole, actual_min_systole)

	desired_max_diastole = 42.1610
	test(desired_max_diastole, actual_max_diastole)
	desired_min_diastole = 26.0390
	test(desired_min_diastole, actual_min_diastole)

	print "get_duration_distributions.py has been tested successfully"