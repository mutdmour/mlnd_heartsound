# function [despiked_signal] = schmidt_spike_removal(original_signal, fs)
#
# This function removes the spikes in a signal as done by Schmidt et al in
# the paper:
# Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
# (2010). Segmentation of heart sound recordings by a duration-dependent
# hidden Markov model. Physiological Measurement, 31(4), 513-29.
#
# The spike removal process works as follows:
# (1) The recording is divided into 500 ms windows.
# (2) The maximum absolute amplitude (MAA) in each window is found.
# (3) If at least one MAA exceeds three times the median value of the MAA's,
# the following steps were carried out. If not continue to point 4.
#   (a) The window with the highest MAA was chosen.
#   (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
#   (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
#   (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
#   (e) The defined noise spike was replaced by zeroes.
#   (f) Resume at step 2.
# (4) Procedure completed.
#
## Inputs:
# original_signal: The original (1D) audio signal array
# fs: the sampling frequency (Hz)
#
## Outputs:
# despiked_signal: the audio signal with any spikes removed.
#
# This code is derived from the paper:
# S. E. Schmidt et al., "Segmentation of heart sound recordings by a
# duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
# no. 4, pp. 513-29, Apr. 2010.

import numpy as np
#function [despiked_signal] = 
def schmidt_spike_removal(original_signal, fs):

    ## Find the window size
    # (500 ms)
    windowsize = int(np.round(fs/2.))

    ## Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize

    ## Reshape the signal into a number of windows:
    # print len(original_signal), trailingsamples, windowsize
    sampleframes = original_signal[0:(len(original_signal)-trailingsamples)]
    # print sampleframes
    sampleframes = np.reshape(sampleframes, (len(sampleframes)/windowsize, windowsize))
    # print sampleframes

    ## Find the MAAs:
    # MAAs = [max(i) for i in MAAs]
    MAAs = np.amax(np.abs(sampleframes),axis=1)
    # print "MAAs", MAAs

    def hasSpike(arr): #xxx find a np function to do this
        maximum = np.median(arr)*3
        for i in arr:
            if (i > maximum):
                return True
        return False

    # While there are still samples greater than 3* the median value of the
    # MAAs, then remove those spikes:
    # print hasSpike(MAAs)
    while(hasSpike(MAAs)):
        # print MAAs
        #Find the window with the max MAA:
        # print "MAAs", MAAs
        window_num = np.argmax(MAAs)
        # print "->", window_num
        # print sampleframes[window_num][:]
        
        #Find the postion of the spike within that window:
        #window = np.abs(sampleframes[window_num][:])
        spike_position = np.argmax(np.abs(sampleframes[window_num][:]))
        # print "spike_position", spike_position
        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        zero_crossings = np.abs(np.diff(np.sign(sampleframes[window_num][:])))
        if (len(zero_crossings) == 0):
            zero_crossings = [0]
        zero_crossings = [1 if i > 1 else 0 for i in zero_crossings ] + [0]
        # print "zero_crossings", sampleframes[window_num][:], zero_crossings
        
        #Find the start of the spike, finding the last zero crossing before
        #spike position. If that is empty, take the start of the window:
        # spike_start = max([0, np.nonzero(zero_crossings)[0][-1]])
        # print zero_crossings[0:spike_position+1]
        spike_start = np.nonzero(zero_crossings[0:spike_position+1])[0]
        # print spike_start
        if (len(spike_start) > 0):
            spike_start = spike_start[-1]
        else:
            spike_start = 0
        # print "spike_start", spike_start

        #Find the end of the spike, finding the first zero crossing after
        #spike position. If that is empty, take the end of the window:
        zero_crossings[0:spike_position+1] = [0]*(spike_position+1)
        spike_end = np.nonzero(zero_crossings)[0]
        if (len(spike_end) > 0):
            spike_end = spike_end[0] + 1
        else:
            spike_end = windowsize
        # print "spike_end", spike_end
        
        #Set to Zero
        # print sampleframes[window_num][spike_start:spike_end]
        sampleframes[window_num][spike_start:spike_end] = [1.00e-4] * (spike_end - spike_start)
        # print sampleframes[window_num][spike_start:spike_end]

        #Recaclulate MAAs
        MAAs = np.amax(np.abs(sampleframes),axis=1)
        # print "->", MAAs

    # print sampleframes
    despiked_signal = np.reshape(sampleframes,-1) #len(sampleframes)*len(sampleframes[0]))
    # print despiked_signal

    # Add the trailing samples back to the signal:
    trailing = original_signal[len(despiked_signal):]
    # trailing = np.reshape(trailing, -1)
    # print trailing
    if (len(trailing) > 0):
        despiked_signal = np.concatenate((despiked_signal, trailing))

    return despiked_signal


if __name__ == '__main__':
    # a = [-5,-6,0,1,3,4,-5,-4,-4,-1,1,1] #len 12
    # assert(schmidt_spike_removal(a,6) == [-5,-6,0,1,3,4,-5,-4,-4,-1,1,1])

    # a = [-5,-6,0,1,3,4,-5,-4,-4,-1,1,1,7] #len 13
    # assert(schmidt_spike_removal(a,6) == [-5,-6,0,1,3,4,-5,-4,-4,-1,1,1,7])

    from numpy.testing import assert_almost_equal
    def test(original, fs, desired_despiked, print_output=False):
        actual = schmidt_spike_removal(np.array(original,dtype=np.float16),fs)
        if (print_output):
            print np.around(actual.tolist(),decimals=4)
        assert_almost_equal(actual, np.array(desired), decimal=4)

    a = [3,2,4,5,1]
    a += [3,-1,-79,3,5]
    a += [-5,-6,0,1,3] 
    a += [-5,-6,0,1,3] 
    a += [-5,-6,0,1,3] 
    a += [-5,-6,0,1,3] 
    a += [4,-5,-71,-4,-1]

    desired = [3,2,4,5,1,3,-1,1.000000e-04,1.000000e-04,1.000000e-04,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,1.000000e-04,1.000000e-04,1.000000e-04,1.000000e-04,1.000000e-04]
    test(a, 10, desired)

    desired = [3,2,4,5,1,1.000000e-04,1.000000e-04,1.000000e-04,3,5,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,-71,-4,-1]
    test(a, 8, desired)

    desired = [3,2,4,5,1,3,-1,1.000000e-04,1.000000e-04,1.000000e-04,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,-71,-4,-1]
    test(a, 12, desired)

    desired = [3,2,4,5,1,3,1.000000e-04,1.000000e-04,3,5,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,1.000000e-04,1.000000e-04,-1]
    test(a,3,desired)

    desired = [3,2,4,5,1,3,-1,1.000000e-04,1.000000e-04,1.000000e-04,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,-71,-4,-1]
    test(a,11,desired)

    a[7] = 99
    desired = [3,2,4,5,1,3,1.000000e-04,1.000000e-04,1.000000e-04,1.000000e-04,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,-71,-4,-1]
    test(a,11,desired)

    desired = [3,2,4,5,1,3,1.000000e-04,1.000000e-04,3,5,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,-5,-6,0,1,3,4,-5,1.000000e-04,1.000000e-04,-1]
    test(a,4,desired)
    
    print "schmidt_spike_removal.py has been tested successfully"
    
