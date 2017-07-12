# function expanded_qt = expand_qt(original_qt, old_fs, new_fs, new_length)
# 
# Function to expand the derived HMM states to a higher sampling frequency. 
#
# Developed by David Springer for comparison purposes in the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
## INPUTS:
# original_qt: the original derived states from the HMM
# old_fs: the old sampling frequency of the original_qt
# new_fs: the desired sampling frequency
# new_length: the desired length of the qt signal

## Outputs:
# expanded_qt: the expanded qt, to the new FS and length

# not used anywere in code
# function expanded_qt = 
import numpy as np
def expand_qt(original_qt, old_fs, new_fs, new_length):
    # original_qt = original_qt
    expanded_qt = np.zeros(new_length)

    indeces_of_changes = np.nonzero(np.diff(original_qt))
    indeces_of_changes = np.append(indeces_of_changes, len(original_qt))

    start_index = -1
    vals = []
    for i in range(0,len(indeces_of_changes)):
        # start_index
        end_index = indeces_of_changes[i]
        mid_point = np.round((end_index - start_index)/2) + start_index

        value_at_mid_point = original_qt[mid_point]
        
        old_fs=float(old_fs)
        new_fs=float(new_fs)
        expanded_start_index = round(((start_index+1)/old_fs)*new_fs) + 1
        expanded_end_index = round(((end_index+1)/old_fs)*new_fs)

        if(expanded_end_index > new_length):
            expanded_end_index = new_length
        expanded_qt[expanded_start_index-1:expanded_end_index] = value_at_mid_point

        start_index = end_index
    return expanded_qt.astype(int)

if __name__ == '__main__':
    import scipy.io
    original_qt = scipy.io.loadmat('./test_data/expand_qt/original_qt.mat',struct_as_record=False)
    original_qt = original_qt['original_qt']
    original_qt = np.reshape(original_qt,np.shape(original_qt)[1])

    actual = expand_qt(original_qt,50,1000,34000).astype(int)

    desired = scipy.io.loadmat('./test_data/expand_qt/expanded_qt.mat',struct_as_record=False)
    desired = desired['expanded_qt']
    desired = np.reshape(desired,np.shape(desired)[0])

    np.testing.assert_array_equal(actual, desired)
    print "expand_qt.py has been tested successfully"