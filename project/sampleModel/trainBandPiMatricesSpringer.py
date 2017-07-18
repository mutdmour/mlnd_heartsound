# function [B_matrix, pi_vector, total_obs_distribution] = trainBandPiMatricesSpringer(state_observation_values)
#
# Train the B matrix and pi vector for the Springer HMM.
# The pi vector is the initial state probability, while the B matrix are
# the observation probabilities. In the case of Springer's algorith, the
# observation probabilities are based on a logistic regression-based
# probabilities. 
#
## Inputs:
# state_observation_values: an Nx4 cell array of observation values from
# each of N PCG signals for each (of 4) state. Within each cell is a KxJ
# double array, where K is the number of samples from that state in the PCG
# and J is the number of feature vectors extracted from the PCG.
#
## Outputs:
# The B_matrix and pi arrays for an HMM - as Springer et al's algorithm is a
# duration dependant HMM, there is no need to calculate the A_matrix, as
# the transition between states is only dependant on the state durations.
# total_obs_distribution:
#
# Developed by David Springer for the paper:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound 
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.

import numpy as np
from sklearn.linear_model import LogisticRegression
import mnrfit

def trainBandPiMatricesSpringer(state_observation_values):
    ## Prelim
    number_of_states = 4

    ## Set pi_vector
    # The true value of the pi vector, which are the initial state
    # probabilities, are dependant on the heart rate of each PCG, and the
    # individual sound duration for each patient. Therefore, instead of setting
    # a patient-dependant pi_vector, simplify by setting all states as equally
    # probable:

    pi_vector = [0.25,0.25,0.25,0.25]

    ## Train the logistic regression-based B_matrix:


    # Initialise the B_matrix as a 1x4 cell array. This is to hold the
    # coefficients of the trained logisitic regression model for each state.
    B_matrix = np.empty(number_of_states,dtype=object)


    statei_values = [np.array([],dtype=object)]*4#np.empty(number_of_states,dtype=object);
    # print statei_values

    for PCGi in range(0,len(state_observation_values)):
        if (len(statei_values[0]) > 0):
            # print np.shape(statei_values[0]), np.shape(state_observation_values[PCGi][0])
            statei_values[0] = np.hstack((statei_values[0],state_observation_values[PCGi][0]))
            statei_values[1] = np.hstack((statei_values[1],state_observation_values[PCGi][1]))
            statei_values[2] = np.hstack((statei_values[2],state_observation_values[PCGi][2]))
            statei_values[3] = np.hstack((statei_values[3],state_observation_values[PCGi][3]))
        else:  
            # print "yo"
            statei_values[0] = state_observation_values[PCGi][0]
            statei_values[1] = state_observation_values[PCGi][1]
            statei_values[2] = state_observation_values[PCGi][2]
            statei_values[3] = state_observation_values[PCGi][3]
    # print "statei",np.shape(statei_values)

    # In order to use Bayes' formula with the logistic regression derived
    # probabilities, we need to get the probability of seeing a specific
    # observation in the total training data set. This is the
    # 'total_observation_sequence', and the mean and covariance for each state
    # is found:
    # total_observation_sequence = np.concatenate((statei_values[0], statei_values[1], statei_values[2], statei_values[3])) #xxx simplify
    total_observation_sequence = np.hstack((statei_values[0], statei_values[1], statei_values[2], statei_values[3]))
    # print np.shape(total_observation_sequence)
    total_obs_distribution = np.empty(2, dtype=object)
    total_obs_distribution[0] = np.mean(total_observation_sequence, axis=1)
    total_obs_distribution[1] = np.cov(total_observation_sequence)
    # print "obs",total_obs_distribution
    # statei_values = np.array(statei_values)

    for state in range(1,number_of_states+1):
        
        # Randomly select indices of samples from the other states not being 
        # learnt, in order to balance the two data sets. The code below ensures
        # that if class 1 is being learnt vs the rest, the number of the rest =
        # the number of class 1, evenly split across all other classes
        length_of_state_samples = np.shape(statei_values[state-1])[1]
        
        # Number of samples required from each of the other states:
        length_per_other_state = np.floor(length_of_state_samples/(number_of_states-1))
        
        #If the length of the main class / (num states - 1) >
        #length(shortest other class), then only select
        #length(shortect other class) from the other states,
        #and (3* length) for main class
        min_length_other_class = np.inf
        
        for other_state in range(1,number_of_states+1):
            samples_in_other_state = np.shape(statei_values[other_state-1])[1]
            
            if(other_state != state):
                min_length_other_class = min([min_length_other_class, samples_in_other_state])
        
        #This means there aren't enough samples in one of the
        #states to match the length of the main class being
        #trained:
        if( length_per_other_state > min_length_other_class):
            length_per_other_state = min_length_other_class
        
        training_data = np.empty(2, dtype=object)
        for other_state in range(1,number_of_states+1):
            samples_in_other_state = np.shape(statei_values[other_state-1])[1]
                    
            if(other_state == state):
                #Make sure you only choose (n-1)*3 *
                #length_per_other_state samples for the main
                #state, to ensure that the sets are balanced:
                indices = np.random.permutation(samples_in_other_state)[0:length_per_other_state*(number_of_states-1)]
                training_data[0] = statei_values[other_state-1][:,indices]
            else:
                indices = np.random.permutation(samples_in_other_state)[0:length_per_other_state]
                state_data = statei_values[other_state-1][:,indices]
                if (training_data[1] == None):
                    training_data[1] = state_data
                else:
                    training_data[1] = np.hstack((training_data[1],state_data))
                # state_data = statei_values[other_state-1][indices][:]
                # training_data[1] = np.concatenate(training_data[1], state_data)
                # arr = []
                # for i in indices:
                #     if (len(arr)==0):
                #         arr = statei_values[other_state-1][i]
                #     else:
                #         arr = np.hstack((arr, statei_values[other_state-1][i]))
                # training_data[1] = arr
        
        # Label all the data:
        # print training_data[0]
        labels = np.ones(np.shape(training_data[0])[1] + np.shape(training_data[1])[1])
        labels[0:np.shape(training_data[1])[1]] = 2

        # Train the logisitic regression model for this state:
        all_data = np.hstack((training_data[0],training_data[1]))

        # labels = np.reshape(labels,(len(labels),1))
        # all_data = np.reshape(all_data,(len(all_data),1))

        # print "yo", np.shape(all_data), all_data
        B = mnrfit.mnrfit(all_data, np.array([labels]))
        # print np.shape(all_data), all_data
        # print np.shape(labels), labels
        # print np.shape()
        # learner = LogisticRegressionCV(multi_class='multinomial',solver ='lbfgs')
        # learner.fit(all_data,labels)

        B_matrix[state-1] = B

    return B_matrix, pi_vector, total_obs_distribution


if __name__ == '__main__':
    import scipy.io
    state_observation_values = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/state_observation_values.mat',struct_as_record=False)
    state_observation_values = state_observation_values['state_observation_values'] 
    for v in range(0,len(state_observation_values)):
        for i in range(0, len(state_observation_values[v])):
            state_observation_values[v][i] = np.transpose(state_observation_values[v][i])

    actual_B_matrix, actual_pi_vector, actual_total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values)

    # desired_B_matrix = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/B_matrix.mat',struct_as_record=False) ## numpy.ndarray
    # desired_B_matrix = desired_B_matrix['B_matrix']
    # print desired_B_matrix

    # random in code, not sure if can be tested

    print "trainBandPiMatricesSpringer.py has been tested successfully"
