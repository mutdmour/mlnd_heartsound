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

    statei_values = np.array([])#np.empty((number_of_states,1),dtype=object)

    # print np.shape(state_observation_values)[0]
    for PCGi in range(0,np.shape(state_observation_values)[0]):
        vals = np.array([])
        print "yo", PCGi
        for statei in range(0,number_of_states):
            print "statei", statei
            # print '->', np.shape(state_observation_values[PCGi])
            vals = np.concatenate((vals,state_observation_values[:][PCGi]))
            # print '->', np.shape(vals)
            # print np.shape(statei_values[statei])
            # print np.shape(state_observation_values[PCGi])
            # if (len(statei_values[statei]) == 0):
                # statei_values[statei] = state_observation_values[PCGi]
            # else:
                # statei_values[statei] = np.append(statei_values[statei],state_observation_values[PCGi])
            # np.append(statei_values[statei],state_observation_values[PCGi])
            # statei_values[statei] = np.concatenate((statei_values[statei],state_observation_values[PCGi]))
        print "vals", np.shape(vals)
        statei_values = np.concatenate((statei_values,vals))
    print np.shape(statei_values)

    # In order to use Bayes' formula with the logistic regression derived
    # probabilities, we need to get the probability of seeing a specific
    # observation in the total training data set. This is the
    # 'total_observation_sequence', and the mean and covariance for each state
    # is found:
    total_observation_sequence = np.concatenate((statei_values[0], statei_values[1], statei_values[2], statei_values[3])) #xxx simplify
    total_obs_distribution = np.empty(2)
    total_obs_distribution[0] = np.mean(total_observation_sequence)
    total_obs_distribution[1] = np.cov(total_observation_sequence)


    for state in range(0,number_of_states):
        
        # Randomly select indices of samples from the other states not being 
        # learnt, in order to balance the two data sets. The code below ensures
        # that if class 1 is being learnt vs the rest, the number of the rest =
        # the number of class 1, evenly split across all other classes
        length_of_state_samples = len(statei_values[state])
        
        # Number of samples required from each of the other states:
        length_per_other_state = np.floor(length_of_state_samples/(number_of_states-1))
        
        #If the length of the main class / (num states - 1) >
        #length(shortest other class), then only select
        #length(shortect other class) from the other states,
        #and (3* length) for main class
        min_length_other_class = np.inf
        
        for other_state in range(0,number_of_states):
            samples_in_other_state = len(statei_values[other_state])
            
            if(other_state != state):
                min_length_other_class = min([min_length_other_class, samples_in_other_state])
        
        #This means there aren't enough samples in one of the
        #states to match the length of the main class being
        #trained:
        if( length_per_other_state > min_length_other_class):
            length_per_other_state = min_length_other_class
        
        training_data = np.empty(2)
        
        for other_state in range(0,number_of_states):
            samples_in_other_state = len(statei_values[other_state])
                    
            if(other_state == state):
                #Make sure you only choose (n-1)*3 *
                #length_per_other_state samples for the main
                #state, to ensure that the sets are balanced:
                indices = np.random.perumutation(samples_in_other_state,length_per_other_state*(number_of_states-1))
                training_data[0] = statei_values[other_state][indices][:]
            else:
                indices = np.random.perumutation(samples_in_other_state,length_per_other_state)
                state_data = statei_values[other_state][indices][:]
                training_data[1] = np.concatenate(training_data[1], state_data)
        
        # Label all the data:
        labels = np.ones(len(training_data[0]) + len(training_data[1]))
        labels[0:len(training_data[1])] = 2
        
        # Train the logisitic regression model for this state:
        all_data = np.concatenate(training_data[0],training_data[1]) #xxx not sure
        learner = LogisticRegression()
        B = learner.fit(all_data,labels)

        B_matrix[state] = B

    return B_matrix, pi_vector, total_obs_distribution


if __name__ == '__main__':
    # import scipy.io
    # state_observation_values = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/state_observation_values.mat',struct_as_record=False)
    # state_observation_values = np.transpose(state_observation_values['state_observation_values'])
    # # print np.shape(state_observation_values)
    # print state_observation_values[0:10][0]
    # res_B_matrix, res_pi_vector, res_total_obs_distribution = trainBandPiMatricesSpringer(state_observation_values)
    # B_matrix = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/B_matrix.mat',struct_as_record=False) ## numpy.ndarray
    # B_matrix = B_matrix['B_matrix'][0]
    # print np.shape(state_observation_values)
    # print B_matrix
    pass