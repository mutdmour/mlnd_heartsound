# function [delta, psi, qt] = viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs, figures)
#
# This function calculates the delta, psi and qt matrices associated with
# the Viterbi decoding algorithm from:
# L. R. Rabiner, "A tutorial on hidden Markov models and selected
# applications in speech recognition," Proc. IEEE, vol. 77, no. 2, pp.
# 257-286, Feb. 1989.
# using equations 32a - 35, and equations 68 - 69 to include duration
# dependancy of the states.
#
# This decoding is performed after the observation probabilities have been
# derived from the logistic regression model of Springer et al:
# D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
# Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
#
# Further, this function is extended to allow the duration distributions to extend
# past the beginning and end of the sequence. Without this, the label
# sequence has to start and stop with an "entire" state duration being
# fulfilled. This extension takes away that requirement, by allowing the
# duration distributions to extend past the beginning and end, but only
# considering the observations within the sequence for emission probability
# estimation. More detail can be found in the publication by Springer et
# al., mentioned above.
#
## Inputs:
# observation_sequence: The observed features
# pi_vector: the array of initial state probabilities, dervived from
# "trainSpringerSegmentationAlgorithm".
# b_matrix: the observation probabilities, dervived from
# "trainSpringerSegmentationAlgorithm".
# heartrate: the heart rate of the PCG, extracted using
# "getHeartRateSchmidt"
# systolic_time: the duration of systole, extracted using
# "getHeartRateSchmidt"
# Fs: the sampling frequency of the observation_sequence
# figures: optional boolean variable to show figures
#
## Outputs:
# logistic_regression_B_matrix:
# pi_vector:
# total_obs_distribution:
# As Springer et al's algorithm is a duration dependant HMM, there is no
# need to calculate the A_matrix, as the transition between states is only
# dependant on the state durations.

# function [delta, psi, qt] = 
import default_Springer_HSMM_options
import get_duration_distributions as gdd
import numpy as np
from mnrval import mnrval
from mvnpdf import mvnpdf
import sys

def viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs,figures=False):
    Fs = float(Fs)
    heartrate = float(heartrate)

    ## Preliminary
    springer_options = default_Springer_HSMM_options.default_Springer_HSMM_options()

    T = np.shape(observation_sequence)[1]
    N = 4 # Number of states

    # Setting the maximum duration of a single state. This is set to an entire
    # heart cycle:
    max_duration_D = np.round((1*(60/heartrate))*Fs)

    #Initialising the variables that are needed to find the optimal state path along
    #the observation sequence.
    #delta_t(j), as defined on page 264 of Rabiner, is the best score (highest
    #probability) along a single path, at time t, which accounts for the first
    #t observations and ends in State s_j. In this case, the length of the
    #matrix is extended by max_duration_D samples, in order to allow the use
    #of the extended Viterbi algortithm:
    delta = np.ones((N,T + max_duration_D-1),dtype='float16')*(-np.inf)

    #The argument that maximises the transition between states (this is
    #basically the previous state that had the highest transition probability
    #to the current state) is tracked using the psi variable.
    psi = np.zeros((N,T+ max_duration_D-1))

    #An additional variable, that is not included on page 264 or Rabiner, is
    #the state duration that maximises the delta variable. This is essential
    #for the duration dependant HMM.
    psi_duration = np.zeros((N,T + max_duration_D-1))

    ## Setting up observation probs
    observation_probs = np.zeros((N,T))

    for n in range(0,N):
        
        #MLR gives P(state|obs)
        #Therefore, need Bayes to get P(o|state)
        #P(o|state) = P(state|obs) * P(obs) / P(states)
        #Where p(obs) is derived from a MVN distribution from all
        #obserbations, and p(states) is taken from the pi_vector:
        pihat = mnrval(b_matrix[n],observation_sequence)
        
        for t in range(0,T):
            
            Po_correction = mvnpdf(observation_sequence[:,t],total_obs_distribution[0],total_obs_distribution[1])
            
            #When saving the coefficients from the logistic
            #regression, it orders them P(class 1) then P(class 2). When
            #training, I label the classes as 0 and 1, so the
            #correct probability would be pihat(2).
            r  = (pihat[1][t]*Po_correction)/pi_vector[n]
            observation_probs[n][t] = r

    ## Setting up state duration probabilities, using Gaussian distributions:
    [d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole] = gdd.get_duration_distributions(heartrate,systolic_time)

    duration_probs = np.zeros((N,3*Fs))
    duration_sum = np.zeros(N)
    realmin = sys.float_info.min
    max_duration_D = int(max_duration_D);


    # TODO no need to loop through each point, if only certain ranges are used
    for state_j in range(0,N):
        for d in range(0, max_duration_D):
            if(state_j == 0):
                duration_probs[state_j][d] = mvnpdf(d+1,d_distributions[state_j,0],d_distributions[state_j,1])
                
                if(d+1 < min_S1 or d+1 > max_S1):
                    duration_probs[state_j][d]= realmin

            elif(state_j==2):
                duration_probs[state_j][d] = mvnpdf(d+1,d_distributions[state_j,0],d_distributions[state_j,1])
                
                if(d+1 < min_S2 or d+1 > max_S2):
                    duration_probs[state_j][d]= realmin
                
            elif(state_j==1):
                
                duration_probs[state_j][d] = mvnpdf(d+1,d_distributions[state_j,0],d_distributions[state_j,1])
                
                if(d+1 < min_systole or d+1 > max_systole):
                    duration_probs[state_j][d]= realmin
                
            elif (state_j==3):
                
                duration_probs[state_j][d] = mvnpdf(d+1,d_distributions[state_j,0],d_distributions[state_j,1])
                
                if(d+1 < min_diastole or d+1 > max_diastole):
                    duration_probs[state_j][d]= realmin
        duration_sum[state_j] = np.sum(duration_probs[state_j][:])

    if(len(duration_probs)>3*Fs):
        duration_probs[:][(3*Fs+1):] = []

    # if(figures):
    #     figure('Name', 'Duration probabilities')
    #     plot(duration_probs(1,:)./ duration_sum(1),'Linewidth',2)
    #     hold on
    #     plot(duration_probs(2,:)./ duration_sum(2),'r','Linewidth',2)
    #     hold on
    #     plot(duration_probs(3,:)./ duration_sum(3),'g','Linewidth',2)
    #     hold on
    #     plot(duration_probs(4,:)./ duration_sum(4),'k','Linewidth',2)
    #     hold on
    #     legend('S1 Duration','Systolic Duration','S2 Duration','Diastolic Duration')
    #     pause()


    # ## Perform the actual Viterbi Recursion:
    qt = np.zeros(np.shape(delta)[1])
    ## Initialisation Step

    #Equation 32a and 69a, but leave out the probability of being in
    #state i for only 1 sample, as the state could have started before time t =
    #0.
    delta[:,0] = np.log(pi_vector) + np.log(observation_probs[:,0]) #first value is the probability of intially being in each state * probability of observation 1 coming from each state

    #Equation 32b
    psi[:,0] = -2


    # The state duration probabilities are now used.
    #Change the a_matrix to have zeros along the diagonal, therefore, only
    #relying on the duration probabilities and observation probabilities to
    #influence change in states:
    #This would only be valid in sequences where the transition between states
    #follows a distinct order.
    a_matrix = [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    ## Run the core Viterbi algorith

    # if (springer_options.use_mex):
    #
    # #     ## Run Mex code
    # #     # Ensure you have run the mex viterbi_PhysChallenge.c code on the
    # #     # native machine before running this:
    # #     # [delta, psi, psi_duration] = viterbi_Springer(N,T,a_matrix,max_duration_D,delta,observation_probs,duration_probs,psi, duration_sum)
    #
    #     print "Mex is not used here, as it's a matlab function"
    #
    # else:
        
    ## Recursion

    ## The Extended Viterbi algorithm:

    #Equations 33a and 33b and 69a, b, c etc:
    #again, ommitting the p(d), as state could have started before t = 1

    # This implementation extends the standard implementation of the
    # duration-dependant Viterbi algorithm by allowing the durations to
    # extend beyond the start and end of the time series, thereby allowing
    # states to "start" and "stop" outside of the recorded signal. This
    # addresses the issue of partial states at the beginning and end of the
    # signal being labelled as the incorrect state. For instance, a
    # short-duration diastole at the beginning of a signal looks a lot like
    # systole, and can lead to labelling errors.

    # t spans input 2 to T + max_duration_D:

    for t in range(1,T+ max_duration_D -1):
        for j in range(0,N):
            for d in range(0,max_duration_D):
                #The start of the analysis window, which is the current time
                #step, minus d (the time horizon we are currently looking back),
                #plus 1. The analysis window can be seen to be starting one
                #step back each time the variable d is increased.
                # This is clamped to 1 if extending past the start of the
                # record, and T-1 is extending past the end of the record:
                start_t = t - d
                if(start_t < 1):
                    start_t = 1
                if(start_t > T-1):
                    start_t = T-1

                #The end of the analysis window, which is the current time
                #step, unless the time has gone past T, the end of the record, in
                #which case it is truncated to T. This allows the analysis
                #window to extend past the end of the record, so that the
                #timing durations of the states do not have to "end" at the end
                #of the record.
                end_t = t+1
                if(t>T):
                    end_t = T

                #Find the max_delta and index of that from the previous step
                #and the transition to the current step:
                #This is the first half of the expression of equation 33a from
                #Rabiner:
                arr_delta = delta[:,start_t-1] + np.transpose(np.log(a_matrix[:][j]))
                max_index = np.argmax(arr_delta)
                max_delta = arr_delta[max_index]

                #Find the normalised probabilities of the observations over the
                #analysis window:
                probs = np.prod(observation_probs[j,start_t-1:end_t])


                #Find the normalised probabilities of the observations at only
                #the time point at the start of the time window:
                if(probs ==0):
                    probs = realmin
                emission_probs = np.log(probs)


                #Keep a running total of the emmission probabilities as the
                #start point of the time window is moved back one step at a
                #time. This is the probability of seeing all the observations
                #in the analysis window in state j:
                if(emission_probs == 0 or np.isnan(emission_probs)):
                    emission_probs =realmin

                #Find the total probability of transitioning from the last
                #state to this one, with the observations and being in the same
                #state for the analysis window. This is the duration-dependant
                #variation of equation 33a from Rabiner:
                #                 fprintf('log((duration_probs(j,d)./duration_sum(j))):#d\n',log((duration_probs(j,d)./duration_sum(j))))
                delta_temp = max_delta + (emission_probs)+ np.log((duration_probs[j][d]/duration_sum[j]))


                #Unlike equation 33a from Rabiner, the maximum delta could come
                #from multiple d values, or from multiple size of the analysis
                #window. Therefore, only keep the maximum delta value over the
                #entire analysis window:
                #If this probability is greater than the last greatest,
                #update the delta matrix and the time duration variable:

                if(delta_temp>delta[j,t]):
                    delta[j,t] = delta_temp
                    psi[j,t] = max_index
                    psi_duration[j,t] = d

    ## Termination

    # For the extended case, need to find max prob after end of actual
    # sequence:

    # Find just the delta after the end of the actual signal
    temp_delta = delta[:,T:]
    #Find the maximum value in this section, and which state it is in:
    idx = np.argmax(temp_delta)
    pos = np.unravel_index(idx, np.shape(temp_delta))
    pos = pos[0]

    # Change this position to the real position in delta matrix:
    pos = pos+T

    #1) Find the last most probable state
    #2) From the psi matrix, find the most likely preceding state
    #3) Find the duration of the last state from the psi_duration matrix
    #4) From the onset to the offset of this state, set to the most likely state
    #5) Repeat steps 2 - 5 until reached the beginning of the signal


    #The initial steps 1-4 are equation 34b in Rabiner. 1) finds P*, the most
    #likely last state in the sequence, 2) finds the state that precedes the
    #last most likely state, 3) finds the onset in time of the last state
    #(included due to the duration-dependancy) and 4) sets the most likely last
    #state to the q_t variable.

    #1)
    state = np.argmax(delta[:,pos])

    #2)
    offset = pos
    preceding_state = psi[state,offset]

    #3)
    # state_duration = psi_duration(offset, state)
    onset = offset - psi_duration[state,offset]

    #4)
    qt[onset:offset+1] = state

    #The state is then updated to the preceding state, found above, which must
    #end when the last most likely state started in the observation sequence:
    state = preceding_state

    count = 0
    #While the onset of the state is larger than the maximum duration
    #specified:
    while (onset > 1):
        
        #2)
        offset = onset-1
        #     offset_array(offset,1) = inf
        preceding_state = psi[state,offset]
        #     offset_array(offset,2) = preceding_state
        
        #3)
        #     state_duration = psi_duration(offset, state)
        onset = offset - psi_duration[state,offset]
        
        #4)
        #     offset_array(onset:offset,3) = state
        
        if(onset<1):
            onset = 0
        qt[onset:offset+1] = state
        state = preceding_state
        count = count +1
        
        if(count> 1000):
            break

    qt = qt[0:T]+1
    psi = psi + 1

    return delta, psi, qt


if __name__ == '__main__':
    heartrate = 51.413881748071980
    systolic_time = 0.391000000000000
    Fs = 50
    pi_vector = np.array([.25]*4)

    import scipy.io
    observation_sequence = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/observation_sequence.mat',struct_as_record=False)
    observation_sequence = observation_sequence['observation_sequence']
    observation_sequence = np.transpose(observation_sequence) ## PCG Features

    b_matrix = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/b_matrix.mat',struct_as_record=False)
    b_matrix = b_matrix['b_matrix'][0]
    b_matrix = map(lambda x: np.reshape(x, np.shape(x)[0]), b_matrix)
    # b_matrix = np.transpose(b_matrix)
    # b_matrix = [np.transpose(b[0]) for b in b_matrix]
    # b_matrix = np.reshape(b_matrix, (np.shape(b_matrix)[0],np.shape(b_matrix)[2]))

    x = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/total_obs_distribution.mat',struct_as_record=False)
    x = x['total_obs_distribution']
    total_obs_distribution = np.empty(2, dtype=object)
    total_obs_distribution[0] = x[0][0][0]
    total_obs_distribution[1] = np.transpose(x[1][0])

    [delta, psi, qt] = viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs)
    # print np.shape(total_obs_distribution), total_obs_distribution

    x = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/delta.mat',struct_as_record=False)
    desired_delta = np.transpose(x['delta'])

    x = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/psi.mat',struct_as_record=False)
    desired_psi = np.transpose(x['psi'])

    x = scipy.io.loadmat('./test_data/viterbiDecodePCG_Springer/qt.mat',struct_as_record=False)
    desired_qt = x['qt'][0]

    np.testing.assert_allclose(delta, desired_delta, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(psi, desired_psi, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(qt, desired_qt, rtol=1e-3, atol=1e-3)


    print "viterbiDecodePCG_Springer.py has been tested successfully"
