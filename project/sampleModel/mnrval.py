import numpy as np

def mnrval(beta,x):
    # Validate the size of beta, and compute the linear predictors.
    n = np.shape(x)[1]
    pstar = np.shape(beta)[0]
    # k = len(beta) + 1 #2
    eta = (np.zeros(n)+beta[0]) + np.dot(np.transpose(x),beta[1:len(beta)])
    pi = np.array([np.exp(eta-eta), np.exp(-eta)]) # rescale so max probability is 1
    sum_pi = np.sum(pi,0)
    pi = pi / np.array([sum_pi, sum_pi])     # renormalize for real probabilities
    return pi

if __name__ == '__main__':
    import scipy.io

    x = scipy.io.loadmat('./test_data/mnrval/beta.mat', struct_as_record=False)
    x = x['beta']
    beta = np.reshape(x,(np.size(x)))

    x = scipy.io.loadmat('./test_data/mnrval/x.mat', struct_as_record=False)
    x = x['x']
    x = np.transpose(x)

    actual = mnrval(beta, x)

    x = scipy.io.loadmat('./test_data/mnrval/pred.mat', struct_as_record=False)
    x = x['pred']
    desired = np.transpose(x)

    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)
    print "mnrval.py has been tested successfully"