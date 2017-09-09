import numpy as np

def mvnpdf(X, Mu, Sigma):
    # Get size of data.  Column vectors provisionally interpreted as multiple scalar data.
    if (len(np.shape(X)) > 0):
        d = np.shape(X)[0]
    else:
        d = 1
        Sigma = np.array([[Sigma]])
        X = np.array([[X]])
        Mu = np.array([[Mu]])


    X0 = X - Mu
    R = np.linalg.cholesky(Sigma)
    xRinv = np.linalg.lstsq(R,X0)[0]
    logSqrtDetSigma = sum(np.log(np.diag(R)))

    # The quadratic form is the inner products of the standardized data
    quadform = np.sum(xRinv**2,axis=0)

    return np.exp(-0.5*quadform - logSqrtDetSigma - d*np.log(2*np.pi)/2)

if __name__ == '__main__':
    import scipy.io

    x = scipy.io.loadmat('./test_data/mvnpdf/X.mat', struct_as_record=False)
    X = x['X'][0]

    x = scipy.io.loadmat('./test_data/mvnpdf/Mu.mat', struct_as_record=False)
    Mu = x['Mu'][0]

    x = scipy.io.loadmat('./test_data/mvnpdf/Sigma.mat', struct_as_record=False)
    x = x['Sigma']
    Sigma = np.transpose(x)

    actual = mvnpdf(X, Mu, Sigma)

    desired = 0.108173095701383
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)


    # another
    X = 1
    Mu = 6
    Sigma = 1
    actual = mvnpdf(X,Mu,Sigma)
    desired = 1.486719514734299e-06
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

    print "mnrval.py has been tested successfully"