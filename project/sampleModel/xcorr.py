import numpy as np
#autocorrelation
def xcorr(x):
    # Perform all auto- and cross-correlations of the columns of x.
    defaultMaxlag = np.shape(x)[0] - 1
    # If it has been supplied, fetch maxlag from varargin and validate it.
    # maxlag = fetchMaxlag(maxlagIdx,defaultMaxlag,varargin{:})
    maxlag = defaultMaxlag
    # Peform the auto- and cross-correlations.
    c1 = autocorr(x,maxlag)
    # Scale the output, if requested.
    c = scaleXcorr(c1)

    # Pad the output with zeros if maxlag > defaultMaxlag.
    # c = padOutput(c1,maxlag,defaultMaxlag) ##

    return c

def autocorr(x, maxlag):
    # Compute all possible auto- and cross-correlations of the columns of a
    # matrix input x. Output is clipped based on maxlag but not padded when
    # maxlag >= size(x,1).
    m = np.shape(x)[0]
    mxl = min(maxlag,m - 1)
    ceilLog2 = nextpow2(2*m - 1)
    m2 = 2**ceilLog2

    # Autocorrelation of a column vector.
    X = np.fft.fft(x,n=m2)
    Cr = abs(X)**2
    c1 = np.real(np.fft.ifft(Cr))

    # Keep only the lags we want and move negative lags before positive
    # lags.
    c = np.concatenate((c1[m2 - mxl + np.arange(0,mxl)], c1[np.arange(0,mxl+1)]))
    return c

def scaleXcorr(c):
    # Autocorrelation of a vector.Normalize by c[0].
    mid = (np.shape(c)[0] + 1) / 2 # row corresponding to zero lag.
    c = c / c[mid]
    return c

def nextpow2(n):
    p = np.log2(np.abs(n))
    if (not float(p).is_integer()):
        p += 1
    p = int(np.floor(p))
    return p


if __name__ == '__main__':
    import scipy.io

    x = scipy.io.loadmat('./test_data/xcorr/x.mat', struct_as_record=False)
    x = x['x']
    x = np.reshape(x, np.shape(x)[0])

    actual = xcorr(x)

    c = scipy.io.loadmat('./test_data/xcorr/c.mat', struct_as_record=False)
    c = c['c']
    desired = np.reshape(c, np.shape(c)[0])

    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)
    print "xcorr.py has been tested successfully"