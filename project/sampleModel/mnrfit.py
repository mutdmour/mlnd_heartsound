import numpy as np
def mnrfit(x,y):
    iterlim = 100
    tolpos = (np.finfo(float).eps)**(3./4)
    model = 'nominal'
    # parallel = False
    link = 'logit'
    estdisp = False
    
    p = np.shape(x)[0]
    k,n = np.shape(y)
    if k == 1:
        y = y[0]
        y = np.int16([y == 1, y != 1])
        k = np.shape(y)[0]
        m = np.ones(n,dtype=np.float16)
    else:
        m = sum(y)
    
    pstar = p + 1
    dfe = (n-pstar) * (k-1)
    
    # Set up initial estimates from the data themselves
    pi = (y+0.5) / (1+k/2.) # shrink raw proportions toward uniform
    # print np.shape(y)

    # [b,hess,pi] = nominalFit(x,y,m,pi,n,k,p,pstar,parallel,iterlim,tolpos)
    b = nominalFit(x,y,m,pi,n,k,p,pstar,iterlim,tolpos)
    
    return b



def constrain(x,lower,upper):
    # Constrain between upper and lower limits, and do not ignore NaN
    x[x<lower] = lower
    x[x>upper] = upper
    return x

# #function [b,XWX,pi] = 
def nominalFit(x,y,m,pi,n,k,p,pstar,iterLim,tolpos):

    kron1 = np.array([[1]*pstar]*(k-1))
    kron2 = np.array([range(0,pstar)]*(k-1))

    eta = np.log(pi)

    # Main IRLS loop
    iter = 0
    eps = np.finfo(float).eps
    seps = np.sqrt(eps) # don't depend on class
    convcrit = 1e-6
    b = 0
    # dataClass = superiorfloat(x,y)
    lowerBnd = np.log(eps)
    upperBnd = -lowerBnd
    refine = False # flag to indicate solution may need more refinement

    y = np.transpose(y)
    eta = np.transpose(eta)
    x = np.transpose(x)
    m = np.reshape(np.transpose(m),(len(m),1))
    while (iter <= iterLim):
        # print "*******"
        iter = iter + 1
        # print np.shape(m), np.shape(pi)
        mu = np.reshape(m,(1,len(m))) * pi
        # print np.shape(mu)
        pi = np.transpose(pi)
        mu = np.transpose(mu)

        # Updated the coefficient estimates.
        b_old = b
        XWX = 0
        XWZ = 0
        # print np.shape(pi), mu
        Z_orig = None
        for i in range(0,n):
            a = np.array([mu[i]])
            b = np.array([pi[i]])
            W = np.diag(mu[i]) - np.transpose(a).dot(b)

            # # Adjusted dependent variate
            Z = eta[i].dot(W) + (y[i] - mu[i])

            if (p > 0): # parallel models with p>0 have been weeded out
                xstar = np.array([np.append(np.array([1]),x[i])])

                # Do these computations, but more efficiently
                XWX = XWX + np.kron([W[0][0]], np.transpose(xstar)*xstar)
                XWZ = XWZ + np.kron(np.transpose(Z[0]), np.transpose(xstar))
            else:
                # print "warning: p < 0 "
                XWX = XWX + W[0][0]
                XWZ = XWZ + np.transpose(Z[0])
        # print XWX
        # print XWZ
        # break
        b = np.linalg.lstsq(XWX, XWZ)[0]

        # Update the linear predictors.
        eta_old = eta
        # if (parallel): # parallel models with p>0 have been simplified already
            # eta = repmat(np.transpose(b),n,1)
        # else
        b = np.reshape(b,(pstar,k-1)) #xxx did not change structure
        if p > 0:
            # print np.shape(x), np.shape(b[1:])
            # print x.dot([b[1:]])[0:10]
            eta = b[0][0] + x.dot(b[1:]) # row plus full matrix
        else:
            print "warning: p<0 eta"
            # eta = repmat(b,n,1)

        # print "yo"
        eta = np.hstack((eta,np.zeros((len(eta),1))))
        
        # Check stopping conditions
        # print convcrit, seps, np.abs(b_old)
        cvgTest = np.abs(b-b_old) > convcrit * np.maximum(seps, np.abs(b_old))

        if (iter>iterLim or not np.any(cvgTest)):
            # print np.shape(np.reshnp.max(eta,axis=1))
            pi = np.exp(np.subtract(eta,np.reshape(np.max(eta,axis=1),(len(eta),1))))   # rescale but do not constrain
            # print np.sum(pi,axis=1)
            pi = np.divide(pi,np.reshape(np.sum(pi,axis=1),(len(pi),1)))          # make rows sum to 1
            break

        refine = False
        for backstep in range(0,11):
            # Update the predicted category probabilities constrain to a
            # reasonable range to avoid problems during fitting.
            pi = np.exp(constrain(eta-np.reshape(np.max(eta,axis=1),(len(eta),1)),lowerBnd,upperBnd))
            pi = np.divide(pi,np.reshape(np.sum(pi,axis=1),(len(pi),1))) # make rows sum to 1
            
            # If all observations have positive category probabilities,
            # we can take the step as is.
            if (np.all(pi > tolpos)):
                # print "exit"
                break

            # In either of the following cases the coefficients vector will not
            # yield the pi and eta values, but we have not converged so we will
            # make them consistent in the next iteration. We will not test for
            # convergence while backtracking, because we don't have coefficient
            # estimates that give the shorter steps.
                
            # Otherwise try a shorter step in the same direction.  eta_old is
            # feasible, even on the first iteration.
            elif (backstep < 10):
                eta = eta_old + (eta - eta_old)/5
                refine = True

            # If the step direction just isn't working out, force the
            # category probabilities to be positive, and make the linear
            # predictors compatible with that.
            else:
                pi = np.max(pi,tolpos)
                pi = np.divide(pi,np.sum(pi,axis=1)) # make rows sum to 1
                eta = np.log(pi)
                refine = True
                break
        pi = np.transpose(pi)

    # if (refine and p>0):
        # The iterative reweighted least squares procedure may stall as fitted
        # probabilities get close to 0 or 1. Try to refine the results with
        # some iterations of fminsearch.
#     #     op = optimset('Display','none')
#     #     newb = fminsearch(@(newb)calcdev(newb,size(b),x,y),b,op)
#     #     b = reshape(newb,size(b))
#     #     eta = b(1,:) + x*b(2:pstar,:) # row plus full matrix
#     #     eta = [eta zeros(n,1,'like',eta)]
        
#     #     pi = exp(eta-max(eta,[],2))   # rescale but do not constrain
#     #     pi = pi ./ sum(pi,2)          # make rows sum to 1

#if iter > iterLim
    #warning(message('stats:mnrfit:IterOrEvalLimit'));
# Warning: Maximum likelihood estimation did not converge.  Iteration limit
    return b#,XWX,pi

if __name__ == '__main__':
    import scipy.io

    for i in range(1,5):
        folder = './test_data/mnrfit/'
        all_data_file = folder + 'all_data' + str(i) + '.mat'
        all_data = scipy.io.loadmat(all_data_file,struct_as_record=False)['all_data'+str(i)]
        all_data = np.transpose(all_data)

        labels_file = folder + 'labels' + str(i) + '.mat'
        labels = scipy.io.loadmat(labels_file,struct_as_record=False)['labels'+str(i)]
        labels = np.transpose(labels)

        b_file = folder + 'b' + str(i) + '.mat'
        b = scipy.io.loadmat(b_file,struct_as_record=False)['ans']
        # b = np.transpose(b)

        actual = mnrfit(all_data,labels)
        # desired = np.transpose([[0.5976, -1.7183, 0.9019, -0.0688, 0.0202]])
        np.testing.assert_allclose(actual, b, rtol=1e-5, atol=1e-5)    

    print "mnrfit.py has been tested successfully"