import numpy as np
def mnrfit(x,y):
    iterlim = 100
    tolpos = (np.finfo(float).eps)**(3./4)
    model = 'nominal'
    parallel = False
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
    nominalFit(x,y,m,pi,n,k,p,pstar,parallel,iterlim,tolpos)
    
    # return b



# def constrain(x,lower,upper)
#     # Constrain between upper and lower limits, and do not ignore NaN
#     x(x<lower) = lower
#     x(x>upper) = upper
#     return x

# #function [b,XWX,pi] = 
def nominalFit(x,y,m,pi,n,k,p,pstar,parallel,iterLim,tolpos):

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
    m = np.transpose(m)
    while (iter <= iterLim):
        iter = iter + 1
        mu = m * pi
        pi = np.transpose(pi)
        mu = np.transpose(mu)

        # Updated the coefficient estimates.
        b_old = b
        XWX = 0
        XWZ = 0
        # print np.shape(pi), mu
        for i in range(0,n):
            a = np.array([mu[i]])
            b = np.array([pi[i]])
            W = np.diag(mu[i]) - np.transpose(a).dot(b)

            # # Adjusted dependent variate
            Z = eta[i].dot(W) + (y[i] - mu[i])
            if (p > 0): # parallel models with p>0 have been weeded out
                xstar = np.append(np.array([1]),x[i])
                # Do these computations, but more efficiently
                # XWX = XWX + kron(W(1:k-1,1:k-1), xstar'*xstar)
                # XWZ = XWZ + kron(Z(1:k-1)', xstar')
                W_mat = np.array([kron1[0]]*len(kron1[0])) * W[0][0]
                xstar_mat = np.array([xstar[0:len(kron2[0])]])
                xstar_mat = np.transpose(xstar_mat) * xstar_mat
                XWX = XWX + W_mat.dot(xstar_mat)

                XWZ = XWZ + np.transpose(Z[kron1]) * np.transpose(xstar[kron2])
                print XWZ
                break
            else:
                print "warning: p < 0 "
                # XWX = XWX + W[1:k-1][1:k-1]
                # XWZ = XWZ + np.transpose(Z[1:k-1])
        break
        # b = XWX \ XWZ

#         # Update the linear predictors.
#         eta_old = eta
#         if parallel # parallel models with p>0 have been simplified already
#             eta = repmat(np.transpose(b),n,1)
#         else
#             b = reshape(b,pstar,k-1)
#             if p > 0
#                 eta = b(1,:) + x*b(2:pstar,:) # row plus full matrix
#             else
#                 eta = repmat(b,n,1)

#         eta = [eta zeros(n,1,'like',eta)]
        
#         # Check stopping conditions
#         cvgTest = abs(b-b_old) > convcrit * max(seps, abs(b_old))
#         if iter>iterLim || ~any(cvgTest(:))
#             pi = exp(eta-max(eta,[],2))   # rescale but do not constrain
#             pi = pi ./ sum(pi,2)          # make rows sum to 1
#             break

#         refine = false
#         for backstep = 0:10
#             # Update the predicted category probabilities constrain to a
#             # reasonable range to avoid problems during fitting.
#             pi = exp(constrain(eta-max(eta,[],2),lowerBnd,upperBnd))
#             pi = pi ./ sum(pi,2) # make rows sum to 1
            
#             # If all observations have positive category probabilities,
#             # we can take the step as is.
#             if all(pi(:) > tolpos)
#                 break

#             # In either of the following cases the coefficients vector will not
#             # yield the pi and eta values, but we have not converged so we will
#             # make them consistent in the next iteration. We will not test for
#             # convergence while backtracking, because we don't have coefficient
#             # estimates that give the shorter steps.
                
#             # Otherwise try a shorter step in the same direction.  eta_old is
#             # feasible, even on the first iteration.
#             elseif backstep < 10
#                 eta = eta_old + (eta - eta_old)/5
#                 refine = true

#             # If the step direction just isn't working out, force the
#             # category probabilities to be positive, and make the linear
#             # predictors compatible with that.
#             else
#                 pi = max(pi,tolpos)
#                 pi = pi ./ sum(pi,2) # make rows sum to 1
#                 eta = log(pi)
#                 refine = true
#                 break

#     # if refine && ~parallel && p>0
#     #     # The iterative reweighted least squares procedure may stall as fitted
#     #     # probabilities get close to 0 or 1. Try to refine the results with
#     #     # some iterations of fminsearch.
#     #     op = optimset('Display','none')
#     #     newb = fminsearch(@(newb)calcdev(newb,size(b),x,y),b,op)
#     #     b = reshape(newb,size(b))
#     #     eta = b(1,:) + x*b(2:pstar,:) # row plus full matrix
#     #     eta = [eta zeros(n,1,'like',eta)]
        
#     #     pi = exp(eta-max(eta,[],2))   # rescale but do not constrain
#     #     pi = pi ./ sum(pi,2)          # make rows sum to 1

#     # return b,XWX,pi

if __name__ == '__main__':
    import scipy.io
    all_data1 = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/all_data1.mat',struct_as_record=False)['all_data1']
    all_data1 = np.transpose(all_data1)
    labels1 = scipy.io.loadmat('./test_data/trainBandPiMatricesSpringer/labels1.mat',struct_as_record=False)['labels1']
    labels1 = np.transpose(labels1)
    # print np.shape(all_data1), np.shape(labels1)
    mnrfit(all_data1,labels1)
