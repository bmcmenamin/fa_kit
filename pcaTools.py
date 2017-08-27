import numpy as np
import scipy as sp
from scipy import linalg as spLin

import matplotlib.pyplot as plt

def fillNanPerCol(X):
    for i in range(X.shape[1]):
        nanIdx = np.isnan(X[:,i])
        X[nanIdx, i] = np.mean(X[~nanIdx,i])
    return X

##
## SVD-based PCA decomposition
##

def pca(X):
    U, S, V = np.linalg.svd(X, full_matrices=False)
    D = S**2
    return U, D

##
## Covariance matrix based decomposition
##  may be faster if number datapoints >>> number of dimensions

def pca_covmat(X, denom=None):
    
    if denom is not None:
         D, V = spLin.eigh(a=X, b=denom)
         V = np.real(V)
         D = np.real(D)
    else:
        D, V = np.linalg.eigh(X)

    newOrder = np.argsort(D)[::-1]
    V = V[:, newOrder]
    D = D[newOrder]
    return V, D

##
## Broken stick distro
##

def brokenStick(n):
    _tmp = np.array([1.0/i for i in range(1,n+1)])
    bs = np.sum(_tmp) - np.insert(np.cumsum(_tmp)[:-1], 0, 0)
    bs /= n
    return bs


def wMoments(x, w):
    wMean = np.average(x, weights=w)
    wVar = np.average((x-wMean)**2, weights=w)
    wStd = np.sqrt(wVar)
    return wMean, wStd

def getBSfit(d, bs, w):
    
    d = np.log(d + 1)
    bs = np.log(bs + 1)

    wMu_d, wSd_d = wMoments(d, w)
    wMu_bs, wSd_bs = wMoments(bs, w)
    scale = wSd_d / wSd_bs
    shift = wMu_d - scale*wMu_bs

    bs2 = scale*bs + shift
    wErr = w.T.dot((d - bs2)**2)

    bs2 = np.exp(bs2) - 1   
    return bs2, wErr

def rescale_brokenStick(d):

    #issorted = all(a >= b for a,b in zip(d, d[1:]))
    #assert issorted
    
    if any(d<0):
        dPos = d[d>=0]
        dNeg = d[d<0]
        
        bsPos = rescale_brokenStick(dPos)
        bsNeg = rescale_brokenStick(-dNeg[::-1])
        newBS = np.array(list(bsPos) + list(-bsNeg[::-1]))
    else:
        bs = brokenStick(len(d))
        bsIndex = np.array(range(len(d)))
    
        _w = np.log(bsIndex + 1)
        newBS, mwse = getBSfit(d, bs, _w)

    return newBS 


##
## Extract factors from covariance/correlation matrix
## using communality to iteratively deweight variables
## not captured in top K components
##

def commweightedExtraction(cMat, toKeep, denom=None, verbose=False, maxIter=100):
    origDiag = np.diag(cMat)
    V, D = pca_covmat(cMat, denom=denom)
    Dold = np.copy(D)
    
    tol = 1e-4
    err = np.inf
    iter = 0
    while err > tol:
        communality = np.sum(V[:,toKeep]**2, axis=1)
        np.fill_diagonal(cMat, communality*origDiag)
        #if denom is not None:
        #    np.fill_diagonal(denom, communality*np.diag(denom))
        V, D = pca_covmat(cMat, denom=denom)
        err = np.mean((Dold-D)**2)
        
        if verbose:
            print(err)
        Dold = np.copy(D)
        
        iter += 1
        if iter > maxIter:
            print('hitting iteration limit')
            break
    return V[:,toKeep], D[toKeep]

##
## Performs varimax or quartimax rotation
##

def ortho_rotation(lam, method='varimax', gamma=None, itermax=5000):
    """
    https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0
    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var:
            break
        var = var_new

    # apply rotation to data
    rotLam = lam.dot(R)

    # Flip to make all absmax on the positive end
    toFlip = np.abs(np.min(rotLam, axis=0)) > np.abs(np.max(rotLam, axis=0))
    rotLam[:,toFlip] = -rotLam[:,toFlip]
    rotLam /= np.sqrt(np.mean(rotLam**2, axis=0, keepdims=True))
    return rotLam



##
## threshold components to find strongest loadings
##

def thresholdComps(Vrot):
    Vthresh = np.copy(Vrot)
    cutoffs = 0.75 * np.max(np.abs(Vthresh), axis=0, keepdims=True)
    Vthresh = np.sign(Vthresh) * (np.abs(Vthresh) > cutoffs)
    return Vthresh


 
def fullPCApipeline(X, testX=None, genEigDenomMat=None,
                    plotScree=False, overrideDimension=None, usePAF=True,
                    normBeforeRot=True, rotType='varimax'):

    if (X.shape[0] == X.shape[1]) and np.allclose(X.T, X):
        covarMat = X.copy()
    else:
        covarMat = X.T.dot(X)

    V, D = pca_covmat(covarMat, denom=genEigDenomMat)
    
    # If a test dataset exists, replace eigvalues with xval %age variance explained
    if testX is not None:
        covarMat_test = testX.T.dot(testX)
        Vtest, Dtest = pca_covmat(covarMat_test, denom=genEigDenomMat)
        lodimVtest = V.T.dot(Vtest)
        newD = [0]
        for i in range(V.shape[1]):
            Vrecon = V[:,:(i+1)].dot(lodimVtest[:(i+1),:])
            reconD = np.sum(np.sum(Vrecon**2, axis=0)*Dtest)
            newD.append(reconD)
        D = np.diff(np.array(newD))
    
    # Estimate the number of components to keep by comparing
    # distribution of eigenvalues against the broken stick distribution
    bsDistro = rescale_brokenStick(D)
    
    toKeep = D >= bsDistro
    toKeep = D > D[np.where(~toKeep)[0].min()]

    print("Broken stick distro says to use {} components".format(np.sum(toKeep)))
    if all(D>0):
        print("  this would explain {:.1f}% of total variance".format(100*np.sum(D[toKeep])/np.sum(D)))
    
    if plotScree:
        maxPlot = np.min([40, len(D)])

        fig, ax = plt.subplots(1,1, figsize=[10,5])
        scaleConst = np.sum(np.abs(D))
        ax.plot(range(1,maxPlot+1), D[:maxPlot] / scaleConst, '-ok')
        ax.plot(range(1,maxPlot+1), bsDistro[:maxPlot] / scaleConst, '--r')

        ax.set_yscale('log')

        ax.set_xlabel('Component #')
        ax.set_ylabel('LOG eigenvalues')
        ax.legend(['Observed egienvalues','Null eigenvalues (broken-stick)'])
    
    if overrideDimension is not None:
        toKeep = np.array([i<overrideDimension for i in range(len(D))])
        print("MANUAL OVERRIDE is to use {} components".format(np.sum(toKeep)))
        if all(D>0):
            print("  this would explain {:.1f}% of total variance".format(100*np.sum(D[toKeep])/np.sum(D)))
       
    # Extract components using Principle Axis Factoring (PAF) or PCA
    if usePAF:
        Vext, Dext = commweightedExtraction(covarMat, toKeep, denom=genEigDenomMat)
        Sext = np.sqrt(Dext)
    else:
        Vext = V[:,toKeep]
        Sext = np.sqrt(D[toKeep])
    
    # Varimax rotate components
    if not normBeforeRot:
        Vext = Vext*Sext
    Vrot = ortho_rotation(Vext, method=rotType)
    typePerUser = Vrot.T.dot(X.T)    

    # Sort in order of decreasing importance
    pctVar = np.mean(typePerUser**2,axis=1)
    newOrder = np.argsort(-pctVar)
    Vrot = Vrot[:,newOrder]
    typePerUser = typePerUser[newOrder, :]
    return Vrot, typePerUser

