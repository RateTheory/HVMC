#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:13:43 2020
VMC Implementation in Python
@author: shaama
"""

import math
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from scipy.fftpack import dct, idct
import numpy.ma as ma
import copy
from matplotlib.pyplot import matshow
import matplotlib.pyplot as plt
import imageio

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def vmc(Xinit, sampmask, samples, options, Xtrue=None): 
    
    #############################################
    #OPTIONS
    #############################################
    
    #degree
    if options.d == None: 
        d = 2
    else:
        d = options.d
        
    #kernel
    if options.c == None: 
        if d < math.inf:
            c = 0 #default to homogeneous poly kernel
        else:
            c = 1 #Gaussian RBF with bandwidth 1
    else:
        c = options.c
    
    #Noise level
    if options.epsilon == None:
        epsilon = 0 #no noise, equality constraints
    else:
        epsilon = options.epsilon #noise
    
    #Max iterations
    if options.niter == None:
        niter = 5000
    else:
        niter = options.niter
        
    #Initial smoothing parameter
    if options.gamma0 == None:
        gamma0 = 1;
    else:
        gamma0 = options.gamma0
    
    #Minimum allowed gamma
    if options.gammamin == None:
        gammamin = 1e-16
    else:
        gammamin = options.gammamin
        
    #Gamma decrease factor
    #recommended: 1.001-1.100
    if options.eta == None:
        eta = 1.01;
    else:
        eta = options.eta
        
    #Schatten-p penalty value
    #p = 0: log-det penalty
    #p = 1: nuclear norm
    #p = 0.5: works well for most settings
    if options.p == None:
        p = 0.5
    else:
        p = options.p
        
    #Exit tolerance for convergence
    if options.exit_tol == None:
        exit_tol = 1.e-8
    else:
        exit_tol = options.exit_tol
    
    #options for eigendecomposition/svd
    #currently using only kernel-eig
    #will change when we expand to larger systems
    #rmax also not specified
        
    #eigtol
    if options.eigtol == None:
        eigtol = 1.e-4
    else:
        eigtol = options.eigtol
    
        
    #dct transform
    if options.dct_trans == None:
        dct_trans = False
    else:
        dct_trans = options.dct_trans
    #cost function
    #separate function cost_function
        
    #############################################
    #Initialize variables
    #############################################
    scalefac = np.sqrt(np.max(np.sum(np.abs(np.power(Xinit,2)))))
    #scalefac = np.sqrt(np.max(np.sum(np.abs(np.power(Xinit,2)), axis=0)))
    #Soften/remove mask
    X = Xinit/scalefac
    samples = samples/scalefac
    epsilon = epsilon/scalefac
    
    #Soften/remove mask
    #X.soften_mask()
    
    Xold = X
    q = 1-p/2
    
    cost = []
    error = []
    update = []
    
    if Xtrue.any():
        Xtrue = Xtrue/scalefac
        error.append(norm(X-Xtrue)/norm(Xtrue))
    
    #Create boolean mask from sampmask; True indicates sampled
    boolmask = (1-sampmask).astype(bool)
    
    #############################################
    #Iterate
    #TODO - include functionality for additional
    #sampling
    #############################################
    step = 0
    ndcount=0
    ims = []
    nds=[]

    while step<=niter:
        if dct_trans:
            #print(step)
            #1. Transform basis and rescale
            #X = fft2(X)
            #scale2 = np.sqrt(np.sum(np.abs(np.power(X,2))))
            #X = X/scale2
            X = dct(X,norm='ortho')
            #scale2 = np.sqrt(np.sum(np.abs(np.power(X,2))))
            #X = X/scale2
            
        #2. Kernel
        G = np.matmul(X.T, X)
        cmat = np.ones(np.shape(G))*c
    
        if d < math.inf: #polynomial kernel
            K = np.power(np.add(G,cmat),d) #CHECKED
            #print(K)
        else: #rbf kernel
            print("RBF Kernel not implemented, using poly")
            K = np.power(np.add(G,cmat),d)
        #print(K)
        if not check_symmetric(K):
            print('K not symmetric')
        #3. Decompose and sort
        D, V = np.linalg.eigh(K)

        ev = -np.sort(-abs(D))
        idx = np.argsort(abs(D))
        idx = idx[::-1]
        V = V[:,idx]
        
        if step==0:#initialize gamma
            if gamma0==0:
                gamma = 0.01*ev[0]
                #print('VMC set gamma0=','{:.4f}'.format(gamma))
            else:
                gamma = gamma0
        
        
        #Compute weighted matrix - 
        #Only implemented for kernel-eig right now
        evmat = np.add(ev,gamma)
        E = np.diag(np.power(evmat,-q)) 
        
        #print(evmat,evinv)
        W = np.matmul(V,np.matmul(E,V.T)) #CHECKED
        
        #Projected gradient descent step
        if d==1:
            gradX = np.matmul(X,W)
        elif d==2: 
            gradX = 2*np.matmul(X,np.multiply(W,np.add(G,cmat)))
        elif d>2 and d<math.inf:
            gradX = d*np.matmul(X,np.multiply(W,np.power(np.add(G,cmat),d-1)))
        elif d == math.inf: 
            print('Not implemented RBF yet. Set d to lower value and rerun')
            #gradX = np.matmul(X,)
        
        #Gradient step
        tau = np.power(gamma,q)
        X = X - tau*gradX
        
        if dct_trans:
            #IFFT
            #X = scale2*X
            #X = ifft2(X)
            X = idct(X,norm='ortho')
            #matshow(X-Xinit); plt.colorbar()
        
        #Constraints
        #VERY CRUDE, NEEDS WORK
        #print(sampmask,samples)
        samplist = copy.deepcopy(samples).tolist()
        if epsilon==0: #Equality constraints
            for ind1, row in enumerate(X): 
                for ind2, val in enumerate(row): 
                    if sampmask[ind1,ind2]==0:
                        X[ind1,ind2] = samplist.pop(0)
        else: #Project onto norm ball
            #samplist = np.array(samplist)    
            nd = norm(X[boolmask]-samplist)
            nds.append(nd) #
            if nd > epsilon:
                ndcount = ndcount+1
                alpha = (epsilon/nd)**2
                modlist = (alpha*X[boolmask] + (1-alpha)*samples).tolist()
                for ind1, row in enumerate(X): 
                    for ind2, val in enumerate(row): 
                        if sampmask[ind1,ind2]==0:
                            X[ind1,ind2] = modlist.pop(0)
        
        #matshow(X-Xinit); plt.colorbar()
        #Decrease smoothing parameter
        gamma = gamma/eta
        gamma = max(gamma,gammamin)
        #Compute cost
        cost.append(cost_function(p,gammamin,ev))
        
        #Compute error if you have ground truth
        if Xtrue.any():
            error.append(norm(X-Xtrue)/norm(Xtrue))
        
        #Check for Convergence
        updatestep = norm(X-Xold)/norm(Xold)
        update.append(updatestep) 
        
        if updatestep<exit_tol:
            print(' VMC reached exit tolerance at iteration ', step)
            print(' nd>epsilon for ', ndcount, ' iterations')
            break
        
        Xold = X
        step+=1
        
        ##### SJQ 12/15/2020 - Checking error vs. iteration; comment out once unneeded #####
        # if step %10 ==0:
        #     colerr_init = []
        #     colerr_vmc = []
        #     errortol = 1.e-3 #1e-3
        #     transpose = True
        #     if transpose == True:
        #         for j in range(Xtrue.shape[1]):
        #             colerr_init.append(norm(Xinit[:,j]/scalefac-Xtrue[:,j])/norm(Xtrue[:,j]))
        #             colerr_vmc.append(norm(X[:,j]-Xtrue[:,j])/norm(Xtrue[:,j]))
        #     else:
        #         for j in range(Xtrue.shape[0]):
        #             colerr_init.append(norm(Xinit[:,j]/scalefac-Xtrue[:,j])/norm(Xtrue[:,j]))
        #             colerr_vmc.append(norm(X[:,j]-Xtrue[:,j])/norm(Xtrue[:,j]))
            
        #     colerr_init.sort()
        #     colerr_init = colerr_init[::-1]
            
        #     #count_init = sum(map(lambda x: x <= errortol, colerr_init))
            
        #     colerr_vmc.sort()
        #     colerr_vmc = colerr_vmc[::-1]
        #     #count_vmc = sum(map(lambda x: x <= errortol, colerr_vmc))
            
        #     if transpose:
        #         cols = np.arange(Xtrue.shape[1])
        #     else:
        #         cols = np.arange(Xtrue.shape[0])
            
        #     fig, ax = plt.subplots(figsize=(8,4),dpi=200)
        #     ax.set_yscale('log')
        #     ax.set_ylim([1.e-8,1.e+2])
        #     ax.plot(cols,colerr_init,linestyle='solid',color='r',label='Initial',lw=1.75)
        #     ax.plot(cols,colerr_vmc,linestyle='dashed',color='b',label='HVMC',lw=1.75)
    
        #     ax.legend()
        #     ax.set_xlabel('Columns')
        #     ax.set_ylabel('Error')
        #     ax.grid()
        #     anntext = str(step)+'th iteration'
        #     ax.annotate(anntext,(0,1.e1),fontsize=13)
            
        #     # Used to return the plot as an image rray
        #     fig.canvas.draw()       # draw the canvas, cache the renderer
        #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #     image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #     ims.append(image)
        #     plt.close()
        ##### SJQ 12/15/2020 - End checking error vs.iteration #####
    
    ### CREATE GIF, COMMENT OUT WHEN UNNEEDED
    
    #imageio.mimsave('./HVMC.gif', ims, fps=60)
    ### END CREATE GIF
    
    
    #Plot Nd vs iteration
    # fig, ax = plt.subplots(figsize=(8,4),dpi=200)
    # ax.plot(list(range(step+1)),nds,linestyle='solid',color='r',label='ND',lw=1.75)
    # ax.set_ylim([0,0.1])
    # ax.axhline(epsilon,xmax=step-1,linestyle='dashed',label='Epsilon')
    # ax.legend()
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Samples ND')
    X = scalefac*X
    #X.mask = ma.nomask
    
    #X = np.real(X)
    return X,cost,update,error
        
        
def cost_function(p,gammamin,ev):
    if p == 0:
        cp = 0.5
        return 0.5*sum(np.log(ev+gammamin)) #TESTED
    else:
        cp = 0.5*p
        return sum(np.power((ev+gammamin),0.5)) #TESTED
    