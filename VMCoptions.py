#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:25:47 2020
Create options object
@author: shaama
Cleaning up from the previous version and removing error calculation for ZPE and deltaG from 
this file
"""

import numpy as np
import numpy.ma as ma
from vmc import vmc
import pandas as pd
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import matshow
#from itertools import combinations 
import os
import shutil
import random

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


class VMCoptions: 
    
    def __init__(self,d=2,c=1.,epsilon=0,niter=5000,gamma0=0,
                 gammamin=1e-7,eta=1.01,p=0.5,exit_tol=1.e-6,eigtol=1e-7,dct_trans=False):
        self.d = d
        self.c = c
        self.epsilon = epsilon
        self.niter = niter
        self.gamma0 = gamma0
        self.gammamin = gammamin
        self.eta = eta
        self.p = p
        self.exit_tol = exit_tol
        self.eigtol = eigtol
        self.dct_trans = dct_trans

def create_sampmask(density,Xtrue,typemask='element',prioritize='True'):
    #Prioritize argument sorts indices such that the TS gets sampled most
    #if this is a half reaction going from TS to minimum.
    dim1 = Xtrue.shape[0]
    dim2 = Xtrue.shape[1]
    if typemask=='element':
        sampmask = np.random.choice([0,1],size=(dim1,dim2),p=[density,1-density])
    elif typemask=='column':
        sampmask = np.random.choice([0,1],size=(1,dim2),p=[density,1-density])
        sampmask = np.repeat(sampmask,dim1,axis=0)
    elif typemask =='row':
        sampmask = np.random.choice([0,1],size=(dim1,1),p=[density,1-density])
        sampmask = np.repeat(sampmask,dim2,axis=1)
        
    if prioritize: 
        sumvals = np.sum(sampmask,axis=0)
        idx = sumvals.argsort()
        newmask = np.take(sampmask,idx,axis=1)
        sampmask=newmask
    return sampmask

def create_xinit(sampmask,Xtrue): 
    Xinit = ma.masked_array(Xtrue,mask=sampmask,hard_mask=False)
    return Xinit#.data



def get_eigenvalue_matrix(df,tag='All'):
    evals = []
    energies = []
    if tag in ['R','P','T']:
        for index,row in df.iterrows():
            if row.tag==tag:
                evalrow = []
                for x in row.Eigenvalues:
                    #if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    elif tag=='All':
        for index,row in df.iterrows():
            evalrow=[]
            for x in row.Eigenvalues:
                #if abs(x)>cutoff:
                evalrow.append(x)
            evals.append(evalrow)
            energies.append(row.Energy)
    elif tag=='Rhalf': 
        for index,row in df.iterrows():
            if index <=0:
                evalrow=[]
                for x in row.Eigenvalues:
                    #if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    elif tag=='Phalf': 
        for index,row in df.iterrows():
            if index >=0:
                evalrow=[]
                for x in row.Eigenvalues:
                    #if abs(x)>cutoff:
                    evalrow.append(x)
                evals.append(evalrow)
                energies.append(row.Energy)
    return evals, energies


def generate_col_combo_mask(combo,Xtrue):
    #Based on a combination array of the indices to be masked, create the column mask 
    dim1 = Xtrue.shape[0]
    dim2 = Xtrue.shape[1]
    
    sampmask = np.ones([1,dim2])
    for i in combo:
        sampmask[0,i] = 0
    sampmask = np.repeat(sampmask,dim1,axis=0)

    return sampmask
##VMC

pathtag = 'All' #tags - R: reactant; P: product; T: TS; All: everything; Rhalf; Phalf
filename = 'Systems/Sn2Ar1_dft2svp_step100_disp5000_num100.pkl'
df = pd.read_pickle(filename)

# Determine if ordered, then parse
if 'ordered' in filename:
    theory = filename.split('_')[2]
    system = filename.split('_')[0]
    step = ''.join([x for x in filename.split('_')[3] if x not in ['s','t','e','p']])
    disp = ''.join([x for x in filename.split('_')[4] if x not in ['d','i','s','p']])
    num = ''.join([x for x in filename.split('_')[5] if x not in ['n','u','m','.','p','k','l']])
    ordered ='_ordered' #to add to filename
else:
    #unordered
    theory = filename.split('_')[1]
    system = filename.split('_')[0]
    step = ''.join([x for x in filename.split('_')[2] if x not in ['s','t','e','p']])
    disp = ''.join([x for x in filename.split('_')[3] if x not in ['d','i','s','p']])
    num = ''.join([x for x in filename.split('_')[4] if x not in ['n','u','m','.','p','k','l']])
    ordered = ''

# theory = filename.split('_')[3]
# system = filename.split('_')[0]
# step = ''.join([x for x in filename.split('_')[4] if x not in ['s','t','e','p']])
# disp = ''.join([x for x in filename.split('_')[5] if x not in ['d','i','s','p']])
# num = ''.join([x for x in filename.split('_')[6] if x not in ['n','u','m','.','p','k','l']])




Xtrue, energies = get_eigenvalue_matrix(df,pathtag)
Xtrue = np.array(Xtrue)


Xtrue = Xtrue.T
dim1 = Xtrue.shape[0]
dim2 = Xtrue.shape[1]
matsize = dim1*dim2

##### SETTINGS PT1

transpose = True #False
noise  = False
trial_specific_epsil = False #each trial has a different epsil; see trials loop
artificial_noise = False #Use true if using simulated gaussian noise

if noise:
    stdev = '1e-03' #Sigma for the normal distribution of noise added
    # Xtrue_noise = Xtrue+np.random.normal(0,stdev,Xtrue.shape)
    epsilons=[]
    if artificial_noise:
        Xtrue_noise = np.load('Xtrue_noise'+stdev+'_'+system[3:]+'.npy')
    else:
        noise_filename = 'CF3CH3_ordered_dft2tzvp-freqsvp_step50_disp5000_num100.pkl'
        df_noise = pd.read_pickle(noise_filename)
        Xtrue_noise, energies = get_eigenvalue_matrix(df_noise,pathtag)
        Xtrue_noise = np.array(Xtrue_noise)
        Xtrue_noise = Xtrue_noise.T
    if trial_specific_epsil:
        prefac = 2.5 #scalar of norm diff between true samples and noisy samples
    else:
        epsilon=0
        pass
else:
    epsilon = 0

'''
if Xtrue.shape[0]>Xtrue.shape[1]:
    Xtrue = Xtrue.T
    transpose = True
'''

##### SETTINGS PT2

density = 0.05  #Only applies for any variant of element sampling 
ntrials = 100
trial = 0 #Starting trial index. Leave at 0.
priority = False#slightly non-random sampling where TS modes are sampled more
sampmask_type = 'GuaranteeAllRows' #['Normal', 'RandomColumn','ManualColumn','GuaranteeAllRows','Hybrid', 'Manual']
d=2 # power, Default 2
p=0.5 # Schatten-p  norm Default 0.5, 


#If we want to do random column sampling, make list of all combinations of columns
if sampmask_type == 'RandomColumn':
    k = 7 #The number of columns we want to sample


# Construct filename      
resultfile = ''
resultfile += 'resultsT_'+pathtag+'_ntrials'+str(ntrials)+'_'+system + ordered +'_'+theory+'_step'+step+'_disp'+disp+'_num'+num+'_samptype'+sampmask_type
if sampmask_type == 'Manual': #Name of the manual sampling scheme
    resultfile+= 'ColRow'
if priority:
    resultfile += '_priority'
if sampmask_type not in ['ManualColumn','RandomColumn','Manual']:
    resultfile += '_density'+str(density)
if sampmask_type == 'RandomColumn':
    resultfile += '_RandCol' + str(k) 
if noise:
    if artificial_noise:
        resultfile+= '_noise'+stdev
        if trial_specific_epsil:
            resultfile +='_epsil'+str(prefac)+'NormDiff'
        else:
            resultfile +='_epsil'+"{:.1e}".format(epsilon)
    else: 
        resultfile+= '_noiseSvpTzvp'
        if trial_specific_epsil:
            resultfile +='_epsil'+str(prefac)+'NormDiff'
        else:
            resultfile +='_epsil'+"{:.1e}".format(epsilon)
if d !=2: # If non default p and d    
    resultfile += '_d'+str(d)
if p!= 0.5:
    resultfile += '_p'+str(p)
resultfile +='.pkl'


print('Beginning trials for: ', resultfile )


#Error calculation raw data
colerr_init = []
colerr_vmc = []
init_err = []
fin_err = []
all_cost = []
all_update = []
all_error = []
all_colerr_init = []
all_colerr_vmc = []
all_count_init = []
all_count_vmc = []
n95 = 0
n90 = 0
alldict = []
trial_densities=[]

skipped = 0


while trial<ntrials:
    if sampmask_type == 'Normal': #Random element 
        sampmask = create_sampmask(density,Xtrue,'element',prioritize=priority)
        
    elif sampmask_type == 'RandomColumn':
        #n=np.random.randint(low=0,high=len(all_combo))
        #combo=all_combo[n]
        
        combo = tuple(random.sample(range(dim2),k))
        sampmask=generate_col_combo_mask(combo,Xtrue)
    elif sampmask_type == 'Hybrid':
        #10-7-20 This is for 1-front-1-back with random sampling in between. 1-Front-1-back included in samp. density
        #12-29-20 Determination of actual density is OUTDATED. Fix later
        actual_density = density - 2/dim2
        sampmask = create_sampmask(actual_density,Xtrue,'element',prioritize=priority)
        for row in sampmask:
            row[0]=0
            row[-1]=0
        raise Exception('Hybrid needs corrected sampling densities')
    elif sampmask_type == 'ManualColumn':
        
        #sampmask = np.array([[0]*3 + [1]*9 + [0]*1 + [1]*8 + [0]*3])
        #sampmask = np.array([[0]*3 + [1]*9 + [0]*1 + [1]*9 + [0]*2])
        #sampmask = np.array([[1]*17 + [0]*7])
        #sampmask = np.array([[1,0,1] + [1,0,1,1] + [1,0,1] + [1,1,0,1] + [1,0,1] + [1,1,0,1] +[1,0,1]])
        
        #sampmask = np.array([[0]+[1]*22+[0]])
        
        sampmask = np.ones([1,dim2])
        sampmask[0][[0,57]]=0
        
        #sampmask[0,4]=0
        sampmask = np.repeat(sampmask,dim1,axis=0)
    elif sampmask_type == 'Manual': # Can use this space to make an arbitrary sampmask
        sampmask = np.ones([dim1,dim2])
        
        #Sample bottom ten rows
        sampmask[-10:-1,:]=0
        
        #Sampple R, P regions
        sampmask[:,[0,-1]]=0
        #sampmask[:,0:3]=0
        #sampmask[:,-4:-1]=0
        
        #Sample some of R region
        sampmask[:,10:15]=0
        
        #Sample some of P region
        sampmask[:,39:45]=0
        
        #Sample some of TS region
        sampmask[:,22:27]=0

        
    elif sampmask_type == 'GuaranteeAllRows':
        #actual_density = density - 1/dim2
        actual_density = (density*dim2-1)/(dim2-1)
        sampmask = create_sampmask(actual_density,Xtrue,'element',prioritize=priority)        
        row_ind = np.random.randint(dim2,size=dim1)
        sampmask[list(range(dim1)),row_ind]=0
    if noise:
        Xinit = create_xinit(sampmask,Xtrue_noise)
    else:
        Xinit = create_xinit(sampmask,Xtrue)
    samples = Xinit.compressed()
    
    Xinit = Xinit.filled(fill_value=0) #Fill values of Xinit with zero
    
    
    #check if entire rows are not sampled, skip trial if true -SJQ
    #if skip_trial_no_row and 0 in np.sum(Xinit,axis=1):
        #print('Row not sampled, skipped')
        #skipped = skipped+1
        #continue
    print('Beginning trial ', trial+1, ' of ', ntrials)
    if noise and trial_specific_epsil:
        boolsampmask = (1-sampmask).astype(bool)
        
        epsilon =prefac* norm(Xtrue[boolsampmask]-samples)
        print(' Epsilon = ', epsilon)
        epsilons.append(epsilon)
    
    #Run VMC
    options = VMCoptions(
        d=d,
        p=p,
        niter=10000,
        gammamin=1e-16,
        c=1.,
        exit_tol=1.e-8,
        epsilon=epsilon, 
        #eta = 1.001,
        )  
    
    #defaults
    #options = VMCoptions(d=2,p=0.5,niter=10000,gammamin=1e-16,c=1.,exit_tol=1.e-8,epsilon=epsilon)  
    
    
    #exit_tol 1e-12: 10000 iterations are not enough to converge
    #exit_tol 1e-10 or gammamin 1e-8: 10000 iterations are not enough to converge for larger systems
    Xfinal, cost, update, error = vmc(Xinit, sampmask, samples, options, Xtrue)
    
        
    all_cost.append(cost)
    all_update.append(update)
    all_error.append(error)
    init_err.append(norm(Xinit-Xtrue)/norm(Xtrue))
    fin_err.append(norm(Xfinal-Xtrue)/norm(Xtrue))
    
    
    #Column errors
    colerr_init = []
    colerr_vmc = []
    errortol = 1.e-3 #1e-3
    
    if transpose == True:
        for j in range(Xtrue.shape[1]):
            colerr_init.append(norm(Xinit[:,j]-Xtrue[:,j])/norm(Xtrue[:,j]))
            colerr_vmc.append(norm(Xfinal[:,j]-Xtrue[:,j])/norm(Xtrue[:,j]))
    else:
        for j in range(Xtrue.shape[0]):
            colerr_init.append(norm(Xinit[j,:]-Xtrue[j,:])/norm(Xtrue[j,:]))
            colerr_vmc.append(norm(Xfinal[j,:]-Xtrue[j,:])/norm(Xtrue[j,:]))
        
    
    colerr_init.sort()
    colerr_init = colerr_init[::-1]
    
    count_init = sum(map(lambda x: x <= errortol, colerr_init))
    
    colerr_vmc.sort()
    colerr_vmc = colerr_vmc[::-1]
    count_vmc = sum(map(lambda x: x <= errortol, colerr_vmc))
    
    all_colerr_init.append(colerr_init)
    all_colerr_vmc.append(colerr_vmc)
    trial_densities.append(len(samples)/matsize)
    
    print(" Initial status: %columns < error threshold = ", int(count_init*100./len(colerr_init)))
    print(" VMC Performance: %columns < error threshold = ", int(count_vmc*100./len(colerr_vmc)))
    all_count_init.append(int(count_init*100./len(colerr_init)))
    all_count_vmc.append(int(count_vmc*100./len(colerr_vmc)))
    
    
    if int(count_vmc*100./len(colerr_vmc))>=95:
        n95+=1
    if int(count_vmc*100./len(colerr_vmc))>=90:
        n90+=1
    
    if transpose: 
        alldict.append({'Trial': trial,
                        'Xinit': Xinit,
                        'Xtrue': Xtrue,
                        'Energies': energies,
                        'Xfinal': Xfinal,})
    else:
        alldict.append({'Trial': trial,
                        'Xinit': Xinit.T,
                        'Xtrue': Xtrue.T,
                        'Energies': energies,
                        'Xfinal': Xfinal.T})
    
    trial +=1
    
    
if transpose:
    cols = np.arange(Xtrue.shape[1])
else:
    cols = np.arange(Xtrue.shape[0])

# Colerr
fig, ax = plt.subplots(figsize=(8,4),dpi=200)
ax.set_yscale('log')
ax.set_ylim([1.e-8,1.e+2])

# % Change per iteration
fig1, ax1 = plt.subplots(figsize=(8,4),dpi=200)
ax1.set_yscale('log')
ax1.set_ylim([1.e-8,1.e+2])

for i in range(ntrials):
    ax.plot(cols,all_colerr_init[i],linestyle='solid',color='r',label='Initial',lw=1.75)
    ax.plot(cols,all_colerr_vmc[i],linestyle='dashed',color='b',label='HVMC',lw=1.75)
    if i==0:
        ax.legend()
        ax.set_xlabel('Columns')
        ax.set_ylabel('Error')
        ax.grid()
        anntext = r'$\rho$ = '+str(density)+', '+str(ntrials)+' trials'
        ax.annotate(anntext,(0,1.e1),fontsize=13)


    ##### SJQ 12/16/2020 Check %change vs. iteration #####
    ax1.set_yscale('log')
    ax1.set_ylim([1.e-8,1.e+2])
    ax1.plot(list(range(len(all_update[i]))),all_update[i],linestyle='dashed',color='b',label='Trials',lw=1.75)
    if i==0:
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('% Change')
        ax1.grid()
        anntext = r'$\rho$ = '+str(density)+', '+str(ntrials)+' trials, System =' + system[3:]
        ax1.annotate(anntext,(0,1.e1),fontsize=13)
    ##### SJQ 12/16/2020 END Check %change vs. iteration #####

# Check HVMC matrix structure

# % Change per iteration
fig3, ax3 = plt.subplots(figsize=(8,12),dpi=200)
for i in range(Xfinal.shape[0]):
    ax3.plot(list(range(Xfinal.shape[1])),Xfinal[i,:],linestyle='solid',label='Matrix Elements',lw=1.35,marker=r'$\odot$')
    if i==0:
        ax3.legend()
        ax3.set_xlabel('Reaction Path Step')
        ax3.set_ylabel('Vibrational Frequency')
        ax3.set_title('HVMC-recovered Matrix of the Last Trial')
        ax3.grid()


# Compile results 
alldf = pd.DataFrame.from_dict(alldict)    

#Store figures and output, overwrite old ones 
colerrFile='Figures/'+resultfile+'/ColErr.png'
updateFile='Figures/'+resultfile+'/UpdateErr.png'
outfile='Figures/'+resultfile+'/hvmc.out'
targetDir='Figures/'+resultfile+'/'

if os.path.isdir(targetDir):
    shutil.rmtree(targetDir)
    
os.mkdir(targetDir)

print("Number of trials where recovery >90% = ", n90, file=open(outfile, "a"))    
print("Number of trials where recovery >95% = ", n95, file=open(outfile, "a"))    

print("Average initial = ", np.mean(all_count_init), file=open(outfile, "a"))
print("Average VMC = ", np.mean(all_count_vmc), file=open(outfile, "a"))
print("Average Density  = ", np.mean(trial_densities), file=open(outfile, "a"))
if noise and trial_specific_epsil:
    print('Average Epsilon = ', np.mean(epsilons), file=open(outfile, "a"))
#if skip_trial_no_row:
    #print(' Skipped trials: ', skipped, file=open(outfile, "a"))

fig.savefig(colerrFile)
fig1.savefig(updateFile)

alldf.to_pickle('ResultsPKL/'+resultfile)
print(resultfile) 
