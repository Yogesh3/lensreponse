import numpy as np
import myfuncs as mf
import healpy as hp
import time
from mpi4py import MPI 
import sys

"""
Calculate the kappa autospectrum with and without m-dependent inverse variance weights for a set of sims. Applies a reponse function calibrated from sims. Calculates full covariance matrix and saves autospectra for each sim.
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Nprocs = comm.Get_size()

#Paths 
PROJECT_PATH = "/project/r/rbond/ymehta3/output/lensresponse/gerrit/"
OUTPATH = "/project/r/rbond/ymehta3/output/lensresponse/gerrit/"
kappa_sims_path = '/scratch/r/rbond/gfarren/unWISExACT_outputs/sims/kappa/'
kappa_sim_name_path = '/project/r/rbond/ymehta3/input_data/kappa_sims/' + 'okappas_gerrit.txt'
kappa_mask_name_path = '/project/r/rbond/ymehta3/input_data/' + 'name_dr6mask.txt'

#Load Reponse Functions
R_names = ['R2_smoothed_32.npy', 'R2_smoothed_64.npy', 'R2_smoothed_128.npy']
Rs = []
for R_name in R_names:
    R_current = np.load(PROJECT_PATH + R_name)
    R_current = np.where(R_current == 0., np.nan, R_current)
    Rs.append(R_current)
    
#Load Kappa Variances
var_names = ['okappa_var_32.npy', 'okappa_var_64.npy', 'okappa_var_128.npy']
var_list = []
for var_name in var_names:
    var_list.append(np.load(PROJECT_PATH + var_name))

#Load Kappa Mask
kmask = mf.readMap(kappa_mask_name_path)

#Read Kappa Names
Nsims, kappa_names = mf.readTxt(kappa_sim_name_path)

#Extras
w4 = mf.wn(kmask, 4)
ells = np.arange(len(var_list[0]))
iellmax = ells[-1] + 1
factors = ['32', '64', '128']
BINSIZE = 32
tot_time = 0

#Farm Out Inputs
sims_per_rank = np.zeros(Nprocs, dtype=int)
displacement = np.zeros(Nprocs+1, dtype=int)
if rank == 0:
    sims_per_rank, displacement = mf.calcMPI(Nsims, Nprocs)
comm.Bcast(sims_per_rank, root=0)
comm.Bcast(displacement, root=0)
kappa_names_subset = kappa_names[displacement[rank] : displacement[rank+1]]
rank_Nsims = sims_per_rank[rank]

Cls_unweighted = {'32':[], '64':[], '128':[]}
Cls_weighted = {'32':[], '64':[], '128':[]}
cov_unweighted = {'32':[], '64':[], '128':[]}
cov_weighted = {'32':[], '64':[], '128':[]}
startime = time.time()
for isim, kappa_name in enumerate(kappa_names_subset):
    sim_startime = time.time()
    
    #Read Kappa Sim
    kappa_alm = hp.read_alm(kappa_sims_path + kappa_name)
    kappa_alm = np.cdouble(kappa_alm)
    kappa_lm = mf.idx2lm(kappa_alm)

    #Create Normalized Weights
    W2_list = []
    for var in var_list:
        W2_unnorm = 1 / var.copy()**2
        W2_list.append(W2_unnorm)

    #Create Unweighted Cls
    Cl_kk_unweighted_list = []
    for R in Rs:
        #Calculation
        kappa_alm_response = mf.lm2idx(kappa_lm / R)
        Cl_kappa = hp.alm2cl(kappa_alm_response) / w4
        
        #Binning
        ells_bin, Cl_kk_bin = mf.binning(BINSIZE, ells, Cl_kappa[:iellmax])
        
        Cl_kk_unweighted_list.append(Cl_kk_bin)

    #Create Weighted Cls
    Cl_kk_weighted_list = []
    for i, W2 in enumerate(W2_list):
        #Calculations
        kappa_alm_response = (kappa_lm / Rs[i])[:iellmax, :iellmax]
        Cl_kk_weighted = np.real( kappa_alm_response * np.conjugate(kappa_alm_response) )
        Cl_kk_weighted = mf.lm2cl(Cl_kk_weighted, weights= W2) / w4

        #Binning
        ells_bin, Cl_weighted_bin = mf.binning(BINSIZE, ells, Cl_kk_weighted)
        
        Cl_kk_weighted_list.append(Cl_weighted_bin)

    #Calculate Covariance Matrix
    for i,factor in enumerate(factors):
        Cls_unweighted[factor].append(Cl_kk_unweighted_list[i])
        cov_unweighted[factor].append(np.outer(Cl_kk_unweighted_list[i], Cl_kk_unweighted_list[i]))

        Cls_weighted[factor].append(Cl_kk_weighted_list[i])
        cov_weighted[factor].append(np.outer(Cl_kk_weighted_list[i], Cl_kk_weighted_list[i]))

    #Time Diagnostics
    sim_endtime = time.time()
    tot_time = mf.printSimTime(sim_startime, sim_endtime, rank, isim+1, rank_Nsims, tot_time)


Nells_bin = len(ells_bin)
for factor in factors:
    #Convert to NumPy Arrays
    Cls_unweighted[factor] = np.array(Cls_unweighted[factor])
    cov_unweighted[factor] = np.array(cov_unweighted[factor])
    Cls_weighted[factor] = np.array(Cls_weighted[factor])
    cov_weighted[factor] = np.array(cov_weighted[factor])

    #Collect
    Cls_unweighted_tot = np.zeros((Nsims, Nells_bin))
    Cls_weighted_tot = np.zeros((Nsims, Nells_bin))
    cov_unweighted_tot = np.zeros((Nsims, Nells_bin, Nells_bin))
    cov_weighted_tot = np.zeros((Nsims, Nells_bin, Nells_bin))
    comm.Gather(Cls_unweighted[factor], Cls_unweighted_tot, root=0)
    comm.Gather(Cls_weighted[factor], Cls_weighted_tot, root=0)
    comm.Gather(cov_unweighted[factor], cov_unweighted_tot, root=0)
    comm.Gather(cov_weighted[factor], cov_weighted_tot, root=0)

    if rank == 0:
        #Calculate Covariance Matrix
        avgCl_unweighted = np.mean(Cls_unweighted_tot, axis=0)
        avgCl_weighted = np.mean(Cls_weighted_tot, axis=0)
        avgcov_unweighted = np.mean(cov_unweighted_tot, axis=0)
        avgcov_weighted = np.mean(cov_weighted_tot, axis=0)
        avgcov_unweighted = avgcov_unweighted - np.outer(avgCl_unweighted, avgCl_unweighted)
        avgcov_weighted = avgcov_weighted - np.outer(avgCl_weighted, avgCl_weighted)

        #Save All Cls
        np.save(OUTPATH + 'Cls_kk_unweighted_' + factor + '.npy', Cls_unweighted_tot)
        np.save(OUTPATH + 'Cls_kk_weighted_' + factor + '.npy', Cls_weighted_tot)

        #Save Covariances
        np.save(OUTPATH + 'cov_kk_unweighted_' + factor + '.npy', avgcov_unweighted)
        np.save(OUTPATH + 'cov_kk_weighted_' + factor + '.npy', avgcov_weighted)
        np.save(OUTPATH + 'avgCl_kk_unweighted_' + factor + '.npy', avgCl_unweighted)
        np.save(OUTPATH + 'avgCl_kk_weighted_' + factor + '.npy', avgCl_weighted)


#Time Diagnostics
if rank == 0:
    endtime = time.time()
    mf.printTotalTime(startime, endtime, Nthings = Nsims)