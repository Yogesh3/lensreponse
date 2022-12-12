import numpy as np
import myfuncs as mf
import healpy as hp
import time
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Nprocs = comm.Get_size()

#Paths 
PROJECT_PATH = "/project/r/rbond/ymehta3/"
OUTPATH = PROJECT_PATH + "output/lensresponse/gerrit/"
kappa_sims_path = '/scratch/r/rbond/msyriac/data/sims/alex/v0.4/'
kappa_sim_name_path = PROJECT_PATH + '/input_data/kappa_sims/' + 'ikappas_gerrit.txt'
kappa_mask_name_path = '/project/r/rbond/ymehta3/input_data/' + 'name_dr6mask.txt'

#Load Kappa Mask
kmask = mf.readMap(kappa_mask_name_path)

#Read Kappa Names
Nsims, kappa_names = mf.readTxt(kappa_sim_name_path)

#Extras
w4 = mf.wn(kmask, 4)
BINSIZE = 32
LMAX = 4000
ells = np.arange(LMAX + 1)
iellmax = ells[-1] + 1
tot_time = 0

#Farm Out Inputs
sims_per_rank, displacement = mf.calcMPI(Nsims, Nprocs)
kappa_names_subset = kappa_names[displacement[rank] : displacement[rank+1]]
rank_Nsims = sims_per_rank[rank]

Cls = []
cov = []
startime = time.time()
for isim, kappa_name in enumerate(kappa_names_subset):
    sim_startime = time.time()
    
    #Read Kappa Sim
    phi_alm = hp.read_alm(kappa_sims_path + kappa_name)
    phi_alm = np.cdouble(phi_alm)
    fl = ells * (ells+1) / 2
    kappa_alm = hp.almxfl(phi_alm, fl)

    #Calculate Power Spectrum
    Cl_kk = hp.alm2cl(kappa_alm) #/ w4

    #Binning
    ells_bin, Cl_kk_bin = mf.binning(BINSIZE, ells, Cl_kk[:iellmax])

    #Calculate Covariance Matrix
    Cls.append(Cl_kk_bin)
    cov.append(np.outer(Cl_kk_bin, Cl_kk_bin))

    #Time Diagnostics
    sim_endtime = time.time()
    tot_time = mf.printSimTime(sim_startime, sim_endtime, rank, isim+1, rank_Nsims, tot_time)

Nells_bin = len(ells_bin)
#Convert to NumPy Arrays
Cls = np.array(Cls)
cov = np.array(cov)

#Collect
Cls_tot = np.zeros((Nsims, Nells_bin))
cov_tot = np.zeros((Nsims, Nells_bin, Nells_bin))
comm.Gather(Cls, Cls_tot, root=0)
comm.Gather(cov, cov_tot, root=0)

if rank == 0:
    #Calculate Covariance Matrix
    avgCl = np.mean(Cls_tot, axis=0)
    avgcov = np.mean(cov_tot, axis=0)
    avgcov = avgcov - np.outer(avgCl, avgCl)

    #Save Avg Cls and Covariance
    np.save(OUTPATH + 'input_Cls_kk.npy', Cls_tot)
    np.save(OUTPATH + 'input_cov_kk.npy', avgcov)
    np.save(OUTPATH + 'input_avgCl_kk.npy', avgCl)

    #Time Diagnostics
    endtime = time.time()
    mf.printTotalTime(startime, endtime, Nthings = Nsims)