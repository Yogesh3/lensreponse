import myfuncs as mf
import numpy as np
import healpy as hp
from pixell import enmap, curvedsky
import orphics.maps
import time
import sys
import csv
from mpi4py import MPI 
from scipy.interpolate import interp2d
import argparse

def wn(mask1, n1, mask2=None, n2=None):
    """TODO: check pixel area average"""
    pmap = orphics.maps.psizemap(mask1.shape, mask1.wcs)
    if mask2 is None:
        output = np.sum(mask1**n1 * pmap) /np.pi / 4.
    else:
        output = np.sum(mask1**n1 * mask2**n2 * pmap) /np.pi / 4.
    return output


def idx2lm(aidx):
    lmax = hp.Alm.getlmax(len(aidx))
    alms = np.zeros((lmax+1, lmax+1), dtype=np.complex128)

    ells, ems = hp.Alm.getlm(lmax)
    idxs = hp.Alm.getidx(lmax, ells, ems)
    alms[ells, ems] = aidx[idxs]
    
    return alms


def lm2idx(alm):
    lmax = alm.shape[0] - 1
    
    ells, ems = hp.Alm.getlm(lmax)
    idxs = hp.Alm.getidx(lmax, ells, ems)
    
    aidx = np.zeros(len(idxs), dtype=np.complex128)
    aidx[idxs] = alm[ells, ems]
    
    return aidx


def vecNanStat(x, statistic="mean", **npkwargs):
    """Vectorizes NumPy's nanmean and nanmedian functions

    Args:
        x (list of numpy arrays): list of arrays of which to calculate the statistic
        statistic ("mean", "median"): Which statistic to use. Defaults to "mean".
    """ 
    statlist = []   
    for array in x:
        if statistic == "mean":
            statlist.append(np.nanmean(array, **npkwargs))
        elif statistic == 'median':
            statlist.append(np.nanmedian(array, **npkwargs))
    return statlist


# def writeWrapper(fname, headerlist):
#     fileobj = open(fname, 'w')   
#     writer = csv.writer(fileobj)
#     writer.writerow(headerlist)
#     return writer


#Parser
parser = argparse.ArgumentParser(description='Calculates kappa response function in harmonic and fourier space.')
parser.add_argument('--sim_version', type=str, default='v3', choices=['v3', 'v2', 'gerrit', 'release'])
parser.add_argument('--space', type=str, default='both', choices=['harmonic', 'fourier', 'both'])
parser.add_argument('--fft_normalize', type=str, default='no', choices=['yes', 'no'], help='normalize kappa sims by the fft normalization curve from symlens before calculating the harmonic response function?')
parser.add_argument('--nsims', type=int, default=0, help='number of simulations (leave blank for max sims in this set)')
parser.add_argument('--outdir', type=str, default='project', choices=['project', 'scratch'])
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Nprocs = comm.Get_size()

#Paths
SCRATCH = '/scratch/r/rbond/ymehta3/'
PROJECT = '/project/r/rbond/ymehta3/'
INPUT_PATH = PROJECT + 'input_data/kappa_sims/'
inkappa_path = '/scratch/r/rbond/msyriac/data/sims/alex/v0.4/'
# inkappa_path = '/project/r/rbond/ymehta3/output/COSMO2017_10K_acc3_lenspotentialCls.dat'
Al_path = SCRATCH + 'twodratio_TT.fits'

#Get Kapps Sims Names
if args.sim_version == 'v2':
    outkappa_path = '/home/r/rbond/jiaqu/scratch/DR6/coaddMV4phoct9/stage_scatter/'
    kappasims_names = INPUT_PATH + 'iokappas_v2.txt'
    OUTPUT_PATH = '/project/r/rbond/ymehta3/output/lensresponse/v2/'

elif args.sim_version == 'v3':
    outkappa_path = '/project/r/rbond/jiaqu/kappa_maps/sims/'
    kappasims_names = INPUT_PATH + 'iokappas_v3.txt'
    ikappa_names = INPUT_PATH + 'ikappas_v3.txt'
    okappa_names = INPUT_PATH + 'kappasims_v3.txt'
    OUTPUT_PATH = '/project/r/rbond/ymehta3/output/lensresponse/v3/'
    mask_path = '/home/r/rbond/jiaqu/scratch/DR6/maps/sims/map_masks/BN_bottomcutsmooth1.fits'

elif args.sim_version == 'release':
    outkappa_path = '/home/r/rbond/jiaqu/DR6lensing_backup/DR6lensing/full_mask/output/release_sims/'
    kappasims_names = INPUT_PATH + 'iokappas_release.txt'
    ikappa_names = INPUT_PATH + 'ikappas_release.txt'
    okappa_names = INPUT_PATH + 'kappasims_release.txt'
    OUTPUT_PATH = '/project/r/rbond/ymehta3/output/lensresponse/release/'
    mask_path = PROJECT + 'kmask_release.fits'

elif args.sim_version == 'gerrit':
    outkappa_path = '/scratch/r/rbond/gfarren/unWISExACT_outputs/sims/kappa/'
    kappasims_names = INPUT_PATH + 'iokappas_gerrit.txt'
    if (args.outdir).lower() == 'project':
        OUTPUT_PATH = PROJECT + 'output/lensresponse/gerrit/'
    elif (args.outdir).lower() == 'scratch':
        OUTPUT_PATH = SCRATCH + 'output/lensresponse/gerrit/'
    else:
        OUTPUT_PATH = args.outdir
    mask_path = '/home/r/rbond/jiaqu/scratch/DR6/downgrade/downgrade/act_mask_20220316_GAL060_rms_70.00_d2sk.fits'
    mc_corr_path = outkappa_path + 'all_MV_mc_bias_MV_sims1-400'

#Output Stats Files Info
if rank == 0:
    header_harm = ["avg R for all ells", "avg R up to ell=1000", "avg R up to ell=2000",
                    "median R for all ells", "median R up to ell=1000", "median R up to ell=2000"]
    header_harm = ', '.join(header_harm)
    header_fft = ["avg R for all ells", "median R for all ells"]
    header_fft = ', '.join(header_fft)
    harm_stats_outpath = OUTPUT_PATH + 'stats_harmonic_' + args.sim_version + '.csv'
    fft_stats_outpath = OUTPUT_PATH + 'stats_fft_' + args.sim_version + '.csv'

#Read Mask Map
mask_map = enmap.read_map(mask_path)

#Create Fourier Filter
if args.fft_normalize == 'yes':
    Al = enmap.read_map(Al_path)
    low_kys, low_kxs = Al.lmap()
    Al = np.where(np.isnan(Al), 1, Al)
    high_kys, high_kxs = mask_map.lmap()
    f_Al = interp2d(low_kxs[0,:], low_kys[:,0], Al) 
    Al_highres = f_Al(high_kxs[0,:], high_kys[:,0])
    Al_highres = enmap.fftshift(Al_highres) 

#Worker Rank Initializations
fnames_all = []
mc_bias = []
rows_per_rank = np.zeros(Nprocs, dtype= int)
displacement = np.zeros(Nprocs+1, dtype= int)

#Calculating Kappa File Names Division
Ncol = 2
if rank == 0:
    startime = time.time() 

    #Get All of the Sims Names
    if args.sim_version == 'gerrit':
        with open(kappasims_names) as iok_files, open(mc_corr_path) as bias_obj:
            reader = csv.reader(iok_files, delimiter=' ')
            for line in reader:
                fnames_all.append(line)

                for line in bias_obj:
                    mc_bias.append(float(line.rstrip('\n')))
                # mc_bias = np.array(mc_bias)

    else:
        with open(kappasims_names) as iok_files:
            reader = csv.reader(iok_files, delimiter=' ')
            for line in reader:
                fnames_all.append(line)

    if args.nsims != 0:
        fnames_all = fnames_all[:args.nsims]

    base_work, remainder = divmod(len(fnames_all), Nprocs)    
    
    for p in range(Nprocs):
        #Number of Sims Per Rank
        if p < remainder:
            rows_per_rank[p] = base_work + 1
        else:
            rows_per_rank[p] = base_work

        #Starting Index For Each Rank
        displacement[p] = sum(rows_per_rank[:p])

    #Add Ending Index
    displacement[-1] = sum(rows_per_rank) + 1
    
#Farm Out Inputs
comm.Bcast(rows_per_rank, root=0)
comm.Bcast(displacement, root=0)
mc_bias = comm.bcast(mc_bias, root=0)
mc_bias = np.array(mc_bias, dtype=np.double)
fnames_all = comm.bcast(fnames_all, root=0)
fnames_subset = fnames_all[displacement[rank] : displacement[rank+1]]

#Initializations
simnum = 1
tot_time = 0
harm_stats_tot = []
fft_stats_tot = []
rank_Nsims = rows_per_rank[rank]
Nsims = sum(rows_per_rank)

for simfiles in fnames_subset:
    sim_startime = time.time()
    
    print(f'\nRank {rank} starting sim {simnum}/{rank_Nsims}')
    sys.stdout.flush()

    #Read Output Alms
    okappa_filename = outkappa_path + simfiles[1]
    if args.sim_version == 'v2':
        okappaarray = np.load(okappa_filename)
        okappaarray = (okappaarray[1]+okappaarray[2]+okappaarray[3]+okappaarray[4]+okappaarray[5]+okappaarray[6])/6
        okappa = np.cdouble(okappaarray)
    else: 
        okappa = hp.read_alm(okappa_filename)
        okappa = np.cdouble(okappa)
        if args.sim_version == 'gerrit':
            okappa = hp.sphtfunc.almxfl(okappa, mc_bias)     # undo isotropic response function from Gerrit
    
    #Preliminary Info from Output Alms
    olmax = hp.Alm.getlmax(len(okappa))

    #Make Output Map
    footprint = enmap.zeros(mask_map.shape, wcs=mask_map.wcs)
    omap = curvedsky.alm2map(okappa, footprint, tweak=True)

    #Read Input Alms
    ikappa_filename = inkappa_path + simfiles[0]
    iphi_alms = hp.read_alm(ikappa_filename)
    iphi_alms = np.cdouble(iphi_alms)

    #Preliminary Info from Input Alms
    ilmax = hp.Alm.getlmax(len(iphi_alms))
    iells = np.arange(ilmax+1)

    #Convert Input phi to kappa
    fl = iells * (iells+1) / 2
    ikappa_alms = hp.almxfl(iphi_alms, fl)

    #Make Masked Input Map
    imap = curvedsky.alm2map(ikappa_alms, footprint, tweak=True)
    imap_masked = imap * mask_map**2         # why am I squaring the mask?
    ikappa_alms = curvedsky.map2alm(imap_masked, lmax= ilmax, tweak=True)

    #Convert kappas To l,m Grid
    ikappalm = idx2lm(ikappa_alms)
    ikappalm = ikappalm[:olmax+1, :olmax+1]
    okappalm = idx2lm(okappa)

    if args.space == 'fourier' or args.space == 'both' or args.fft_normalize == 'yes':
        #Fourier Transforms
        fft_imap = enmap.fft(imap_masked, normalize='physical')
        fft_omap = enmap.fft(omap, normalize='physical')

    #Initialize
    if simnum == 1:
        if args.space == 'harmonic' or args.space == 'both':
            harm_R1       = np.zeros((olmax+1, olmax+1))
            harm_avgnum   = np.zeros((olmax+1, olmax+1))
            harm_avgdenom = np.zeros((olmax+1, olmax+1))
        if args.space == 'fourier' or args.space == 'both':
            fft_R1       = np.zeros(omap.shape)
            fft_sumnum   = np.zeros(omap.shape)
            fft_sumdenom = np.zeros(omap.shape)
    
    #Harmonic Calculations
    if args.space == 'harmonic' or args.space == 'both':
        #Response Function Normalization
        if args.fft_normalize == 'yes':
            fft_omap_filt = fft_omap * Al_highres
            omap_filt = enmap.ifft(fft_omap_filt)
            okappalm = curvedsky.map2alm(np.real(omap_filt), lmax=olmax, tweak=True)
            okappalm = idx2lm(okappalm)

        #Current Sim Cross/Auto Spectra in Harmonic Space 
        harm_sim_num = np.real(ikappalm * np.conjugate(okappalm))
        harm_sim_denom = np.real(ikappalm * np.conjugate(ikappalm))
        harm_sim_num = np.where(harm_sim_num==0, np.nan, harm_sim_num)
        harm_sim_denom = np.where(harm_sim_denom==0, np.nan, harm_sim_denom)
    
        #R1: avg of ratios
        harm_sim_R1 = harm_sim_num / harm_sim_denom
        harm_R1 += harm_sim_R1
    
        #R2: ratio of avgs
        harm_avgnum += harm_sim_num
        harm_avgdenom += harm_sim_denom
    
        #Harmonic Statistics
        harm_sim_R1_Ls = [harm_sim_R1[2:, :], harm_sim_R1[2:1000, :1000], harm_sim_R1[2:2000, :2000]]
        harm_sim_R1_means = vecNanStat(harm_sim_R1_Ls, 'mean')
        harm_sim_R1_meds = vecNanStat(harm_sim_R1_Ls, 'median')
        harm_stats_sim = harm_sim_R1_means + harm_sim_R1_meds
        harm_stats_tot.append(harm_stats_sim)
    
    #Fourier Calculations
    if args.space == 'fourier' or args.space == 'both':
        #Current Sim Cross/Auto Spectra in Fourier Space 
        fft_sim_num = np.real(fft_imap * np.conjugate(fft_omap))
        fft_sim_denom = np.real(fft_imap * np.conjugate(fft_imap))
        if simnum == 1:
            np.save(f'/scratch/r/rbond/ymehta3/fft_num_{rank}.npy', fft_sim_num)
            np.save(f'/scratch/r/rbond/ymehta3/fft_denom_{rank}.npy', fft_sim_denom)
    
        #R1: avg of ratios
        fft_sim_R1 = fft_sim_num / fft_sim_denom
        fft_R1 += fft_sim_R1
    
        #R2: ratio of avgs
        fft_sumnum += fft_sim_num
        fft_sumdenom += fft_sim_denom
        
        #Statistics of R1: Fouriers
        fft_sim_R1_means = vecNanStat(fft_sim_R1, 'mean')
        fft_sim_R1_meds = vecNanStat(fft_sim_R1, 'median')
        fft_stats_sim = fft_sim_R1_means + fft_sim_R1_meds
        fft_stats_tot.append(fft_stats_sim)
    
    #Time Diagnostics
    sim_endtime = time.time()
    tot_time = mf.printSimTime(sim_startime, sim_endtime, rank, simnum, rank_Nsims, tot_time)

    simnum += 1

#Harmonic Collection and Saving
if args.space == 'harmonic' or args.space == 'both':
    harm_stats = np.array(harm_stats_tot)

    #Gather Harmonic Quantities
    Nls, Nms = harm_R1.shape
    harm_R1_tot = np.zeros((Nprocs, Nls, Nms))
    harm_avgnum_tot = np.zeros((Nprocs, Nls, Nms))
    harm_avgdenom_tot = np.zeros((Nprocs, Nls, Nms))
    comm.Gather(harm_R1, harm_R1_tot, root=0)
    comm.Gather(harm_avgnum, harm_avgnum_tot, root=0)
    comm.Gather(harm_avgdenom, harm_avgdenom_tot, root=0)

    #Gather Harm Stats Info
    hstats_Ncol = harm_stats.shape[1]
    hstats_count = rows_per_rank * hstats_Ncol
    hstats_disp = np.array([sum(hstats_count[:r]) for r in range(Nprocs)])
    harm_stats_tot = np.zeros((Nsims, hstats_Ncol))
    comm.Gatherv(harm_stats, [harm_stats_tot, hstats_count, hstats_disp, MPI.DOUBLE], root=0)

    if rank == 0:
        harm_R1avg = np.sum(harm_R1_tot, axis=0) / Nsims

        harm_avgnum = np.sum(harm_avgnum_tot, axis=0)  / Nsims
        harm_avgdenom = np.sum(harm_avgdenom_tot, axis=0) / Nsims
        harm_R2avg = harm_avgnum / harm_avgdenom

        #Stats: Writting to Log
        np.savetxt(harm_stats_outpath, harm_stats_tot, header=header_harm)

        #Set Save Names
        savenames = ['harm_R1', 'harm_R2', 'harm_avgnum', 'harm_avgdenom']
        if args.fft_normalize == 'yes':
            for i, name in enumerate(savenames):
                savenames[i] = name + '_Al'
        for i, name in enumerate(savenames):
            savenames[i] = name + '_' + str(Nsims)
        
        #Save Outputs
        saveobjs = [harm_R1avg, harm_R2avg, harm_avgnum, harm_avgdenom]
        for name, obj in zip(savenames, saveobjs):
            np.save(OUTPUT_PATH + name + '.npy', obj)

#Fourier Collection and Saving
if args.space == 'fourier' or args.space == 'both':
    fft_stats = np.array(fft_stats_tot)

    #Gather FFT Quantities
    Nky, Nkx = fft_R1.shape
    fft_R1_tot = np.zeros((Nprocs, Nky, Nkx))
    fft_sumnum_tot = np.zeros((Nprocs, Nky, Nkx))
    fft_sumdenom_tot = np.zeros((Nprocs, Nky, Nkx))
    comm.Gather(fft_R1, fft_R1_tot, root=0)
    comm.Gather(fft_sumnum, fft_sumnum_tot, root=0)
    comm.Gather(fft_sumdenom, fft_sumdenom_tot, root=0)

    #Gather FFT Stats Info
    fstats_Ncol = fft_stats.shape[1]
    fstats_count = rows_per_rank * fstats_Ncol
    fstats_disp = np.array([sum(fstats_count[:r]) for r in range(Nprocs)])
    fft_stats_tot = np.zeros((Nsims, fstats_Ncol))
    comm.Gatherv(fft_stats, [fft_stats_tot, fstats_count, fstats_disp, MPI.DOUBLE], root=0)

    if rank == 0:
        fft_R1avg = np.sum(fft_R1_tot, axis=0) / Nsims

        fft_avgnum = np.sum(fft_sumnum_tot, axis=0)  / Nsims
        fft_avgdenom = np.sum(fft_sumdenom_tot, axis=0)  / Nsims
        fft_R2avg = fft_avgnum / fft_avgdenom

        #Stats: Writting to Log
        np.savetxt(fft_stats_outpath, fft_stats_tot, header=header_fft)

        #Set Save Names
        savenames = ['fft_R2', 'fft_avgnum', 'fft_avgdenom']
        if args.nsims != 0:
            for i, name in enumerate(savenames):
                savenames[i] = name + '_' + str(args.nsims)

        #Save outputs
        saveobjs = [fft_R2, fft_avgnum, fft_avgdenom]
        for name, obj in zip(savenames, saveobjs):
            np.save(OUTPUT_PATH + name + '.npy', obj)

if rank == 0:
    endtime = time.time()
    mf.printTotalTime(startime, endtime, Nthings= Nsims)