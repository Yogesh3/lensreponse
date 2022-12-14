import numpy as np
from pixell import enmap
import argparse
import myfuncs as mf
import time

"""
Smooths given response function.
"""

#Parser
parser = argparse.ArgumentParser(description= "Smooths given response functions")
parser.add_argument('--sim_version', type=str, default='gerrit', choices=['v3', 'v2', 'gerrit'])
parser.add_argument('--outdir', type=str, default='project', choices=['project', 'scratch'])
args = parser.parse_args()

#Paths 
if args.sim_version == 'gerrit':
    R_names_file = '/project/r/rbond/ymehta3/input_data/kappa_sims/R_gerr_coarse_names.txt'
if (args.outdir).lower() == 'project':
    OUTDIR = '/project/r/rbond/ymehta3/output/lensresponse/'
elif (args.outdir).lower() == 'scratch':
    OUTDIR = '/scratch/r/rbond/ymehta3/output/lensresponse/'
else:
    OUTDIR = args.outdir
R_path = OUTDIR + args.sim_version + "/"
OUTPATH = OUTDIR + args.sim_version + "/"


#Read Response Functions Names
N_Rs, R_names = mf.readTxt(R_names_file)

factor_list = [32, 64, 128]
# factor_list = [32]

for iR, R_name in enumerate(R_names):
    start = time.time()
    
    #Read Response Function
    R_name = str(R_name)
    R2 = np.load(R_path + R_name)
    R2_map = enmap.enmap(R2)
    
    for factor in factor_list:
        #Smooth 
        R_lowres = R2_map.downgrade(factor, op=np.nanmean)

        #Fill in the Diagonal
        for i in range(len(R_lowres) - 1):
            R_lowres[i, i+1] = (R_lowres[i+1, i+1] + R_lowres[i, i]) / 2

        #Interpolate
        R_blocked = R_lowres.project(R2_map.shape, R2_map.wcs, order=1)

        iborder = int(factor/2)

        #Fill in the First ems
        slopes = R_blocked[(iborder):, iborder+3] - R_blocked[(iborder):, iborder+2]
        yintercepts = R_blocked[(iborder):, iborder]
        xs = np.arange(iborder) + 1
        leftbox = np.ones( R_blocked[(iborder):, :iborder].shape )
        extrapolated = yintercepts[..., None]  -  slopes[..., None] * xs[None, ::-1] * leftbox
        R_blocked[iborder:, :iborder] = extrapolated

        #Fill in the First ells
        slopes = R_blocked[iborder+3, :iborder+1] - R_blocked[iborder+2, :iborder+1]
        yintercepts = R_blocked[iborder, :iborder+1]
        xs = np.arange(iborder) + 1
        leftbox = np.ones( R_blocked[:iborder, :iborder+1].shape )
        extrapolated = yintercepts[None, ...]  -  slopes[None, ...] * xs[::-1, None] * leftbox
        R_blocked[:iborder, :iborder+1] = extrapolated
        
        #Save
        np.save(OUTPATH + 'R2_smoothed_' + str(factor) + '_' + args.sim_version + '_' + R_name[8:11] + '.npy', R_blocked)

    #Time Diagnostics
    end = time.time()
    time_min, time_sec = divmod(end-start, 60)
    time_sec = round( time_sec )
    print(f'\nTook {time_min:.0f} min and {time_sec} sec for response function {iR+1}/{N_Rs}')
    print('\n')