import numpy as np
from pixell import enmap
import argparse
import myfuncs as mf
import time
from pathlib import Path

"""
Smooths given response function.
"""

#Parser
parser = argparse.ArgumentParser(description= "Smooths given response functions")
parser.add_argument('--sim_version', type=str, choices=['v3', 'v2', 'gerrit', 'release', 'nonoise'])
parser.add_argument('--names_file', type=str, default=None)
parser.add_argument('--channel', type=str, choices=['MV', 'POL'])
parser.add_argument('--factors', type=int, action='append', help='Smoothing factor(s)')
parser.add_argument('--R_def', type=str, choices=['R2', 'R3'])
parser.add_argument('--only_max_sims', type=mf.str2bool, default='no', help='Only smooth response function corresponding to the maximum number of sims? If not, uses response functions from the text file')
parser.add_argument('--clobber', type=mf.str2bool, default='no')
parser.add_argument('--outdir', type=str, default='project', choices=['project', 'scratch'])
args = parser.parse_args()

#Names File
if args.names_file is None:
    if args.sim_version == 'release' or args.sim_version == 'nonoise':
        R_names_file = '/project/r/rbond/ymehta3/input_data/kappa_sims/' + '_'.join([args.R_def, args.sim_version, args.channel]) + '_coarse_names.txt'
    else:
        R_names_file = '/project/r/rbond/ymehta3/input_data/kappa_sims/R_' + args.sim_version + '_coarse_names.txt'
else:
    R_names_file = args.names_file

#Paths 
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
if args.only_max_sims:
    R_names = [R_names[-1]]

#Smoothing Factors
factor_list = args.factors
# factor_list = [32]

for iR, R_name in enumerate(R_names):
    start = time.time()
    
    #Read Response Function
    R_name = str(R_name)
    R2 = np.load(R_path + R_name)
    R2_map = enmap.enmap(R2)
    
    for factor in factor_list:
        #Outputs
        if args.sim_version == 'release' or args.sim_version == 'nonoise':
            outname = args.R_def + '_' + args.channel + '_smoothed_' + str(factor) + '_' + args.sim_version + '_' + R_name[-7:-4] + '.npy'        
        else:
            outname = args.R_def + '_smoothed_' + str(factor) + '_' + args.sim_version + '_' + R_name[-7:-4] + '.npy'        
        write_to_path = OUTPATH + outname
        pathobj = Path(write_to_path)
        if not args.clobber and pathobj.is_file():
            continue

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
        np.save(write_to_path, R_blocked)

    #Time Diagnostics
    end = time.time()
    time_min, time_sec = divmod(end-start, 60)
    time_sec = round( time_sec )
    print(f'\nTook {time_min:.0f} min and {time_sec} sec for {R_name}  ({iR+1}/{N_Rs})')
    print('\n')