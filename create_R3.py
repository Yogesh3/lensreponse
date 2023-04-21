import numpy as np
import orphics.maps
from pixell import enmap
import myfuncs as mf
import argparse

#Parser
parser = argparse.ArgumentParser(description='Calculates R3 from numerators of response.py output')
parser.add_argument('--sim_version', type=str, choices=['release', 'nonoise'])
parser.add_argument('--channel', type=str, default='MV', choices=['MV', 'POL'])
args = parser.parse_args()

#Paths
PROJECT_IN = '/project/r/rbond/ymehta3/input_data/kappa_sims/'
PROJECT_OUT = '/project/r/rbond/ymehta3/output/lensresponse/'
outfile_path = PROJECT_OUT + args.sim_version + '/' 
mask_path = '/home/r/rbond/jiaqu/scratch/DR6/downgrade/downgrade/act_mask_20220316_GAL060_rms_70.00_d2sk.fits'
theory_path = PROJECT_IN + 'iclkk_th.txt'
avgnum_names_file = PROJECT_IN + 'avgnum_'+ args.sim_version + '_' + args.channel + '_coarse_names.txt'

#Read Mask
mask_map = enmap.read_map(mask_path)

#Calculate W Factors
w4 = orphics.maps.wfactor(4, mask_map)

#Get Theory Curve
cl_kk_th = np.loadtxt(theory_path)
cl_kk_th[:2] = 1

#Get Numerator Names
_, avgnum_names = mf.readTxt(avgnum_names_file)

for avgnum_name in avgnum_names:
    #Read in Numerator
    avgnum = np.load(PROJECT_OUT + args.sim_version + '/' + avgnum_name)
    
    #Calculate R3
    Nells = avgnum.shape[0]
    R3 = avgnum / w4 / cl_kk_th[:Nells, np.newaxis]

    #Save R3
    savename = avgnum_name.split('_')
    savename[1] = 'R3'
    savename = '_'.join(savename)
    np.save(outfile_path + savename, R3)
