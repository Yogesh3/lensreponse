import numpy as np
import healpy as hp
import time
from myfuncs import idx2lm, printTotalTime, readTxt

"""
Calculates the variance of kappa sims.
"""
#Paths
INPUT_PATH = '/project/r/rbond/ymehta3/input_data/kappa_sims/'
# outkappa_path = '/project/r/rbond/jiaqu/kappa_maps/sims/'
outkappa_path = '/scratch/r/rbond/gfarren/unWISExACT_outputs/sims/kappa/'
# okappa_names = INPUT_PATH + 'kappasims_v3.txt'
okappa_names = INPUT_PATH + 'okappas_gerrit.txt'
# OUTPUT_PATH = '/project/r/rbond/ymehta3/output/lensresponse/v3/'
R_names_file = '/project/r/rbond/ymehta3/input_data/kappa_sims/R_gerr_smooth_names.txt'
R_path = "/project/r/rbond/ymehta3/output/lensresponse/gerrit/"
OUTPUT_PATH = '/project/r/rbond/ymehta3/output/lensresponse/gerrit/'

#Read Response Function Names
N_Rs, R_names = readTxt(R_names_file)

#Read Kappa Names
_, knames_tot = readTxt(okappa_names)

start = time.time()

for R_name in R_names:
    start_R = time.time()

    #Split up the Sims
    Nsims = int(R_name[22:25])
    knames = knames_tot[:Nsims]

    var = 0
    for kappa_name in knames:
        kappa_name = kappa_name.rstrip('\n')

        #Load Kappa
        kappa_alm = hp.read_alm(outkappa_path + kappa_name)
        kappa_alm = np.cdouble(kappa_alm)

        #Convert to l,m Grid
        kappa_lm = idx2lm(kappa_alm)

        #Calculate Variance
        var += np.real( kappa_lm * np.conjugate(kappa_lm) )

    var /= Nsims
    var = np.where(var == 0, np.nan, var)

    ellcut = 1000
    factor_list = [32, 64, 128]
    factor_list = np.repeat(factor_list, len(R_names)/len(factor_list))

    for R_name, factor in zip(R_names, factor_list):
        #Load Response Function
        R = np.load(R_path + R_name)

        #Save
        np.save(OUTPUT_PATH + 'okappa_var_' + str(factor) + '_' + str(Nsims) + '.npy', var[:ellcut+1, :ellcut+1] / R[:ellcut+1, :ellcut+1]**2)

    #Time Outputs
    end_R = time.time()
    printTotalTime(start_R, end_R, Nthings=Nsims)

#Time Stats
end = time.time()
printTotalTime(start, end, Nthings=N_Rs, type='response functions')