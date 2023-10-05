### Fit the demeaned Tp and Ts data points for all the time intervals of one spatial bin

from os.path import exists, join
from os import makedirs
import numpy as np
from scipy.odr import Data, Model, ODR
from random import choices
from datetime import datetime

from utilities import EstimateVpVs

from matplotlib import pyplot as plt

###

## Inputs
# Bin name
areanm = 'D1'

# Distance name
distnm = 'Distance_2.0-'

# Group name
timenm = 'ShortRange_30days'

# Proprocess folder name
procnm = 'InterceptRemoval_Xcorrmax0.6_Nptsmin7_dTpRange0.050-0.150_Resmax0.005'

# Parent input folder
dirnm_in_prt = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/Archive/TXT/FaultAreas'

# Output parent directories
dirnm_out_arc_prt  = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/Archive/TXT/FaultAreas'
dirnm_out_gmt_prt  = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/GMT/TXTFile/FaultAreas'

# Threshold for removing outliers
numvar = 3

# Number of bootstrap resamplings
numbtstrp = 500

## Iterate over each bin
print('Processing '+areanm+'...')

# Estimate the Vp/Vs ratio for each bin
path_in = join(dirnm_in_prt, areanm, distnm, timenm, procnm, 'DiffTimes.dat')
with open(path_in) as fp:
    lines = fp.readlines()

list_evstinfo = []
list_timediff_p = []
list_timediff_s = []
for line in lines:
    fields = line.split()
    evid1 = fields[0]
    evid2 = fields[1]
    stnm = fields[2]
    timediff_p = float(fields[3])
    timediff_s = float(fields[4])

    list_timediff_p.append(timediff_p)
    list_timediff_s.append(timediff_s)
    list_evstinfo.append((evid1, evid2, stnm))

# plt.scatter(list_timediff_p, list_timediff_s)
# plt.show()

list_timediff_p_filt, list_timediff_s_filt, list_evstinfo_filt, r_opt_filt, std_r_filt, rms_filt, r_opt, std_r, rms = EstimateVpVs(list_timediff_p, list_timediff_s, list_evstinfo, numvar, numbtstrp)

## Save the results
print('Saving the results...')
dirnm_out_arc = join(dirnm_out_arc_prt, areanm, distnm, timenm, procnm)
if not exists(dirnm_out_arc):
    makedirs(dirnm_out_arc)
    print(dirnm_out_arc+' is created.')

dirnm_out_gmt = join(dirnm_out_gmt_prt, areanm, distnm, timenm, procnm)
if not exists(dirnm_out_gmt):
    makedirs(dirnm_out_gmt)
    print(dirnm_out_gmt+' is created.')

# Save the filtered data points
path_out = join(dirnm_out_arc, 'DiffTimes_filtered.dat')
numlin = len(list_timediff_p_filt)
with open(path_out, 'w') as fp:
    for i in range(numlin):
        evid1 = list_evstinfo_filt[i][0]
        evid2 = list_evstinfo_filt[i][1]
        stnm = list_evstinfo_filt[i][2]
        timediff_p = list_timediff_p_filt[i]
        timediff_s = list_timediff_s_filt[i]
        line = evid1+' '+evid2+' '+stnm+' '+str(timediff_p)+' '+str(timediff_s)+'\n'
        fp.write(line)
print(path_out+' is written.')

print(path_out+' is written.')

path_out = join(dirnm_out_gmt, 'DiffTimes_filtered.xy')
array_timediff_p_filt = np.array(list_timediff_p_filt)
array_timediff_s_filt = np.array(list_timediff_s_filt)
array_out = np.stack((array_timediff_p_filt, array_timediff_s_filt), axis=-1)
np.savetxt(path_out, array_out)

print(path_out+' is written.')

# Save the Vp/Vs ratios
path_out = join(dirnm_out_arc, 'OptimumVpVs_ODR_filtered.dat')
with open(path_out, 'w') as fp:
    fp.write(str(r_opt_filt)+' '+str(std_r_filt))

print(path_out+' is written.')

path_out = join(dirnm_out_arc, 'OptimumVpVs_ODR.dat')
with open(path_out, 'w') as fp:
    fp.write(str(r_opt)+' '+str(std_r))

print(path_out+' is written.')

path_out = join(dirnm_out_arc, 'RMS_ODR_filtered.dat')
with open(path_out, 'w') as fp:
    fp.write(str(rms_filt))

print(path_out+' is written.')

path_out = join(dirnm_out_arc, 'RMS_ODR_.dat')
with open(path_out, 'w') as fp:
    fp.write(str(rms))

print(path_out+' is written.')

path_out = join(dirnm_out_gmt, 'OptimumVpVs_ODR_filtered.dat')
with open(path_out, 'w') as fp:
    fp.write(str(r_opt_filt)+' '+str(std_r_filt))

print(path_out+' is written.')

path_out = join(dirnm_out_gmt, 'OptimumVpVs_ODR.dat')
with open(path_out, 'w') as fp:
    fp.write(str(r_opt)+' '+str(std_r))

print(path_out+' is written.')

path_out = join(dirnm_out_gmt, 'RMS_ODR_filtered.dat')
with open(path_out, 'w') as fp:
    fp.write(str(rms_filt))

print(path_out+' is written.')

path_out = join(dirnm_out_gmt, 'RMS_ODR_.dat')
with open(path_out, 'w') as fp:
    fp.write(str(rms))

print(path_out+' is written.')

print('Results saved.')
