## Prepare the data points for linear regression

### The input event information is in Cartesian coordinates!

from os import makedirs
from os.path import join, exists
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

from utilities import RemoveDataIntercept, BinSlopeAndDiffPtime

from netCDF4 import Dataset

###

## Inputs

# Input cross-correlation results
dirnm_in_prt = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/Archive/TXT/FaultAreas'

# Bin name
areanm = 'E2'

# Inter-event distance name
distnm = 'Distance_2.0-'

# Time interval name
timenm = 'ShortRange_30days'

# Minimum cross-correlation peak value
xcorrmax_min = 0.6

# Minimum number of travel-time pairs for each event pair
numxcorr_min = 7

# Minimum line-fitting residual
meanres_max = 0.005

dtprng_min = 0.05
dtprng_max = 0.15

# Output parent directories
dirnm_out_arc_prt  = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/Archive/TXT/FaultAreas'
dirnm_out_gmt_txt_prt  = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/GMT/TXTFile/FaultAreas'
dirnm_out_gmt_grd_prt  = '/Users/til008/Research/WorkingDirectory/GofarInSituVpVs/GMT/GRDFile/FaultAreas'


## Read the input files
print('Processing '+areanm+'...')

# Read the input time interval list


# Event info
print('Reading the event information...')
path_in = join(dirnm_in_prt, areanm, 'EventInfo.dat')
with open(path_in) as fp:
    lines_in = fp.readlines()

dict_evinfo = {}
for line in lines_in:
    fields = line.split()
    evid = fields[0]
    x_ev = float(fields[7])
    y_ev = float(fields[8])
    dep_ev = float(fields[9])

    dict_evinfo[evid] = (x_ev, y_ev, dep_ev)

# Xcorr info
path_in = join(dirnm_in_prt, areanm, distnm, timenm, 'XcorrInfo.dat')

with open(path_in) as fp:
    lines_xcorr = fp.readlines()

# Remove the intercepts
output = RemoveDataIntercept(xcorrmax_min, numxcorr_min, meanres_max, dtprng_min, dtprng_max, dict_evinfo, lines_xcorr)
flag_err = output.ErrorFlag
list_tdiffinfo = output.DiffTimeInfo
list_timediff_p = output.PdiffTime
list_timediff_s = output.SdiffTime
numpair_in = output.InputEventPairNumber
numpair_int_out = output.IntermediateOutputEventPairNumber
numpair_fin_out = output.FinalOutputEventPairNumber
list_pairinfo_in = output.InputPairinfo
list_pairinfo_int_out = output.IntermediateOutputPairInfo
list_pairinfo_fin_out = output.FinalOutputPairInfo

list_r_out = output.Slope
list_r_filt_out = output.FilteredSlope
list_b_out = output.Intercept
list_b_filt_out = output.FilteredIntercept
list_dtprng_out = output.dTpRange

perc_numpair_int = float(numpair_int_out)/float(numpair_in)*100
perc_numpair_fin = float(numpair_fin_out)/float(numpair_int_out)*100


print(str(numpair_in)+' event pairs in the input file.')
print(str(numpair_int_out)+' ('+str(perc_numpair_int)+'% retaining rate in the first step) event pairs remain after the CC value criterion is applied.')
print(str(numpair_fin_out)+' ('+str(perc_numpair_fin)+'% retaining rate in the second step) event pairs remain after all the QC criteria are applied.')

if flag_err == -1:
    # Plot the data points
    fig, ax = plt.subplots()
    ax.scatter(list_timediff_p, list_timediff_s)
    ax.plot([-0.2, 0.2], [-0.36, 0.36], 'r--')
    ax.plot([-0.2, 0.2], [-0.34, 0.34], 'r')
    ax.set_xlabel('Diff. P time (s)')
    ax.set_ylabel('Diff. S time (s)')
    ax.set_aspect('equal')
    plt.show()

    # Compute the median of the slopes
    vect_r = np.array(list_r_out)
    med = np.median(vect_r)
    print('The median of the unfiltered slope is '+str(med))

    vect_r = np.array(list_r_filt_out)
    med = np.median(vect_r)
    print('The median of the filtered slope is '+str(med))

    # Plot the Vp/Vs and b histograms
    fig, (ax1, ax2) = plt.subplots(2, 1)

    edges = np.linspace(-3, 3, num=121)
    ax1.hist(list_r_out, bins=edges)
    ax1.hist(list_r_filt_out, bins=edges)

    edges = np.linspace(-0.5, 0.5, num=121)
    ax2.hist(list_b_out, bins=edges)
    ax2.hist(list_b_filt_out, bins=edges)

    plt.show()

    # Compute the joint distribution between dTp and r
    edges_dtprng = np.linspace(0, 1, 101)
    edges_r = np.linspace(-1, 4, 51)
    array_binfreq, grids_dtprng, grids_r = BinSlopeAndDiffPtime(list_dtprng_out, list_r_out, edges_dtprng, edges_r)

    fig, ax = plt.subplots()
    ax.imshow(array_binfreq, extent=(np.amin(grids_dtprng), np.amax(grids_dtprng), np.amin(grids_r), np.amax(grids_r)), cmap='plasma', origin='lower')
    ax.set_aspect('auto')
    plt.show()

    # Save the results
    # Save the cross-correlation results
    print('Saving the results...')
    print('Saving to the archive folder...')
    procnm = 'InterceptRemoval_Xcorrmax'+'{:.1f}'.format(xcorrmax_min)+'_Nptsmin'+str(numxcorr_min)+'_dTpRange'+'{:.3f}'.format(dtprng_min)+'-'+'{:.3f}'.format(dtprng_max)+'_Resmax'+'{:.3f}'.format(meanres_max)
    dirnm_out = join(dirnm_out_arc_prt, areanm, distnm, timenm, procnm)
    if not exists(dirnm_out):
        makedirs(dirnm_out)
        print(dirnm_out+' is created.')

    num_timediff_out = len(list_timediff_p)
    path_out = join(dirnm_out ,'DiffTimes.dat')
    with open(path_out,'w') as fp:
        for i in range(num_timediff_out):
            tdiffinfo = list_tdiffinfo[i]
            evid1 = tdiffinfo[0]
            evid2 = tdiffinfo[1]
            stnm = tdiffinfo[2]
            timediff_p = list_timediff_p[i]
            timediff_s = list_timediff_s[i]
            fp.write('%s %s %s %f %f\n' % (evid1, evid2, stnm, timediff_p, timediff_s))
            print(path_out+' is written.')

    path_out = join(dirnm_out ,'RetainingRate_int.dat')
    with open(path_out,'w') as fp:
        fp.write(str(perc_numpair_int)+'\n')
        print(path_out+' is written.')

    path_out = join(dirnm_out ,'RetainingRate_fin.dat')
    with open(path_out,'w') as fp:
        fp.write(str(perc_numpair_fin)+'\n')
    print(path_out+' is written.')

    path_out = join(dirnm_out ,'InputEventPairInfo.dat')
    with open(path_out,'w') as fp:
        for pairinfo in list_pairinfo_in:
            evid1 = pairinfo[0]
            evid2 = pairinfo[1]
            fp.write(evid1+' '+evid2+'\n')
    print(path_out+' is written.')

    path_out = join(dirnm_out ,'OutputEventPairInfo_int.dat')
    with open(path_out,'w') as fp:
        for pairinfo in list_pairinfo_int_out:
            evid1 = pairinfo[0]
            evid2 = pairinfo[1]
            fp.write(evid1+' '+evid2+'\n')
    print(path_out+' is written.')

    path_out = join(dirnm_out ,'OutputEventPairInfo_fin.dat')
    with open(path_out,'w') as fp:
        for pairinfo in list_pairinfo_fin_out:
            evid1 = pairinfo[0]
            evid2 = pairinfo[1]
            fp.write(evid1+' '+evid2+'\n')
    print(path_out+' is written.')

    print('Saving to the GMT folder...')
    dirnm_out = join(dirnm_out_gmt_txt_prt, areanm, distnm, timenm, procnm)
    if not exists(dirnm_out):
        makedirs(dirnm_out)
        print(dirnm_out+' is created.')

    path_out = join(dirnm_out, 'DiffTimes.xy')
    with open(path_out,'w') as fp:
        for i in range(num_timediff_out):
            timediff_p = list_timediff_p[i]
            timediff_s = list_timediff_s[i]
            fp.write('%f %f\n' % (timediff_p, timediff_s))

    # Save the 2D distribution
    dirnm_out = join(dirnm_out_gmt_grd_prt, areanm, distnm, timenm, procnm)
    if not exists(dirnm_out):
        makedirs(dirnm_out)
        print(dirnm_out+' is created.')

    filenm_out = 'dTpRangeSlopeDistribution.grd'
    path_out = join(dirnm_out, filenm_out)
    rootgrp = Dataset(path_out, 'w')

    dim_dtprng = rootgrp.createDimension('dtprange', len(grids_dtprng))
    dim_r = rootgrp.createDimension('r', len(grids_r))

    var_dtprng = rootgrp.createVariable('dtprange', 'f4', ('dtprange'))
    var_r = rootgrp.createVariable('r', 'f4', ('r'))
    var_freq = rootgrp.createVariable('frequency', 'f4', ('r', 'dtprange'))

    var_dtprng[:] = grids_dtprng
    var_r[:] = grids_r
    var_freq[:] = array_binfreq

    rootgrp.close()
    print(path_out+' is written.')

    print('Results saved.')
