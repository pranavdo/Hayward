## Find lambda3/lambda1 for an event cluster
#Faster, Optimize it 
#Parallel Computing, Pytorch (useful tool for parallel processing), Convolution, Julia (another language for parallel processing, existing code for convolution) -> Task for the next week
#Workflow for what I am going to do? (What is the optimizing process looking like)
#
def Geo2Loc(list_evcoord):
    import numpy as np

    re = 6371

    list_evlo = []
    list_evla = []
    list_evdp = []

    for evcoord in list_evcoord:
        evlo = evcoord[0]
        evla = evcoord[1]
        evdp = evcoord[2]

        list_evlo.append(evlo)
        list_evla.append(evla)
        list_evdp.append(evdp)

    array_evlo = np.array(list_evlo)
    array_evla = np.array(list_evla)
    array_evdp = np.array(list_evdp)

    ctrlo = np.mean(array_evlo)
    ctrla = np.mean(array_evla)
    ctrdp = np.mean(array_evdp)

    array_evlo_demean = array_evlo-ctrlo
    array_evla_demean = array_evla-ctrla
    array_evud = array_evdp-ctrdp

    array_evsn = array_evla_demean/180*np.pi*re
    array_evwe = array_evlo_demean/180*np.pi*re*np.cos(ctrla/180*np.pi)

    return array_evwe, array_evsn, array_evud

## Find eigenvalue ratios for an event cluster. The event coordinates must be in local coordinates!
def FindEigenValueRatio_xyz(array_evcoord):
    import numpy as np
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    pca.fit(array_evcoord)

    array_eigval = pca.singular_values_
    eigval1 = array_eigval[0]
    eigval2 = array_eigval[1]
    eigval3 = array_eigval[2]
    rat = eigval1/eigval3
    return rat

## Find eigenvalue ratio for a set of Tp and Ts measurements
def FindEigenValueRatio_tpts(array_tpts):
    import numpy as np
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(array_tpts)

    array_eigval = pca.singular_values_
    eigval1 = array_eigval[0]
    eigval2 = array_eigval[1]
    rat = eigval1/eigval2
    return rat

# Read and parse the output file of deptable
def ReadDepTable(path_in):
    import numpy as np

    with open(path_in) as fp:
        lines = fp.readlines()

    line = lines[1]
    fields = line.split()
    numdist = int(fields[0])
    numdep = int(fields[1])

    line = lines[2]
    fields = line.split()
    array_dep = np.array(list(map(float, fields)))

    array_dist = np.zeros((numdist, 1))
    array_tt = np.zeros((numdist, numdep))

    for i in range(numdist):
        line = lines[i+3]
        fields = line.split()

        array_dist[i] = float(fields[0])
        array_tt[i, :] = np.array(list(map(float, fields[1:])))

    return array_tt, array_dist, array_dep


## Extract the time shifts from xcorr data without demeaning or removing intercepts
def ExtractXcorrTimeShifts(lines_xcorr):
    list_timeshift_p = []
    list_timeshift_s = []

    numlin = len(lines_xcorr)
    ind_lin = 0
    list_stnm = []
    list_timeshift_p = []
    list_timeshift_s = []
    while ind_lin < numlin:
        # Read the two event IDs and the number of cross-correlation values for this two events
        line = lines_xcorr[ind_lin]
        fields = line.split()
        evid1 = fields[0]
        evid2 = fields[1]
        numxcorr = int(fields[2])

        for i in range(numxcorr):
            ind_lin = ind_lin+1
            line = lines_xcorr[ind_lin]
            fields = line.split()
            stnm = fields[0]
            timeshift_p = float(fields[2])
            #xcorrmax_p = float(fields[3])
            timeshift_s = float(fields[4])
            #xcorrmax_s = float(fields[5])

            list_timeshift_p.append(timeshift_p)
            list_timeshift_s.append(timeshift_s)

        ind_lin = ind_lin+1

    return list_timeshift_p, list_timeshift_s

## Demean the data for a single input file
# Station names are also output, Dec 20, 2021

def DemeanData_DeviationReject_Gofar(xcorrmax_min, numxcorr_min, devtimeshift_max, dict_ev, lines_xcorr):
    import numpy as np

    ## Process the cross-correlation results for each event pair
    numlin = len(lines_xcorr)
    ind = 0
    list_tdiffinfo_out = []
    list_timediff_p = []
    list_timediff_s = []
    list_evid_out = []
    list_evx_out = []
    list_evy_out = []
    list_evz_out = []

    numpair_proc = 0
    numpair_fin_out = 0
    print('Processing each event pair')
    while ind < numlin:
        # Read the two event IDs and the number of cross-correlation values for this two events
        line = lines_xcorr[ind]
        fields = line.split()
        evid1 = fields[0]
        evid2 = fields[1]
        numxcorr = int(fields[2])

        if numxcorr < numxcorr_min:
            ind = ind+numxcorr+1
        else:
            list_stnm = []
            list_timeshift_p = []
            list_timeshift_s = []
            for i in range(numxcorr):
                ind = ind+1
                line = lines_xcorr[ind]
                fields = line.split()
                stnm = fields[0]
                timeshift_p = float(fields[2])
                xcorrmax_p = float(fields[3])
                timeshift_s = float(fields[4])
                xcorrmax_s = float(fields[5])

                #print((xcorrmax_p,xcorrmax_s))

                if xcorrmax_p >= xcorrmax_min and xcorrmax_s >= xcorrmax_min:
                    list_stnm.append(stnm)
                    list_timeshift_p.append(timeshift_p)
                    list_timeshift_s.append(timeshift_s)

            numxcorr = len(list_timeshift_p)
            if numxcorr >= numxcorr_min:
                medtimeshift_p = np.median(list_timeshift_p)
                medtimeshift_s = np.median(list_timeshift_s)

                list_stnm_pass = []
                list_timeshift_p_pass = []
                list_timeshift_s_pass = []
                for j in range(numxcorr):
                    stnm = list_stnm[j]
                    timeshift_p = list_timeshift_p[j]
                    timeshift_s = list_timeshift_s[j]

                    if abs(timeshift_p-medtimeshift_p) <= devtimeshift_max and abs(timeshift_s-medtimeshift_s) <= devtimeshift_max:
                        list_stnm_pass.append(stnm)
                        list_timeshift_p_pass.append(timeshift_p)
                        list_timeshift_s_pass.append(timeshift_s)

                list_stnm = list_stnm_pass
                list_timeshift_p = list_timeshift_p_pass
                list_timeshift_s = list_timeshift_s_pass
                numxcorr = len(list_timeshift_p)

                if numxcorr >= numxcorr_min:
                    array_timediff_p = np.array(list_timeshift_p)-np.mean(list_timeshift_p)
                    array_timediff_s = np.array(list_timeshift_s)-np.mean(list_timeshift_s)

                    list_pairinfo = []
                    for k in range(numxcorr):
                        stnm = list_stnm[k]
                        list_pairinfo.append((evid1, evid2, stnm))

                    list_tdiffinfo_out.extend(list_pairinfo)
                    list_timediff_p.extend(list(array_timediff_p))
                    list_timediff_s.extend(list(array_timediff_s))
                    numpair_fin_out = numpair_fin_out+1

                    if not evid1 in list_evid_out:
                        list_evid_out.append(evid1)

                        evcoord = dict_ev[evid1]
                        evx = evcoord[0]
                        evy = evcoord[1]
                        evz = evcoord[2]
                        list_evx_out.append(evx)
                        list_evy_out.append(evy)
                        list_evz_out.append(evz)

                    if not evid2 in list_evid_out:
                        list_evid_out.append(evid2)

                        evcoord = dict_ev[evid2]
                        evx = evcoord[0]
                        evy = evcoord[1]
                        evz = evcoord[2]
                        list_evx_out.append(evx)
                        list_evy_out.append(evy)
                        list_evz_out.append(evz)


            ind = ind+1

        numpair_proc = numpair_proc+1
        if numpair_proc%1000 == 0:
            print(str(numpair_proc)+' event pairs processed.')

    numev_out = len(list_evid_out)
    num_timediff_out = len(list_timediff_p)
    print('In total, '+str(numev_out)+' events, '+str(numpair_fin_out)+' event pairs, and '+str(num_timediff_out)+' differential times included.')

    if numev_out < 4:
        print('Fewer than 4 events included. The cluster cannot be processed.')
        flag_err = -1
        asprat = np.nan

        return flag_err, list_tdiffinfo_out, list_timediff_p, list_timediff_s, list_evid_out, list_evx_out, list_evy_out, list_evz_out, asprat
    else:
        flag_err = 1

    ## Compute the eigenvalue ratio of the cluster
    list_evcoord_out = list(zip(list_evx_out, list_evy_out, list_evz_out))
    array_evx = np.array(list_evx_out)
    array_evy = np.array(list_evy_out)
    array_evz = np.array(list_evz_out)
    array_evcoord = np.stack((array_evx, array_evy, array_evz), axis=-1)
    asprat = FindEigenValueRatio_xyz(array_evcoord)
    print('The eigenvalue ratio of the cluster is '+'{:.2f}'.format(asprat))

    return flag_err, list_tdiffinfo_out, list_timediff_p, list_timediff_s, list_evid_out, list_evx_out, list_evy_out, list_evz_out, asprat


## wig_prf_avg the intercept term and remove it from the differential travel-time of each event pair
def LinearModel_intercept(beta, x):
    y = beta[0]+beta[1]*x
    return y

def RemoveDataIntercept(xcorrmax_min, numxcorr_min, meanres_max, dtprng_min, dtprng_max, dict_ev, lines_xcorr):
    import numpy as np
    from scipy.odr import Data, Model, ODR

    from matplotlib import pyplot as plt

    ## Define the output class
    class InterceptRemovalOutput:
        def __init__(self, flag_err, list_tdiffinfo_out, list_timediff_p_out, list_timediff_s_out, list_evid_out, numpair_in, numpair_int_out, numpair_fin_out, list_pairinfo_in, list_pairinfo_int_out, list_pairinfo_fin_out, list_r, list_r_filt, list_b, list_b_filt, list_dtprng, list_dtprng_filt):
            self.ErrorFlag = flag_err
            self.DiffTimeInfo = list_tdiffinfo_out
            self.PdiffTime = list_timediff_p_out
            self.SdiffTime = list_timediff_s_out
            self.EventID = list_evid_out
            self.InputEventPairNumber = numpair_in
            self.IntermediateOutputEventPairNumber = numpair_int_out
            self.FinalOutputEventPairNumber = numpair_fin_out
            self.InputPairinfo = list_pairinfo_in
            self.IntermediateOutputPairInfo = list_pairinfo_int_out
            self.FinalOutputPairInfo = list_pairinfo_fin_out
            self.Slope = list_r
            self.FilteredSlope = list_r_filt
            self.Intercept = list_b
            self.FilteredIntercept = list_b_filt
            self.dTpRange = list_dtprng
            self.FiltereddTpRange = list_dtprng_filt

    ## Process the cross-correlation results for each event pair
    numlin = len(lines_xcorr)
    ind_lin = 0
    list_tdiffinfo_out = []
    list_pairinfo_in = []
    list_pairinfo_int_out = []
    list_pairinfo_fin_out = []
    list_timediff_p = []
    list_timediff_s = []
    list_evid_out = []

    numpair_proc = 0
    numpair_in = 0
    list_r = []
    list_b = []
    list_dtprng = []
    list_dtp_evpair = []
    list_dts_evpair = []
    list_info_evpair_int = []
    list_tdiffinfo_evpair_fin = []
    print('Processing each event pair...')
    while ind_lin < numlin:
        # Read the two event IDs and the number of cross-correlation values for this two events
        line = lines_xcorr[ind_lin]
        fields = line.split()
        evid1 = fields[0]
        evid2 = fields[1]
        numxcorr = int(fields[2])

        list_pairinfo_in.append((evid1, evid2))

        if numxcorr < numxcorr_min:
            ind_lin = ind_lin+numxcorr+1
        else:
            list_tdiffinfo = []
            list_timeshift_p = []
            list_timeshift_s = []
            for i in range(numxcorr):
                ind_lin = ind_lin+1
                line = lines_xcorr[ind_lin]
                fields = line.split()
                stnm = fields[0]
                timeshift_p = float(fields[2])
                xcorrmax_p = float(fields[3])
                timeshift_s = float(fields[4])
                xcorrmax_s = float(fields[5])

                if xcorrmax_p >= xcorrmax_min and xcorrmax_s >= xcorrmax_min:
                    list_tdiffinfo.append((evid1, evid2, stnm))
                    list_timeshift_p.append(timeshift_p)
                    list_timeshift_s.append(timeshift_s)

            numxcorr = len(list_timeshift_p)
            if numxcorr >= numxcorr_min:
                list_pairinfo_int_out.append((evid1, evid2))

            # Remove the data point with the largest residual and perform the fitting again
            while numxcorr >= numxcorr_min:
                array_timeshift_p = np.array(list_timeshift_p)
                array_timeshift_s = np.array(list_timeshift_s)

                model_odr  = Model(LinearModel_intercept)
                data_odr = Data(array_timeshift_p, array_timeshift_s)
                odr = ODR(data_odr, model_odr, beta0=[0, 1.73])
                output_odr = odr.run()

                meanres = np.sqrt(output_odr.sum_square/numxcorr)

                # Determine if the residual is below the threshold
                if meanres < meanres_max:
                    b_opt = output_odr.beta[0]
                    r_opt = output_odr.beta[1]

                    list_r.append(r_opt)
                    list_b.append(b_opt)

                    # Remove the intercept term from the data
                    array_timediff_s = array_timeshift_s-b_opt
                    array_timediff_p = array_timeshift_p

                    dtp_rng = np.max(array_timediff_p)-np.min(array_timediff_p)

                    list_dtprng.append(dtp_rng)
                    list_dtp_evpair.append(array_timediff_p)
                    list_dts_evpair.append(array_timediff_s)
                    list_tdiffinfo_evpair_fin.append(list_tdiffinfo)
                    list_pairinfo_fin_out.append((evid1, evid2))

                    break

                array_res = np.square(output_odr.delta)+np.square(output_odr.eps)
                ind_max = np.argmax(array_res)

                list_tdiffinfo.pop(ind_max)
                list_timeshift_p.pop(ind_max)
                list_timeshift_s.pop(ind_max)

                numxcorr = len(list_timeshift_p)

            ind_lin = ind_lin+1

    ## Remove the event pairs with slopes outside the acceptable range
    vect_r = np.array(list_r)
    vect_ind = np.argwhere((vect_r > 0.5) & (vect_r < 3))
    vect_ind = vect_ind[:, 0]
    list_r_filt = [list_r[i] for i in vect_ind]
    list_b_filt = [list_b[i] for i in vect_ind]
    list_dtprng_filt = [list_dtprng[i] for i in vect_ind]
    list_dtp_evpair = [list_dtp_evpair[i] for i in vect_ind]
    list_dts_evpair = [list_dts_evpair[i] for i in vect_ind]
    list_tdiffinfo_evpair_fin = [list_tdiffinfo_evpair_fin[i] for i in vect_ind]
    list_pairinfo_fin_out = [list_pairinfo_fin_out[i] for i in vect_ind]

    ## Remove the event pairs with dtp range outside the acceptable range
    vect_dtprng = np.array(list_dtprng_filt)
    vect_ind = np.argwhere((vect_dtprng > dtprng_min) & (vect_dtprng < dtprng_max))
    vect_ind = vect_ind[:, 0]
    list_r_filt = [list_r_filt[i] for i in vect_ind]
    list_b_filt = [list_b_filt[i] for i in vect_ind]
    list_dtprng_filt = [list_dtprng_filt[i] for i in vect_ind]
    list_dtp_evpair = [list_dtp_evpair[i] for i in vect_ind]
    list_dts_evpair = [list_dts_evpair[i] for i in vect_ind]
    list_tdiffinfo_evpair_fin = [list_tdiffinfo_evpair_fin[i] for i in vect_ind]
    list_pairinfo_fin_out = [list_pairinfo_fin_out[i] for i in vect_ind]

    ## Assemble the output data
    numpair_in = len(list_pairinfo_in)

    numpair_int_out = len(list_pairinfo_int_out)

    numpair_fin_out = len(list_pairinfo_fin_out)
    list_timediff_p_out = []
    list_timediff_s_out = []
    list_tdiffinfo_out = []
    list_evid_out = []
    for i in range(numpair_fin_out):
        array_timediff_p = list_dtp_evpair[i]
        array_timediff_s = list_dts_evpair[i]
        list_tdiffinfo = list_tdiffinfo_evpair_fin[i]

        list_timediff_p_out.extend(list(array_timediff_p))
        list_timediff_s_out.extend(list(array_timediff_s))
        list_tdiffinfo_out.extend(list_tdiffinfo)

        evid1 = list_tdiffinfo[0][0]
        evid2 = list_tdiffinfo[0][1]

        if not evid1 in list_evid_out:
            list_evid_out.append(evid1)

        if not evid2 in list_evid_out:
            list_evid_out.append(evid2)

    numev_out = len(list_evid_out)
    num_timediff_out = len(list_timediff_p_out)
    print('In total, '+str(numev_out)+' events, '+str(numpair_fin_out)+' event pairs, and '+str(num_timediff_out)+' differential times included.')

    if numev_out < 4:
        print('Fewer than 4 events included. The cluster cannot be processed.')
        flag_err = 1
    else:
        flag_err = -1

    output = InterceptRemovalOutput(flag_err, list_tdiffinfo_out, list_timediff_p_out, list_timediff_s_out, list_evid_out, numpair_in, numpair_int_out, numpair_fin_out, list_pairinfo_in, list_pairinfo_int_out, list_pairinfo_fin_out, list_r, list_r_filt, list_b , list_b_filt, list_dtprng, list_dtprng_filt)
    return output

## Bin the r and dtp output by RemoveDataIntercept
def BinSlopeAndDiffPtime(list_dtp, list_r, edges_dtprng, edges_r):
    import numpy as np
    import pandas as pd

    dtprng_max = np.amax(edges_dtprng)
    dtprng_min = np.amin(edges_dtprng)

    r_max = np.amax(edges_r)
    r_min = np.amin(edges_r)

    # Create the data frame
    data = pd.DataFrame(list(zip(list_dtp, list_r)), columns=['dtp range', 'r'])
    data = data[(data['dtp range'] < dtprng_max) & (data['dtp range'] > dtprng_min) & (data['r'] < r_max) & (data['r'] > r_min)]

    data_dtprng = data['dtp range']
    data_r = data['r']

    # Bin the data
    numbin_dtprng = len(edges_dtprng)-1
    numbin_r = len(edges_r)-1

    binind_dtprng = pd.cut(data_dtprng, edges_dtprng, labels=range(numbin_dtprng))
    binind_r = pd.cut(data_r, edges_r, labels=range(numbin_r))

    data_cat = pd.DataFrame({'dtp range bin': binind_dtprng, 'r bin': binind_r})
    data_cat = data_cat.apply(tuple, 1)
    data = data.assign(bin=data_cat)

    grouped = data.groupby(by='bin')
    bincnt_ind = grouped.size()

    array_bincnt = np.zeros((numbin_r, numbin_dtprng))
    for ind_r in range(numbin_r):
        for ind_dtprng in range(numbin_dtprng):
            ind_tup = (ind_dtprng, ind_r)
            count = bincnt_ind[bincnt_ind.index == ind_tup].values
            if np.size(count) > 0:
                array_bincnt[ind_r, ind_dtprng] = count[0]

    array_binfreq = array_bincnt/np.sum(array_bincnt)*100

    # Return the results
    ddtprng = edges_dtprng[1]-edges_dtprng[0]
    grids_dtprng = edges_dtprng-ddtprng/2
    grids_dtprng = grids_dtprng[1:]

    dr = edges_r[1]-edges_r[0]
    grids_r = edges_r-dr/2
    grids_r = grids_r[1:]

    return array_binfreq, grids_dtprng, grids_r

## Estimate the Vp/Vs ratio from the demined differential travel times
def LinearModel(r, x):
    return r*x

def EstimateVpVs(list_timediff_p, list_timediff_s, list_evstinfo, numstd, numbtstrp):
    import numpy as np
    from scipy.odr import Data, Model, ODR
    from random import choices

    # Initial slope
    beta0 = 1.73

    print('Running the SciPy ODR optimizer for the first round...')
    numtdiff = len(list_timediff_p)
    print('In total '+str(numtdiff)+' data points.')

    array_timediff_p = np.array(list_timediff_p)
    array_timediff_s = np.array(list_timediff_s)

    model_odr  = Model(LinearModel)
    data_odr = Data(array_timediff_p, array_timediff_s)
    odr = ODR(data_odr, model_odr, beta0=[beta0])
    output_odr = odr.run()
    r_opt = output_odr.beta[0]
    res_var = output_odr.res_var
    std_r = output_odr.sd_beta[0]
    rms = np.sqrt(output_odr.sum_square/numtdiff)
    print('The optimum slope is '+str(r_opt)+'.')
    print('The slope standard deviation is '+str(std_r)+'.')
    print('The RMS is '+str(rms)+'.')

    res_x = output_odr.delta
    res_y = output_odr.eps
    ressqr = np.square(res_x)+np.square(res_y)
    array_timediff_p_filt = array_timediff_p[ressqr < np.square(numstd)*res_var]
    array_timediff_s_filt = array_timediff_s[ressqr < np.square(numstd)*res_var]
    list_evstinfo_filt = [list_evstinfo[i] for i in np.argwhere(ressqr < np.square(numstd)*res_var)[:, 0]]

    numtdiff_filt = len(array_timediff_p_filt)
    perc = numtdiff_filt/numtdiff*100
    print(str(perc)+'% data points are left after the outliers are removed.')

    print('Running the SciPy ODR optimizer for the second round...')

    model_odr  = Model(LinearModel)
    data_odr = Data(array_timediff_p_filt, array_timediff_s_filt)
    odr = ODR(data_odr, model_odr, beta0=[beta0])
    output_odr = odr.run()
    r_opt_filt = output_odr.beta[0]
    res_var = output_odr.res_var
    std_r_filt = output_odr.sd_beta[0]
    rms_filt = np.sqrt(output_odr.sum_square/numtdiff)
    print('The optimum slope is '+str(r_opt_filt)+'.')
    print('The slope standard deviation is '+str(std_r_filt)+'.')
    print('The RMS is '+str(rms)+'.')

    ## Perform bootstrap resampling
    print(str(numbtstrp)+' bootstrap resampling will be performed.')
    list_r_filt_resamp = []
    for i in range(numbtstrp):
        list_ind = choices(range(numtdiff_filt), k=numtdiff_filt)
        array_timediff_p_resamp = array_timediff_p_filt[list_ind]
        array_timediff_s_resamp = array_timediff_s_filt[list_ind]
        data_odr = Data(array_timediff_p_resamp, array_timediff_s_resamp)
        odr = ODR(data_odr, model_odr, beta0=[beta0])
        output_odr = odr.run()
        r_opt_resamp = output_odr.beta[0]
        list_r_filt_resamp.append(r_opt_resamp)

        if i%50 == 0:
            print(str(i)+' resampling finished.')

    std_r_filt_resamp = np.std(list_r_filt_resamp)
    print('The slope standard deviation from bootstrap resampling is '+str(std_r_filt_resamp)+'.')

    ## Perform jackknife estimation
    # Added on 2022-12-29, Tianze Liu
    print('Performing jackknife estimation...')
    list_r_jk = []
    for i in range(numtdiff_filt):
        array_timediff_p_jk = np.concatenate([array_timediff_p_filt[:i], array_timediff_p_filt[i+1:]])
        array_timediff_s_jk = np.concatenate([array_timediff_s_filt[:i], array_timediff_s_filt[i+1:]])
        data_odr = Data(array_timediff_p_jk, array_timediff_s_jk)
        odr = ODR(data_odr, model_odr, beta0=[beta0])
        output_odr = odr.run()
        r_opt_jk = output_odr.beta[0]
        list_r_jk.append(r_opt_jk)

    array_r_jk = np.array(list_r_jk)
    r_mean = np.mean(array_r_jk)
    std_r_filt_jk = np.sqrt(numtdiff_filt/(numtdiff_filt-1)*np.sum(np.square(array_r_jk-r_mean)))
    print('The jackknife standard deviation is '+str(std_r_filt_jk)+'.')

    list_timediff_p_filt = list(array_timediff_p_filt)
    list_timediff_s_filt = list(array_timediff_s_filt)

    return list_timediff_p_filt, list_timediff_s_filt, list_evstinfo_filt, r_opt_filt, std_r_filt_resamp, std_r_filt_jk, rms_filt, r_opt, std_r, rms

## wig_prf_avg the Vp/Vs ratio directly from cross-correlation results using structred total least square
def EstimateVpVs_StructuredTLS(xcorrmax_min, numxcorr_min, dict_ev, lines_xcorr):
    import numpy as np
    from matplotlib import pyplot as plt

    ## Process the cross-correlation results for each event pair
    numlin = len(lines_xcorr)
    ind_lin = 0
    # list_tdiffinfo_out = []
    # list_timediff_p = []
    # list_timediff_s = []
    # list_evid_out = []
    # list_evx_out = []
    # list_evy_out = []
    # list_evz_out = []

    # List storing the design matrix and the dependent variable
    list_data_out = []
    list_numxcorr_out = []

    numpair_proc = 0
    numpair_fin_out = 0
    print('Reading the input xcorr data...')
    while ind_lin < numlin:
        # Read the two event IDs and the number of cross-correlation values for this two events
        line = lines_xcorr[ind_lin]
        fields = line.split()
        evid1 = fields[0]
        evid2 = fields[1]
        numxcorr = int(fields[2])

        if numxcorr < numxcorr_min:
            ind_lin = ind_lin+numxcorr+1
        else:
            list_data_pair = []

            for i in range(numxcorr):
                ind_lin = ind_lin+1
                line = lines_xcorr[ind_lin]
                fields = line.split()
                stnm = fields[0]
                timeshift_p = float(fields[2])
                xcorrmax_p = float(fields[3])
                timeshift_s = float(fields[4])
                xcorrmax_s = float(fields[5])

                if xcorrmax_p >= xcorrmax_min and xcorrmax_s >= xcorrmax_min:
                    list_data_pair.append((stnm, timeshift_p, timeshift_s))

            numxcorr = len(list_data_pair)
            if numxcorr >= numxcorr_min:
                list_numxcorr_out.append(numxcorr)
                list_data_out.extend(list_data_pair)

            ind_lin = ind_lin+1

    ## Assemble the data matrices
    print('Assembling the data matrices...')
    numdat = len(list_data_out)
    numpair = len(list_numxcorr_out)
    numpara = numpair+1
    print('The design matrix is '+str(numdat)+'x'+str(numpara))

    array_dmat = np.zeros((numdat, numpara))
    array_y = np.zeros((numdat, 1))

    ind_dat = 0
    for i in range(numpair):
        numxcorr = list_numxcorr_out[i]

        for j in range(numxcorr):
            data = list_data_out[ind_dat]
            timeshift_p = data[1]
            timeshift_s = data[2]

            array_dmat[ind_dat, 0] = timeshift_p
            array_dmat[ind_dat, i+1] = 1

            array_y[ind_dat] = timeshift_s

            ind_dat = ind_dat+1

    # array_x = array_dmat[:, 0]
    # plt.scatter(array_x, array_y)
    # plt.show()

    ## Perform TLS wig_prf_avg using SVD
    print('Perform TLS wig_prf_avg using SVD...')
    array_dat = np.concatenate((array_dmat, array_y), axis=1)
    array_u, array_s, array_vh = np.linalg.svd(array_dat)
    array_v = np.transpose(array_vh)
    r_opt = -array_v[0, -1]/array_v[-1, -1]
    print(r_opt)

## Compute the inter-event distance projected to the event-station direction for one event pair
def GetProjectedEventDist(evid1, evid2, stnm, dict_evinfo, dict_stinfo):
    from numpy import array, abs, inner, mean
    from numpy.linalg import norm

    x_ev1 = dict_evinfo[evid1][0]
    y_ev1 = dict_evinfo[evid1][1]
    dep_ev1 = dict_evinfo[evid1][2]

    x_ev2 = dict_evinfo[evid2][0]
    y_ev2 = dict_evinfo[evid2][1]
    dep_ev2 = dict_evinfo[evid2][2]

    x_st = dict_stinfo[stnm][0]
    y_st = dict_stinfo[stnm][1]

    vect_evsep = array([x_ev2, y_ev2, dep_ev2])-array([x_ev1, y_ev1, dep_ev1])
    vect_evctr = (array([x_ev2, y_ev2, dep_ev2])+array([x_ev1, y_ev1, dep_ev1]))/2

    vect_evst = array([x_st, y_st, 0])-vect_evctr
    vect_evst = vect_evst/norm(vect_evst, ord=2)

    dist_proj = abs(inner(vect_evsep, vect_evst))

    return dist_proj

## Compute the avereage event separations projected to the event-station direction
def GetAverageProjectedEventDist(list_evstinfo, dict_evinfo, dict_stinfo):
    from numpy import array, abs, inner, mean
    from numpy.linalg import norm

    list_dist_proj = []
    for evstinfo in list_evstinfo:
        evid1 = evstinfo[0]
        evid2 = evstinfo[1]
        stnm = evstinfo[2]

        dist_proj = GetProjectedEventDist(evid1, evid2, stnm, dict_evinfo, dict_stinfo)

        list_dist_proj.append(dist_proj)

    dist_proj_avg = mean(array(list_dist_proj))

    return dist_proj_avg

## Read the 3D velocity model for Pykonal
# The vertical axis will be reversed! (from dep to z)!
def Read3DvelModel(path):
    from netCDF4 import Dataset
    import numpy as np

    rootgrp = Dataset(path)

    vect_x = rootgrp.variables['x'][:]
    vect_y = rootgrp.variables['y'][:]
    vect_dep = rootgrp.variables['depth'][:]
    array_vp = rootgrp.variables['vp'][:]
    array_vs = rootgrp.variables['vs'][:]

    vect_z = np.amax(vect_dep)-np.flip(vect_dep)
    array_vp = np.flip(array_vp, axis=2)
    array_vs = np.flip(array_vs, axis=2)

    rootgrp.close()

    return vect_x, vect_y, vect_z, array_vp, array_vs

## Read the 3D travel-time field
def Read3DtravelTime(path):
    from netCDF4 import Dataset
    import numpy as np

    rootgrp = Dataset(path)

    vect_x = rootgrp.variables['x'][:]
    vect_y = rootgrp.variables['y'][:]
    vect_dep = rootgrp.variables['depth'][:]
    array_tp = rootgrp.variables['tp'][:]
    array_ts = rootgrp.variables['ts'][:]

    rootgrp.close()

    return vect_x, vect_y, vect_dep, array_tp, array_ts

## Use an interative method to solve the aggregate bulk and shear moduli of a two-phase system with spheroidal inclusions according to Berryman (1980)
def GetNu(kappa, mu):
    nu = (3*kappa-2*mu)/(2*(3*kappa+mu))
    return nu

def GetA(mu1, mu2):
    a = mu2/mu1-1
    return a

def GetB(kappa1, mu1, kappa2, mu2):
    b = 1/3*(kappa2/kappa1-mu2/mu1)
    return b

def GetR(kappa, mu):
    nu = GetNu(kappa, mu)
    r = (1-2*nu)/(2*(1-nu))
    return r

def GetF1(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf1 = 1+a*(3/2*(f+theta)-r*(3/2*f+5/2*theta-4/3))
    return bf1

def GetF2(f, theta, kappa1, mu1, kappa2, mu2):
    from numpy import square

    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf2 = 1+a*(1+3/2*(f+theta)-r/2*(3*f+5*theta))+b*(3-4*r)+a/2*(a+3*b)*(3-4*r)*(f+theta-r*(f-theta+2*square(theta)))
    return bf2

def GetF3(f, theta, kappa1, mu1, mu2):
    a = GetA(mu1, mu2)
    r = GetR(kappa1, mu1)

    bf3 = 1+a*(1-(f+3/2*theta)+r*(f+theta))
    return bf3

def GetF4(f, theta, kappa1, mu1, mu2):
    a = GetA(mu1, mu2)
    r = GetR(kappa1, mu1)

    bf4 = 1+a/4*(f+3*theta-r*(f-theta))
    return bf4

def GetF5(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf5 = a*(-f+r*(f+theta-4/3))+b*theta*(3-4*r)
    return bf5

def GetF6(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf6 = 1+a*(1+f-r*(f+theta))+b*(1-theta)*(3-4*r)
    return bf6

def GetF7(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf7 = 2+a/4*(3*f+9*theta-r*(3*f+5*theta))+b*theta*(3-4*r)
    return bf7

def GetF8(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf8 = a*(1-2*r+f/2*(r-1)+(theta/2)*(5*r-3))+b*(1-theta)*(3-4*r)
    return bf8

def GetF9(f, theta, kappa1, mu1, kappa2, mu2):
    a = GetA(mu1, mu2)
    b = GetB(kappa1, mu1, kappa2, mu2)
    r = GetR(kappa1, mu1)

    bf9 = a*((r-1)*f-r*theta)+b*theta*(3-4*r)
    return bf9

def GetT1(f, theta, kappa1, mu1, kappa2, mu2):
    bf1 = GetF1(f, theta, kappa1, mu1, kappa2, mu2)
    bf2 = GetF2(f, theta, kappa1, mu1, kappa2, mu2)

    bt1 = 3*bf1/bf2
    return bt1

def GetT2(f, theta, kappa1, mu1, kappa2, mu2):
    bf2 = GetF2(f, theta, kappa1, mu1, kappa2, mu2)
    bf3 = GetF3(f, theta, kappa1, mu1, mu2)
    bf4 = GetF4(f, theta, kappa1, mu1, mu2)
    bf5 = GetF5(f, theta, kappa1, mu1, kappa2, mu2)
    bf6 = GetF6(f, theta, kappa1, mu1, kappa2, mu2)
    bf7 = GetF7(f, theta, kappa1, mu1, kappa2, mu2)
    bf8 = GetF8(f, theta, kappa1, mu1, kappa2, mu2)
    bf9 = GetF9(f, theta, kappa1, mu1, kappa2, mu2)

    bt2 = 2/bf3+1/bf4+(bf4*bf5+bf6*bf7-bf8*bf9)/(bf2*bf4)
    return bt2

def GetP(f, theta, kappa1, mu1, kappa2, mu2):
    bt1 = GetT1(f, theta, kappa1, mu1, kappa2, mu2)
    bp = bt1/3
    return bp

def GetQ(f, theta, kappa1, mu1, kappa2, mu2):
    bt2 = GetT2(f, theta, kappa1, mu1, kappa2, mu2)
    bq = bt2/5
    return bq

def GetAggregateModuli_Berryman_spheroid(kappa1, mu1, kappa2, mu2, kappa0, mu0, phi, alpha):
    from numpy import square, power, arccos, sqrt

    diffnm_min = 0.001
    numitr_max = 50

    # Define the shape variables
    theta = alpha/power(1-square(alpha), 3/2)*(arccos(alpha)-alpha*sqrt(1-square(alpha)))
    f = square(alpha)/(1-square(alpha))*(3*theta-2)

    # Initialize the iteration
    mu0 = mu1
    kappa0 = kappa1

    kdiff_nm = 1
    mdiff_nm = 1

    numitr = 0
    while numitr < numitr_max:

        print('Interation '+str(numitr)+'..')
        bp1 = GetP(f, theta, kappa0, mu0, kappa1, mu1)
        bq1 = GetQ(f, theta, kappa0, mu0, kappa1, mu1)

        bp2 = GetP(f, theta, kappa0, mu0, kappa2, mu2)
        bq2 = GetQ(f, theta, kappa0, mu0, kappa2, mu2)

        kappa0_new = ((1-phi)*bp1*kappa1+phi*bp2*kappa2)/((1-phi)*bp1+phi*bp2)
        mu0_new = ((1-phi)*bq1*mu1+phi*bq2*mu2)/((1-phi)*bq1+phi*bq2)

        kdiff_nm = abs((kappa0_new-kappa0)/kappa0)
        mdiff_nm = abs((mu0_new-mu0)/mu0)

        print('The old kappa is '+str(kappa0)+'.')
        print('The updated kappa is '+str(kappa0_new)+'.')
        print('The update is '+str(kdiff_nm*100)+'%')

        print('The old mu is '+str(mu0)+'.')
        print('The updated mu is '+str(mu0_new)+'.')
        print('The update is '+str(mdiff_nm*100)+'%')

        kappa0 = kappa0_new
        mu0 = mu0_new

        if kdiff_nm < diffnm_min and mdiff_nm < diffnm_min:
            print('The update is < '+str(diffnm_min*100)+'%. The terimination condition is reached.')
            break

        numitr = numitr+1

    if numitr >= numitr_max:
        print('The maximum number of iterations of '+str(numitr_max)+' is reached without achieving the termination condition. Exit.')
        raise

    return kappa0, mu0

## Compute the mineral properties at an elevated temperature using the relations from Hacker et al. (2003)
def GetMineralPropertiesTempratureCorrect(kappa0_tmp, mu0, rho0, alpha0, delta, bgamma, sgamma, temp0, temp):
    from numpy import square, sqrt, exp

    a0 = alpha0/(1-10/sqrt(temp0))
    phi = a0*((temp-temp0)-20*(sqrt(temp)-sqrt(temp0)))
    print(phi)

    kappa_tmp = kappa0_tmp*exp(-delta*phi)
    mu = mu0*exp(-bgamma*phi)
    rho = rho0*exp(-phi)
    alpha = a0*(1-10*sqrt(temp))

    kappa_ent = kappa_tmp*(1+temp*sgamma*alpha)

    return kappa_ent, mu, rho
