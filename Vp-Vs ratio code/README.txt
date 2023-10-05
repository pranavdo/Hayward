1.  Run RemoveIntercepts_OnFaultAreaShortRange.py to remove the origin-time errors from the raw data
    The script takes an event-information file EventInfo.dat and a cross-correlation-information file XcorrInfo.dat as inputs and writes its outputs to DiffTime.dat

2.  Run EstimateVpVs_OneFaultAreaShortRange.py to estimate Vp/Vs on the preprocessed data.
    The script takes DiffTime.dat as the input and writes its outputs to OptimumVpVs_ODR_filtered.dat (filtered means the data without outliers).

Note: Both scripts rely on functions defined in the module file utilities.py