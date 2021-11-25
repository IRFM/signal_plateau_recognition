# -*- coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Also if needed: retab
'''
   Statistics computations plateaus
'''
# Standard python modules
from __future__ import (unicode_literals, absolute_import,  \
                        print_function, division)
#import logging
import numpy as np
#import os
#import re
#import scipy.io
import scipy.interpolate as sc_interp
import scipy.signal as sc_sig
from sklearn.neighbors import KernelDensity
#from statsmodels import robust
#import traceback
#import sys
import warnings


__all__ = ['StatsPlateau', 'ErrorStatsPlateau']


class StatsPlateau:
    '''
    Return object with identified non-zeros plateaus from temporal data.
    By default returns 'all', 'high' and 'long' plateaus.

    Parameters
    ----------
    timeRef : one-dimensional float array
        time data
    dataRef : one-dimensional float array, same length as timeRef
        data
    errorPlateau : float, optional (default=0.05)
        One side error or tolerance for plateau computation RELATIVE to
        plateau value (ex: plateau_value +/- plateau_value*errorPlateau)
    minPlateau : float, optional (default=0.3)
        Minimum length of plateau (in timeRef units)
    distPlateau : float, optional (default=0.05)
        Minimum distance between plateaus (in timeRef units).
        If distance between points of plateau is larger than distPlateau
        plateau is split in two distinct plateaus
    plateau : string, optional (default='all')
        Requested plateau for computations (plateau where to apply
        methods, see methods bellow). One of

            ``all``
            Computations on all the plateaus

            ``high``
            Computations on higher plateau

            ``long``
            Computations on longer plateau

    lowLim : float, optional (default: minimum in dataRef)
        Low limit for plateaus
    highLim : float, optional (default: maximum in dataRef)
        High limit for plateaus

    Methods
    -------
    applyFct(timeData, data, [function(s)]) :
        Apply list of function(s) to input data in the same plateau requested when
        initialising the class

    Attributes
    ----------
    dataPlateau : list of float arrays (length: number of found plateaus)
        Data array of each plateau
    dataRef : float array
        Reference data, input dataRef
    dataReqPlateau : float array
        Reference data only in the requested plateau (if 'high' or 'long')
    dtPlateau : float array
        Time interval of each plateaus
    dtReqPlateau : float
        Time interval of requested plateau (if 'high' or 'long')
    highPlateau : float
        Value of higher plateau
    longPlateau : float
        Value of longer plateau
    maskPlateau : list of boolean arrays
        Masks corresponding to found plateaus, to be applied to dataRef or to timeRef
    plateauVal : float array
        Median value in each plateau
    tEndPlateau : float array
        Final time of each plateau
    timePlateau : list of float arrays (length: number of found plateaus)
        Time array of each plateau
    timeRef : float array
        Reference time, input timeRef
    timeReqPlateau : float array
        Reference data only in the requested plateau (if 'high' or 'long')
    tIniPlateau : float array
        Initial time of each plateau
    '''
    def __init__(self, timeRef, dataRef, errorPlateau=0.05, \
                 minPlateau=0.3, distPlateau=0.05, plateau='all', \
                 lowLim=None, highLim=None):

        # RETURN if empty minPlateau or distPlateau small compared to time step
        if (np.max(timeRef[1:] - timeRef[:-1]) >= minPlateau):
            raise ErrorStatsPlateau('minPlateau = '+str(minPlateau), \
              'Minimum length plateau smaller than maximum time step ' \
              +'{0:.4f}'.format(np.max(timeRef[1:] - timeRef[:-1]))+' in timeRef')
        if (np.max(timeRef[1:] - timeRef[:-1]) >= distPlateau):
            raise ErrorStatsPlateau('distPlateau = '+str(distPlateau), \
              'Minimum distance plateau smaller than maximum time step ' \
              +'{0:.4f}'.format(np.max(timeRef[1:] - timeRef[:-1]))+' in timeRef')

        # Init number sample points for kernel density evaluation
        self.__numSamplePts = 1000

        # Initialize lists for storing different plateau data
        self.timeRef     = []
        self.dataRef     = []
        self.plateauVal  = []
        self.tIniPlateau = []
        self.tEndPlateau = []
        self.maskPlateau = []
        self.timePlateau = []
        self.highPlateau = []
        self.longPlateau = []
        self._maskRefReq = []

        # Make 1D arrays
        timeRef = np.atleast_1d(np.squeeze(timeRef))
        dataRef = np.atleast_1d(np.squeeze(dataRef))

        # Compute plateau for init signal
        self._computePlateaus(timeRef, dataRef, errorPlateau, \
                              minPlateau, distPlateau, plateau, \
                              lowLim, highLim)

    def addSignal(self, timeRef, dataRef, errorPlateau=0.05, \
                  minPlateau=0.3, distPlateau=0.05, plateau='all', \
                  lowLim=None, highLim=None):

        # RETURN if empty minPlateau or distPlateau small compared to time step
        if (np.max(timeRef[1:] - timeRef[:-1]) >= minPlateau):
            raise ErrorStatsPlateau('minPlateau = '+str(minPlateau), \
              'Minimum length plateau smaller than maximum time step ' \
              +'{0:.4f}'.format(np.max(timeRef[1:] - timeRef[:-1]))+' in timeRef')
        if (np.max(timeRef[1:] - timeRef[:-1]) >= distPlateau):
            raise ErrorStatsPlateau('distPlateau = '+str(distPlateau), \
              'Minimum distance plateau smaller than maximum time step ' \
              +'{0:.4f}'.format(np.max(timeRef[1:] - timeRef[:-1]))+' in timeRef')

        if (not isinstance(self.plateauVal, list)):
            tmpVal           = self.plateauVal
            self.plateauVal  = []
            self.plateauVal.append(tmpVal)
            tmpVal           = self.tIniPlateau
            self.tIniPlateau = []
            self.tIniPlateau.append(tmpVal)
            tmpVal           = self.tEndPlateau
            self.tEndPlateau = []
            self.tEndPlateau.append(tmpVal)
            tmpVal           = self.maskPlateau
            self.maskPlateau = []
            self.maskPlateau.append(tmpVal)
            tmpVal           = self._maskRefReq
            self._maskRefReq = []
            self._maskRefReq.append(tmpVal)
            tmpVal           = self.timePlateau
            self.timePlateau = []
            self.timePlateau.append(tmpVal)

        # Make 1D arrays
        timeRef = np.atleast_1d(np.squeeze(timeRef))
        dataRef = np.atleast_1d(np.squeeze(dataRef))

        # Interpolate in first time base and keep timeRef as its ref. time
        #dataRef = np.interp(self.timeRef[0], timeRef, dataRef)
        mask_nans = (~np.isnan(timeRef)) & (~np.isnan(dataRef))
        f_interp = sc_interp.interp1d(timeRef[mask_nans], dataRef[mask_nans], \
                                      bounds_error=False)
        dataRef = f_interp(self.timeRef[0])
        timeRef = self.timeRef[0]

        # Compute plateau for other signals
        self._computePlateaus(timeRef, dataRef, errorPlateau, \
                              minPlateau, distPlateau, plateau, \
                              lowLim, highLim)

    def _computePlateaus(self, timeRef, dataRef, errorPlateau=0.05, \
                         minPlateau=0.3, distPlateau=0.05, plateau='all', \
                         lowLim=None, highLim=None):

        # Make sure input data is 1D
        self.timeRef.append(timeRef)
        self.dataRef.append(dataRef)
        self.plateauVal.append(None)
        self.tIniPlateau.append(None)
        self.tEndPlateau.append(None)
        self.maskPlateau.append(None)
        self.timePlateau.append(None)

        # Equally spaced time and data
        self._timeRefIntp = np.linspace(self.timeRef[-1][0], \
                                        self.timeRef[-1][-1], \
                                        self.timeRef[-1].size)
        #self._dataRefIntp = np.interp(self._timeRefIntp, self.timeRef[-1], \
        #                              self.dataRef[-1])
        mask_nans = (~np.isnan(self.timeRef[-1])) & (~np.isnan(self.dataRef[-1]))
        f_interp = sc_interp.interp1d(self.timeRef[-1][mask_nans], \
                                      self.dataRef[-1][mask_nans], \
                                      bounds_error=False)
        self._dataRefIntp = f_interp(self._timeRefIntp)

        self.plateauVal[-1], self.tIniPlateau[-1], self.tEndPlateau[-1], \
        self._pdfPeak, self.maskPlateau[-1], self.timePlateau[-1] \
                     = self._find_plateaus( \
                                self.timeRef[-1], self.dataRef[-1], \
                                self._timeRefIntp, self._dataRefIntp, \
                                errorPlateau, minPlateau, distPlateau)

        if (self.plateauVal[-1].size > 1):
            diffPlateau = self.plateauVal[-1][1:] - self.plateauVal[-1][:-1]
            gapPlateau  = self.tIniPlateau[-1][1:] - self.tEndPlateau[-1][:-1]

            self._errorPlateau2 = errorPlateau

            #while (np.any(np.abs(diffPlateau) \
            #           <= np.abs(errorPlateau*self.plateauVal[-1][:-1])) \
            #       and np.any(gapPlateau <= distPlateau) \
            #       or np.any(gapPlateau <= 0)):
            while (np.any((np.abs(diffPlateau) \
                <= np.abs(errorPlateau*self.plateauVal[-1][:-1])) \
                & (gapPlateau <= distPlateau))):

                self._errorPlateau2 *= 1.05
                if (self._errorPlateau2 > 2*errorPlateau):
                    raise ErrorStatsPlateau('errorPlateau2 = ' \
                                          + str(self._errorPlateau2), \
                                            'Max errorPlateau2 reached')

                self.plateauVal[-1], self.tIniPlateau[-1], self.tEndPlateau[-1], \
                self._pdfPeak, self.maskPlateau[-1], self.timePlateau[-1] \
                             = self._find_plateaus( \
                                        self.timeRef[-1], self.dataRef[-1], \
                                        self._timeRefIntp, self._dataRefIntp, \
                                        self._errorPlateau2, minPlateau, distPlateau)

                if (self.plateauVal[-1].size > 1):
                    diffPlateau = self.plateauVal[-1][1:] - self.plateauVal[-1][:-1]
                    gapPlateau  = self.tIniPlateau[-1][1:] - self.tEndPlateau[-1][:-1]
                else:
                    break

        if (self.plateauVal[-1].size > 0):
            if (lowLim is not None and highLim is None):
                self._maskLim = self.plateauVal[-1] > lowLim
            elif (lowLim is None and highLim is not None):
                self._maskLim = self.plateauVal[-1] < highLim
            elif (lowLim is not None and highLim is not None):
                self._maskLim = (self.plateauVal[-1] > lowLim) \
                              & (self.plateauVal[-1] < highLim)
            else:
                self._maskLim = np.ones(self.plateauVal[-1].size, dtype=bool)

            self.plateauVal[-1]  = self.plateauVal[-1][self._maskLim]
            self.tIniPlateau[-1] = self.tIniPlateau[-1][self._maskLim]
            self.tEndPlateau[-1] = self.tEndPlateau[-1][self._maskLim]
            self._pdfPeak = self._pdfPeak[self._maskLim]
            self.maskPlateau[-1] = self.maskPlateau[-1][self._maskLim]
            self.timePlateau[-1] = [self.timePlateau[-1][kk] \
                 for kk in range(self._maskLim.size) if self._maskLim[kk]]

            if (self.plateauVal[-1].size > 0):
                self.highPlateau.append(np.max(self.plateauVal[-1]))
                self.longPlateau.append(self.plateauVal[-1][np.argmax(self._pdfPeak)])

                # Mask requested plateau
                if (plateau == 'high'):
                    self._maskRefReq.append([self.maskPlateau[-1][np.argmax(self.plateauVal[-1])]])
                elif (plateau == 'long'):
                    self._maskRefReq.append([self.maskPlateau[-1][np.argmax(self._pdfPeak)]])
                elif (plateau == 'all'):
                    self._maskRefReq.append(self.maskPlateau[-1])
                else:
                    print()
                    print('ERROR: unknown plateau option:', plateau)
                    print()
                    raise SyntaxError

                if (len(self._maskRefReq) > 1):
                    self._maskInterRef = self._maskRefReq[0]
                    for ii in range(len(self._maskRefReq) - 1):
                        self._maskIntersec = []
                        for jj in range(self._maskInterRef.shape[0]):
                            for kk in range(self._maskRefReq[ii+1].shape[0]):
                                self._maskIntersec.append( \
                                                   self._maskInterRef[jj] \
                                                 & self._maskRefReq[ii+1][kk])
                        self._maskInterRef = \
                          [ll for ll in self._maskIntersec \
                           if not np.all(~ll)] # negation ll

                    self._maskRefReq = np.asarray(self._maskInterRef)

                    # Filter plateaus smaller that minPlateau
                    self.timePlateau = [None]*len(self._maskRefReq)
                    for ii in range(len(self._maskRefReq)):
                        self.timePlateau[ii] = self.timeRef[0][self._maskRefReq[ii]]

                    self._maskRefReq = \
                      np.asarray([self._maskRefReq[ll] \
                       for ll in range(len(self._maskRefReq)) \
                       if ((self.timePlateau[ll][-1] - self.timePlateau[ll][-0]) \
                        > minPlateau)])

                    # Sort plateaus by time
                    self.timePlateau = [None]*len(self._maskRefReq)
                    self.tIniPlateau = np.full(len(self._maskRefReq), np.nan)
                    self.tEndPlateau = np.full(len(self._maskRefReq), np.nan)
                    for ii in range(len(self._maskRefReq)):
                        self.timePlateau[ii] = self.timeRef[0][self._maskRefReq[ii]]
                        self.tIniPlateau[ii] = self.timePlateau[ii][0]
                        self.tEndPlateau[ii] = self.timePlateau[ii][-1]

                    sort_indexes = np.argsort(self.tIniPlateau)

                    self.tIniPlateau = self.tIniPlateau[sort_indexes]
                    self.tEndPlateau = self.tEndPlateau[sort_indexes]
                    self.maskPlateau = self._maskRefReq[sort_indexes]
                    self._maskRefReq = self._maskRefReq[sort_indexes]
                    self.timePlateau = [self.timePlateau[kk] for kk in sort_indexes]

                    self.dataPlateau = \
                      [[None]*len(self._maskRefReq) for kk in range(len(self.dataRef))]
                    self.dtPlateau = np.full(len(self._maskRefReq), np.nan)
                    for ii in range(len(self.dataRef)):
                        for jj in range(len(self._maskRefReq)):
                            self.dataPlateau[ii][jj] = \
                                 self.dataRef[ii][self.maskPlateau[jj]]
                    for ii in range(len(self._maskRefReq)):
                        self.dtPlateau[ii] = self.timePlateau[ii][-1] \
                                           - self.timePlateau[ii][0]
                else:
                    self.plateauVal  = self.plateauVal[-1]
                    self.tIniPlateau = self.tIniPlateau[-1]
                    self.tEndPlateau = self.tEndPlateau[-1]
                    self.maskPlateau = self.maskPlateau[-1]
                    self._maskRefReq = self._maskRefReq[-1]
                    self.timePlateau = self.timePlateau[-1]

                    self.dataPlateau = [None]*self.plateauVal.size
                    self.dtPlateau   = np.full(self.plateauVal.size, np.nan)
                    for ii in range(self.plateauVal.size):
                        self.dataPlateau[ii] = \
                                       self.dataRef[-1][self.maskPlateau[ii]]
                        self.dtPlateau[ii]   = self.timePlateau[ii][-1] \
                                             - self.timePlateau[ii][0]

                if (plateau == 'high' or plateau == 'long'):
                    self.dataReqPlateau = self.dataRef[0][self._maskRefReq[0]]
                    self.timeReqPlateau = self.timeRef[0][self._maskRefReq[0]]
                    self.dtReqPlateau   = self.timeReqPlateau[-1] \
                                        - self.timeReqPlateau[0]

            else: # Else self.plateauVal[-1] > 0
                raise ErrorStatsPlateau('self.plateauVal[-1].size = ' \
                                        +str(self.plateauVal[-1].size), \
                                        'No plateau found')
        else: # Else self.plateauVal[-1] > 0
            raise ErrorStatsPlateau('self.plateauVal[-1].size = ' \
                                    +str(self.plateauVal[-1].size), \
                                    'No plateau found')

    def applyFct(self, timeData, data, lfunction, sigma_outliers=3):
        '''Apply function or list of functions to data at plateau'''
        # Make sure input data is 1D
        timeData = np.atleast_1d(np.squeeze(timeData))
        data     = np.atleast_1d(np.squeeze(data))

        #self.data = np.interp(self.timeRef[0], timeData, data)
        mask_nans = (~np.isnan(timeData)) & (~np.isnan(data))
        f_interp = sc_interp.interp1d(timeData[mask_nans], data[mask_nans], \
                                      bounds_error=False)
        self.data = f_interp(self.timeRef[0])

        self.mask_outliers = [None]*len(self._maskRefReq)

        if (isinstance(lfunction, list)):
            self.result = \
              [[None]*len(self._maskRefReq) for il in lfunction]
            for ii in range(len(lfunction)):
                for jj in range(len(self._maskRefReq)):
                    self._dataFctPlateau = self.data[self._maskRefReq[jj]]
                    self.mask_outliers[jj] = reject_outliers( \
                                             self._dataFctPlateau, \
                                             m=sigma_outliers, return_mask=True)
                    if (~np.all(self.mask_outliers[jj])):
                        warnings.warn('Removing outliers in plateau')
                    self.result[ii][jj] = eval(lfunction[ii] \
                         + '(self._dataFctPlateau[self.mask_outliers[jj]])')
            self.result = tuple(self.result)
        else:
            self.result = [None]*len(self._maskRefReq)
            for ii in range(len(self._maskRefReq)):
                self._dataFctPlateau = self.data[self._maskRefReq[ii]]
                self.mask_outliers[jj] = reject_outliers( \
                                          self._dataFctPlateau, \
                                          m=sigma_outliers, return_mask=True)
                if (~np.all(self.mask_outliers[jj])):
                    warnings.warn('Removing outliers in plateau')
                self.result[ii] = eval(lfunction \
                                + '(self._dataFctPlateau[self.mask_outliers[jj]])')
            self.result = tuple(self.result)
        return self.result

    def _find_plateaus(self, timeRef, dataRef, timeRefIntp, dataRefIntp, \
                       errorPlateau, minPlateau, distPlateau):
        '''Find plateaus in data'''

        # Use kernel density estimation (KDE) to find most probable values in data
        band_in = 0.5*errorPlateau*np.abs(dataRefIntp.max())

        kde = KernelDensity(kernel='linear', bandwidth=band_in).fit( \
                                             dataRefIntp[:, np.newaxis])

        self._samplePts = np.linspace(dataRefIntp.min(), \
                                      dataRefIntp.max(), self.__numSamplePts)
        self._pdf = np.exp(kde.score_samples(self._samplePts[:, np.newaxis]))

        minPlateauNorm = minPlateau / (timeRefIntp[-1] - timeRefIntp[0])

        #self._pdfFilt       = self._pdf[band_in*self._pdf > minPlateauNorm]
        #self._samplePtsFilt = self._samplePts[band_in*self._pdf > minPlateauNorm]

        try:
            ind_peaks, _ = sc_sig.find_peaks(self._pdf, \
                                  distance=(self.__numSamplePts*errorPlateau))
        except AttributeError as err:
            warnings.warn('Function find_peaks not found, we recommend to update' \
                         + ' scipy to a version >= 1.1.0')
            # Hand made find_peaks function taking into account distance option
            ind_peaks = sc_sig.argrelmax(self._pdf)[0]
            distance_peaks = self.__numSamplePts*errorPlateau

            while np.any(np.diff(ind_peaks) < distance_peaks):
                ind_sort_peaks = np.argsort(self._pdf[ind_peaks])
                mask_del = np.ones(ind_peaks.size, dtype=bool)
                for ii in range(ind_sort_peaks.size):
                    if (ind_sort_peaks[ii]+1 < ind_peaks.size):
                        if (np.abs(ind_peaks[ind_sort_peaks[ii]+1] \
                                 - ind_peaks[ind_sort_peaks[ii]]) \
                           < distance_peaks):
                            mask_del[ind_sort_peaks[ii]] = False
                    if (ind_sort_peaks[ii]-1 >= 0):
                        if (np.abs(ind_peaks[ind_sort_peaks[ii]-1] \
                                 - ind_peaks[ind_sort_peaks[ii]]) \
                           < distance_peaks):
                            mask_del[ind_sort_peaks[ii]] = False
                    if (~np.all(mask_del)):
                        break
                ind_peaks = ind_peaks[mask_del]

        # Filter taking into account minimum plateau length
        ind_peaks = ind_peaks[band_in*self._pdf[ind_peaks] > minPlateauNorm]

        self._ptsPeak = self._samplePts[ind_peaks]
        pdfPeak       = self._pdf[ind_peaks]

        self._medianPeak = np.full(self._ptsPeak.size, np.nan)
        for ii in range(self._ptsPeak.size):
            self._medianPeak[ii] = np.median(dataRefIntp[ \
                (dataRefIntp >= (self._ptsPeak[ii] \
                                     - errorPlateau*np.abs(dataRefIntp.max())))
              & (dataRefIntp <= (self._ptsPeak[ii] \
                                     + errorPlateau*np.abs(dataRefIntp.max())))])

        plateauVal = self._medianPeak[self._medianPeak != 0.]
        pdfPeak    = pdfPeak[self._medianPeak != 0.]

        # RETURN if empty plateauVal because plateau not long enough
        if (plateauVal.size == 0):
            raise ErrorStatsPlateau('minPlateau = '+str(minPlateau), \
                                    'Min length plateau not respected')

        self._mask_tmp       = [None]*plateauVal.size
        self._delta_t_plto   = [None]*plateauVal.size
        self._loc_jumps      = [None]*plateauVal.size
        self._intervals      = [None]*plateauVal.size
        self._diff_intervals = [None]*plateauVal.size
        self._maskPltoTmp    = [None]*plateauVal.size
        self._mask_interv    = [None]*plateauVal.size
        self._filt_interv    = [None]*plateauVal.size
        self._boolPlateau    = np.ones(plateauVal.size, dtype=bool)
        self._repeatPlateau  = np.ones(plateauVal.size, dtype=int)
        # Find each individual plateau for each value in plateauVal
        # two plateaus could share the same plateauVal value
        for ii in range(plateauVal.size):
            if (plateauVal[ii] > 0):
                self._mask_tmp[ii] = \
                    (dataRef < (1.+errorPlateau)*plateauVal[ii]) \
                  & (dataRef > (1.-errorPlateau)*plateauVal[ii])
            elif (plateauVal[ii] < 0):
                self._mask_tmp[ii] = \
                    (dataRef > (1.+errorPlateau)*plateauVal[ii]) \
                  & (dataRef < (1.-errorPlateau)*plateauVal[ii])

            # Find jumps to separate different time spaced plateaus
            self._time_mask = timeRef[self._mask_tmp[ii]]
            if (len(self._time_mask) == 0):
                continue
            self._delta_t_plto[ii] = self._time_mask[1:] \
                                   - self._time_mask[:-1]

            self._loc_jumps[ii] = np.nonzero(self._delta_t_plto[ii] > distPlateau)[0]

            # Compute the intervals of each plateau
            if (self._loc_jumps[ii].size == 0):
                self._intervals[ii] = np.asarray([[0, \
                                     (self._time_mask.size - 1)]])
            else:
                self._intervals[ii] = np.zeros((self._loc_jumps[ii].size + 1, 2), \
                                                dtype=int)
                for jj in range(self._loc_jumps[ii].size + 1):
                    if (jj == 0):
                        self._intervals[ii][jj, 0] = 0
                        self._intervals[ii][jj, 1] = self._loc_jumps[ii][jj]
                    elif (jj == self._loc_jumps[ii].size):
                        self._intervals[ii][jj, 0] = self._loc_jumps[ii][jj-1]+1
                        self._intervals[ii][jj, 1] = (self._time_mask.size - 1)
                    else:
                        self._intervals[ii][jj, 0] = self._loc_jumps[ii][jj-1]+1
                        self._intervals[ii][jj, 1] = self._loc_jumps[ii][jj]

            self._diff_intervals[ii] = self._time_mask[self._intervals[ii][:, 1]] \
                                     - self._time_mask[self._intervals[ii][:, 0]]

            # Filter plateaus that are shorter than self._minPlateauPts
            self._mask_interv[ii] = np.repeat( \
              (self._diff_intervals[ii] >= minPlateau), 2).reshape( \
                                           (self._diff_intervals[ii].size, 2))

            self._filt_interv[ii] = \
                self._intervals[ii][self._mask_interv[ii]].reshape( \
                  (-1, self._intervals[ii].shape[1]))

            # Find mask for reference data using timeRef
            self._maskPltoTmp[ii] = [None]*self._filt_interv[ii].shape[0]
            for jj in range(self._filt_interv[ii].shape[0]):
                if (plateauVal[ii] > 0):
                    self._maskPltoTmp[ii][jj] = \
                      (timeRef \
                     > self._time_mask[int(self._filt_interv[ii][jj, 0])]) \
                    & (timeRef \
                     < self._time_mask[int(self._filt_interv[ii][jj, 1])]) \
                    & (dataRef < (1.+errorPlateau)*plateauVal[ii]) \
                    & (dataRef > (1.-errorPlateau)*plateauVal[ii])
                elif (plateauVal[ii] < 0):
                    self._maskPltoTmp[ii][jj] = \
                      (timeRef \
                     > self._time_mask[int(self._filt_interv[ii][jj, 0])]) \
                    & (timeRef \
                     < self._time_mask[int(self._filt_interv[ii][jj, 1])]) \
                    & (dataRef > (1.+errorPlateau)*plateauVal[ii]) \
                    & (dataRef < (1.-errorPlateau)*plateauVal[ii])
            # END FOR LOOP

        # Empty elements that correspond to short plateaus
        for ii in range(plateauVal.size):
            if (self._maskPltoTmp[ii] is None):
                self._boolPlateau[ii] = False
            elif (len(self._maskPltoTmp[ii]) == 0):
                self._maskPltoTmp[ii] = None
                self._boolPlateau[ii] = False
            elif (len(self._maskPltoTmp[ii]) > 1):
                self._repeatPlateau[ii] = len(self._maskPltoTmp[ii])

        self._maskPltoTmp   = list(filter(None, self._maskPltoTmp))
        plateauVal          = plateauVal[self._boolPlateau]
        pdfPeak             = pdfPeak[self._boolPlateau]
        self._repeatPlateau = self._repeatPlateau[self._boolPlateau]

        # Flatten list of list as one single list
        maskPlateau = np.asarray( \
          [mask for maskVal in self._maskPltoTmp for mask in maskVal])

        # Repeat plateau values if several time separated plateaus are found
        plateauVal = np.repeat(plateauVal, self._repeatPlateau)
        pdfPeak    = np.repeat(pdfPeak, self._repeatPlateau)

        # Sort plateaus by time
        timePlateau = [None]*plateauVal.size
        tIniPlateau = np.full(plateauVal.size, np.nan)
        tEndPlateau = np.full(plateauVal.size, np.nan)
        for ii in range(plateauVal.size):
            timePlateau[ii] = timeRef[maskPlateau[ii]]
            tIniPlateau[ii] = timePlateau[ii][0]
            tEndPlateau[ii] = timePlateau[ii][-1]

        sort_indexes = np.argsort(tIniPlateau)

        tIniPlateau = tIniPlateau[sort_indexes]
        tEndPlateau = tEndPlateau[sort_indexes]
        maskPlateau = maskPlateau[sort_indexes]
        plateauVal  = plateauVal[sort_indexes]
        pdfPeak     = pdfPeak[sort_indexes]
        timePlateau = [timePlateau[kk] for kk in sort_indexes]

        return (plateauVal, tIniPlateau, tEndPlateau, \
                pdfPeak, maskPlateau, timePlateau)


class Error(Exception):
   """Base class for other exceptions"""
   pass

def reject_outliers(data, m=2, return_mask=False):
    if (return_mask):
        return np.abs(data - np.nanmean(data)) <= m * np.nanstd(data)
    else:
        return data[np.abs(data - np.nanmean(data)) <= m * np.nanstd(data)]

class ErrorStatsPlateau(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
