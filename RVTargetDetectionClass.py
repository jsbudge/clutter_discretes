# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:47:27 2019

@author: Josh

@purpose: Range-Velocity Map detection class
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *


class RVTargetDetection(object):

    def __init__(
            self, rangeInd, velInd, logMag, antAz, wavelength, PRF,
            DopFFTLength, nearRange, MPP, NoisePow, radar, radVelError,
            velStep, lonConv=0, latConv=0):
        # record the longitude and latitude conversion factors
        if (lonConv != 0 and latConv != 0):
            self.lonConv = lonConv
            self.latConv = latConv
        else:
            self.lonConv = radar.lonConv
            self.latConv = radar.latConv

        # record the info for determining the range and Doppler values
        self.nearRange = nearRange
        self.MPP = MPP
        self.PRF = PRF
        self.wrapVel = radar.wrapVel
        self.velPP = self.wrapVel * 2 / DopFFTLength
        self.DopFFTLength = DopFFTLength
        self.lamda = wavelength
        self.adc2Volts = radar.adc2Volts

        # record the azimuth uncertainty
        self.azUncertainty = radar.azUncertainty
        # let's just assign a value for the elevation error sigma
        self.eleSigma = 10.0  # meters
        # assign a default range sigma (std dev)
        self.rangeSigma = radar.rngRes * 100.0  # centimeters
        self.radVelSigma = (radVelError + velStep) * 100.0  # cm/s

        # initialize list of the indices
        self.indexList = [(rangeInd, velInd, logMag, antAz)]
        # initialize other variables to zero
        self.rangeM = 0.0
        self.radVelMPerS = 0.0
        self.maxMag = 0.0
        self.antAzR = 0.0
        # initialize the index count to 1
        self.indexCount = 1
        # maintain a flag for if this is a wrapping target or not
        self.isWrapping = False
        # initialize the target inertial position
        self.posI = zeros((3, 1), dtype='float64')
        # get the noise level of the radar 
        self.noiseP = NoisePow
        # initialize a merged flag
        self.merged = False

    def __str__(self):
        rangeVal, radVelVal, maxLogVal = self.getRangeVelocity()
        string = "Range: %0.2f m, radial velocity: %0.2f Hz, Max Response: " + \
                 "%0.2f dB, pixelCount: %d, wrapped: %d\n" % (rangeVal, radVelVal,
                                                              maxLogVal, self.indexCount, self.isWrapping)
        # string += "  Index list:\n"
        # for i in range(self.indexCount):
        #    string += "    rangeIndex: %d, velIndex: %d\n" % self.indexList[i]

        return string

    def addIndex(self, newRangeInd, newVelInd, newLogMag, newAntAz):
        """
        This function checks to see if the detected pixel should be added to
        this targets list based on its range and Doppler index. If the pixel
        is associated with the target, a 1 is returned, otherwise a 0.
        Input:
            rangeInd - the range index (or the row index of the range-Doppler 
                map)
            velInd - the doppler index (or the column index of the 
            range-Doppler map)
        Return:
            status - 1 for success, 0 for failure
        """
        addedIndex = 0
        # Now that we are no longer performing the FFT shift, there should be
        #   no problems with wrapping of the detections since the wrap point
        #   is in the middle of the array in velocity direction
        for i in range(self.indexCount):
            rangeInd, velInd, logMag, antAz = self.indexList[i]
            if (abs(newRangeInd - rangeInd) <= 1 \
                    and abs(newVelInd - velInd) <= 1):
                self.indexList.append(
                    (newRangeInd, newVelInd, newLogMag, newAntAz))
                self.indexCount += 1
                addedIndex = 1
                break

        return addedIndex

    def getRangeVelocity(self):
        """
        This function returns the range, radial velocity, and log magnitude
        """
        rangeVal = 0.0
        radVelVal = 0.0
        maxMagVal = 0.0
        antAzValR = 0.0
        if (self.rangeM != 0 and self.radVelMPerS != 0 and self.maxMag != 0):
            rangeVal = self.rangeM
            radVelVal = self.radVelMPerS
            maxMagVal = self.maxMag
            antAzValR = self.antAzR
        else:
            # Very first thing, let's resolve wrapping issues if this target 
            # has been flagged as being wrapped. It is really kind of sixes to 
            # know which way we need to unwrap without further information or 
            # more of a data history on the target. So, we could either choose 
            # up always, or we could try unwrapping toward the side with the 
            # greater mass. I really don't know if that will buy us anything or 
            # be of much benefit though. Or whether it will be worth the cost 
            # of computation. Maybe I'll do that later, I'm not sure buys me 
            # much of anything.
            aveRangeInd = 0.0
            aveRadVelInd = 0.0
            magSum = 0.0
            tempMax = -1e20
            aveCosAntAz = 0.0
            aveSinAntAz = 0.0
            for i in range(self.indexCount):
                magVal = self.indexList[i][2]
                antAz = self.indexList[i][3]
                if (magVal > tempMax):
                    tempMax = magVal
                aveRangeInd += self.indexList[i][0] * magVal
                tempVel = self.indexList[i][1]
                aveRadVelInd += tempVel * magVal
                aveCosAntAz += cos(antAz) * magVal
                aveSinAntAz += sin(antAz) * magVal
                magSum += magVal

            # finalize the computation of the average indices and convert to 
            # range and Doppler values
            aveRangeInd /= magSum
            aveRadVelInd /= magSum
            aveCosAntAz /= magSum
            aveSinAntAz /= magSum
            # perform wrapping of the radial velocity (since we are dealing
            #   with non-integer indices, we need to take into account the
            #   pixel boundary. Hence the (N-1)/2.)
            #            if( aveRadVelInd >= ( self.DopFFTLength - 1 ) / 2.0 ):
            #                aveRadVelInd -= self.DopFFTLength
            rangeVal = self.nearRange + aveRangeInd * self.MPP / 2
            radVelVal = -aveRadVelInd * self.velPP
            antAzValR = arctan(aveSinAntAz / aveCosAntAz)
            maxMagVal = tempMax
            self.rangeM = rangeVal
            self.radVelMPerS = radVelVal
            self.maxMag = maxMagVal
            self.antAzR = antAzValR

        return rangeVal, radVelVal, maxMagVal, antAzValR

    def setAsMerged(self):
        self.merged = True

    def mergeTargets(self, altTarget):
        """
        This function is responsible for merging two targets if they need to be
        """
        # I need to traverse the index list for each TargetDetection object and 
        # compare the pixels

        shouldMerge = False

        for sI in range(self.indexCount):
            sRange, sVel, sLogMag, sAntAz = self.indexList[sI]

            for aI in range(altTarget.indexCount):
                aRange, aVel, aLogMag, aAntAz = altTarget.indexList[aI]
                if abs(sRange - aRange) <= 1 and abs(sVel - aVel) <= 1:
                    # this means that the they have an indices that touch and
                    # they should be merged
                    shouldMerge = True
                    break
            if (shouldMerge):
                break

        # if they should be merged, let's merge the alternate target into this 
        # one
        if shouldMerge:
            for aI in range(altTarget.indexCount):
                self.indexList.append(altTarget.indexList[aI])
            # update the indexCount for this TargetDetection object
            self.indexCount += altTarget.indexCount
            altTarget.setAsMerged()

        return shouldMerge

    def estimateParameters(
            self, hAgl, airPos_i, airVel_i, boresightVec, effAz, dtedManager):
        """
        This function computes an estimate of the geocoordinate of the target
        as well as it's radial velocity given the input from the radar.
        Inputs:
            airPos_i - 3x1 numpy array with the inertial position of the 
                antenna
            airVel_i - 3x1 numpy array with the inertial velocity of the 
                antenna
            boresightVec - 3x1 numpy array with the normalized vector pointing
                in the direction of boresight
            dtedManager - Object of type of DTEDManager class for looking up
                DTED data
        """
        tRadVel = 0.0
        tarPos_i = zeros((3, 1), dtype='float64')
        tRange = 0.0
        # check if this has already been computed
        if (any(self.posI)):
            tarPos_i = self.posI.copy() + 0.0
        else:
            # first compute the Range and radiall velocity for the target
            tRange, tRadVel, maxVal, tAntAzR = self.getRangeVelocity()

            # compute an initial estimate of the grazing angle using the hAgl 
            # and range
            tDepI = arcsin(hAgl / tRange)
            tAzI = effAz - tAntAzR
            #            tAzI = effAz# REMOVE ME LATER!!!

            # formulate our initial estimate of the inertial target radial 
            # vector
            r_hatI = array([
                [cos(tDepI) * sin(tAzI)],
                [cos(tDepI) * cos(tAzI)],
                [-sin(tDepI)]])

            # get initial estimate of the target position
            tarPos_i = airPos_i + tRange * r_hatI
            # grab the elevation for the target position
            nlat = tarPos_i.item(1) / self.latConv
            nlon = tarPos_i.item(0) / self.lonConv
            tarAlt = dtedManager.getDTEDPoint(nlat, nlon)
            tarPos_i[2, 0] = tarAlt

            # set the iteration limit
            iterLimit = 2
            # initialize the altitude error
            altError = 1e3
            # rangeError = 1e3
            iterCount = 0
            while (abs(altError) > 0.01 and iterCount < iterLimit):
                """Step 1 - look-up the DTED value for the target position 
                    estimate"""
                # update the reference height
                hRef = airPos_i.item(2) - tarAlt

                """Step 2 - recompute the estimate of the target inertial 
                    depression angle and update the target inertial radial 
                    vector"""
                tDepI = arcsin(hRef / tRange)
                r_hatI = array([
                    [cos(tDepI) * sin(tAzI)],
                    [cos(tDepI) * cos(tAzI)],
                    [-sin(tDepI)]])

                """Step 3 - Uupdate the target position"""
                # update the target position
                tarPos_i = airPos_i + tRange * r_hatI

                """Step 7 - Calculate the altitude error to test for need to 
                    iterate again"""
                # now that we have an updated target position, we need to look 
                # up the target
                # altitude for this new lat/lon and then check the error
                nlat = tarPos_i.item(1) / self.latConv
                nlon = tarPos_i.item(0) / self.lonConv
                tarAlt = dtedManager.getDTEDPoint(nlat, nlon)
                # determine the update altitude error
                altError = tarPos_i.item(2) - tarAlt
                # print "After iteration %d alt-error:%0.5f, range-error:%0.5f"
                #    % (iterCount+1, altError, rangeError)
                # update the iteration count
                iterCount += 1

            if (iterCount > iterLimit):
                print("For this detection, we reached the iteration limit.")
            # Now that we have finished estimation of the target position, we 
            # can use the target inertial radial vector and the aircraft 
            # inertial velocity vector to estimate the target radial velocity

            # finalize the inertial radial vector based on the latest dted 
            # look-up update the reference height
            hRef = airPos_i.item(2) - tarAlt

            # recompute the estimate of the target inertial depression angle
            # and update the target inertial radial vector
            tDepI = arcsin(hRef / tRange)
            r_hatI = array([
                [cos(tDepI) * sin(tAzI)],
                [cos(tDepI) * cos(tAzI)],
                [-sin(tDepI)]])
            rCen_hatI = array([
                [cos(tDepI) * sin(effAz)],
                [cos(tDepI) * cos(effAz)],
                [-sin(tDepI)]])
            tarPos_i = airPos_i + r_hatI * tRange

            # Need to remove the residual aircraft induced Doppler that is a
            #   result of compensating range-Doppler only to the center of the
            #   beam of the antenna.
            cenAircraftRadialVel = (rCen_hatI.T.dot(airVel_i)).item(0)
            tarAircraftRadialVel = (r_hatI.T.dot(airVel_i)).item(0)
            residualAircraftVelocity = \
                tarAircraftRadialVel - cenAircraftRadialVel
            tRadVel += residualAircraftVelocity
            #            if( tRadVel > self.wrapVel ):
            #                tRadVel -= self.wrapVel * 2
            if (tRadVel < self.wrapVel * 2):
                tRadVel += self.wrapVel * 2
            if (tRadVel > 0):
                tRadVel -= self.wrapVel * 2
            self.tarRadVelMPerS = tRadVel

            # record the estimated parameters internal to the target object
            self.posI = tarPos_i.copy() + 0.0

        return tarPos_i, tRange, tRadVel, tAntAzR, tAzI

    def getStanagTargetReportData(self, targetReportIndex):
        """
        This function returns an array (list) with data in the expected format 
        to pass into the StanagGenerator class's generateTargetReport method
        """
        # This array needs to be 18 long, but only certain ones need to be 
        # filled in with information, and the rest are zeros or don't matter
        #  0: target report index
        #  1: target latitude (deg)
        #  2: target longitude (deg)
        #  5: target geodetic height (meters)
        #  6: target range rate or radial velocity (cm/s)
        #  7: target wrap velocity (cm/s)
        #  8: target SNR estimate (dB)
        #  9: classification enumeration
        #  11: slant range uncertainty (centimeters)
        #  12: cross range uncertainty (decimeters)
        #  13: height uncertainty (meters)
        #  14: radial velocity uncertainty (cm/s)
        stanagTargetData = [
            targetReportIndex, 39.1, 259.1, 0, 0, 1129, 302.5, 405.7, 12, 0, 0,
            0, 0, 0, 0, 0, 0, 0]
        # get the latitude and longitude from the position
        stanagTargetData[1] = self.posI.item(1) / self.latConv
        stanagTargetData[2] = self.posI.item(0) / self.lonConv
        stanagTargetData[5] = self.posI.item(2)
        # convert the range rate to cm/s and save it to the stanag target 
        # report
        stanagTargetData[6] = self.radVelMPerS * 1e2
        # compute and write out the target wrap velocity 
        stanagTargetData[7] = (self.lamda * self.PRF / 4.0) * 1e2
        # and then the SNR estimate
        SNR = (self.maxMag ** 2 - self.noiseP) / self.noiseP
        stanagTargetData[8] = round(10 * log10(SNR))
        # assign an unknown, live target classification (code 127)
        stanagTargetData[9] = 2
        # save out all of the uncertainty values (1 std dev)
        stanagTargetData[11] = self.rangeSigma
        # calculate the cross-range std dev using the far range small angle
        # approximation for the azimuth uncertainty and target range
        stanagTargetData[12] = self.azUncertainty * self.rangeM * 10.0
        stanagTargetData[13] = self.eleSigma
        stanagTargetData[14] = self.radVelSigma

        return stanagTargetData
