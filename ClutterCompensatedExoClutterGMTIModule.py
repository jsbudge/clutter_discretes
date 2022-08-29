# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:23:44 2019

@author: Josh

@purpose: To provide a place for helper functions to live that are related to
detection and estimation of GMTI targets from clutter-compensated
range-velocity maps.
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from numpy import any as npany
from RVTargetDetectionClass import RVTargetDetection


def computeExoMDV(pVel, azWidth, farGrazeI, effAzI, headingI, Kexo):
    # Calculate the squint angle
    velP = sqrt(pVel.T.dot(pVel)).item(0)
    squint = effAzI - headingI
    if (squint > pi):
        squint -= 2 * pi
    if (squint < -pi):
        squint += 2 * pi
    squint = abs(squint)

    phiFore = max(squint - Kexo * azWidth / (2 * cos(farGrazeI)), 0)
    phiAft = min(squint + Kexo * azWidth / (2 * cos(farGrazeI)), pi)

    MDVapproach = velP * cos(farGrazeI) * (cos(phiFore) - cos(squint))
    MDVrecede = velP * cos(farGrazeI) * (cos(phiAft) - cos(squint))
    # Use the maximum of the two as the MDV
    MDV = max(abs(MDVapproach), abs(MDVrecede))

    return MDV, MDVapproach, MDVrecede


def detectExoClutterMoversRVMap(rangeVel, upMDV, lowMDV, Pfa, radar):
    # Compute the velocity bin spacing
    velPerBin = radar.wrapVel * 2.0 / rangeVel.shape[1]
    upMDVInd = int(ceil(upMDV / velPerBin))
    lowMDVInd = int(floor((radar.wrapVel * 2.0 + lowMDV) / velPerBin)) + 1

    # compute the mean and standard deviation of the range-vel map in the
    # exo-clutter region
    magMean = rangeVel[:, upMDVInd: lowMDVInd].mean()
    noisePower = (magMean / sqrt(pi / 2)) ** 2
    threshold = sqrt(-noisePower * 2 * log(Pfa))
    # allocate an array for the detection map
    detectionMap = zeros_like(rangeVel)
    # perform the thresholding for detection
    detectionMap[:, upMDVInd: lowMDVInd] = \
        rangeVel[:, upMDVInd: lowMDVInd] > threshold

    return detectionMap, noisePower, threshold


def getExoClutterDetectedMoversRV(
        cpiDetections, responseDB, antAz, wavelength, wrapVel, PRF, nearRange,
        MPP, NoisePow, radar, radVelErrors, velStep, lonConv=0,
        latConv=0):
    if (lonConv == 0 and latConv == 0):
        lonConv = radar.lonConv
        latConv = radar.latConv

    """Converts a 2D binary detection output for the ranges to a dictionary
    of movers with their average range bin and Velocity bin.
    Inputs:
        cpiDetections - 2D binary numpy array, size = number of range bins x 
            number of Velocity bins
        responseDB - 2D float numpy array containing the range-Velocity map
            response in dB
    """
    returnTargets = []
    # get a list of all of the indices for the pixel detections
    rangeBins, velBins = nonzero(cpiDetections)
    if (npany(rangeBins)):
        DopFFTLength = cpiDetections.shape[1]
        numDetectedPixels = rangeBins.size
        # convert the range and Doppler bins to floats
        rangeBins = rangeBins + 0.0
        velBins = velBins + 0.0
        # create an array of TargetDetection objects and iniitalize the first 
        # one with the first pixel index in our list
        targets = [RVTargetDetection(
            rangeBins[0], velBins[0],
            responseDB[int(rangeBins[0]), int(velBins[0])],
            antAz[int(rangeBins[0]), int(velBins[0])], wavelength,
            PRF, DopFFTLength, nearRange, MPP, NoisePow, radar,
            radVelErrors[0], velStep, lonConv, latConv)]
        # initialize the number of targets
        numTargets = 1

        # loop through all the detected pixels and assign them to a target
        for iP in range(1, numDetectedPixels):
            # attempt to add the index to the existing targets
            for iT in reversed(range(numTargets)):
                wasIndexAdded = targets[iT].addIndex(
                    rangeBins[iP], velBins[iP],
                    responseDB[int(rangeBins[iP]), int(velBins[iP])],
                    antAz[int(rangeBins[iP]), int(velBins[iP])])
                if (wasIndexAdded):
                    break

            # Check to see if the index was added to a target, if not we need 
            # to create a new target and add the index to it
            if (not wasIndexAdded):
                targets.append(RVTargetDetection(
                    rangeBins[iP], velBins[iP],
                    responseDB[int(rangeBins[iP]), int(velBins[iP])],
                    antAz[int(rangeBins[iP]), int(velBins[iP])],
                    wavelength, PRF, DopFFTLength, nearRange, MPP, NoisePow,
                    radar, radVelErrors[int(rangeBins[iP])], velStep,
                    lonConv, latConv))
                numTargets += 1

        # this was nice, but doing things this more efficient way can result in 
        # some pixels not being associated correctly to the right target, so 
        # let's just add a merge step
        targetKeepList = list(range(numTargets))
        mergedTargets = []
        mergedTargetCount = 0
        for i in range(0, numTargets):
            if (targets[i].merged):
                continue
            # print("index list for target {}: {}".format(
            #    i, targets[i].indexList))
            for j in range(i + 1, numTargets):
                # print("index list for target {}: {}".format(
                #    j, targets[j].indexList))
                if (targets[j].merged):
                    continue
                tracksMerged = targets[i].mergeTargets(targets[j])
                if (tracksMerged):
                    mergedTargets.append(j)
                    # print("Going to remove %d from the targetKeepList" % (j))
                    # print("Contents of targetKeepList: {}".format(
                    #    targetKeepList))
                    targetKeepList.remove(j)
                    mergedTargetCount += 1

        # if any targets were merged, we need not return them          
        for i in range(numTargets - mergedTargetCount):
            returnTargets.append(targets[targetKeepList[i]])

    return returnTargets
