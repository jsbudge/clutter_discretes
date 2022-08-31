from numpy import *
from simulation_functions import enu2llh, getElevation, getElevationMap
from skimage import feature
from skimage.morphology import binary_erosion, binary_dilation
from targetlib import Target

c0 = 299792458.0
TAC = 125e6
DTR = pi / 180
inch_to_m = .0254


def getDopplerLine(effAzI, rangeBins, antVel, antPos, nearRangeGrazeR, azBeamwidthHalf, PRF, wavelength, origin):
    """Compute the expected Doppler vs range for the given platform geometry"""

    # Get the grazing angle in the near range
    # nearRangeGrazeR = radar.nearRangeD * DTR

    # compute the grazing angle for the near range to start
    (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
        effAzI, nearRangeGrazeR, antPos, rangeBins[0], origin)

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = arcsin((antPos[2] + origin[2] - surfaceHeight) / rangeBins)

    # this is a special version of Rvec (it is not 3x1, it is 3xNrv)
    Rvec = array([
        cos(grazeOverRanges) * sin(effAzI),
        cos(grazeOverRanges) * cos(effAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Doppler
    DopplerCen = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerCen > PRF / 2)
    DopplerCen[ind] -= PRF
    ind = nonzero(DopplerCen < -PRF / 2)
    DopplerCen[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    # (NOTE!!!: this is dependent
    # on the antenna pointing vector attitude with respect to the aircraft heading.
    # if on the left side, negative azimuth will be lower Doppler, and positive
    # azimuth will be higher, but on the right side, it will be the opposite, one
    # could use the sign of the cross-product to determine which it is.)
    # if (xmlData.gimbalSettings.lookSide.lower() == 'left'):
    eff_boresight = mean(array([
        cos(grazeOverRanges) * sin(effAzI),
        cos(grazeOverRanges) * cos(effAzI),
        -sin(grazeOverRanges)]), axis=1)
    ant_dir = cross(eff_boresight, antVel)
    azBeamwidthHalf *= sign(ant_dir[2])

    newAzI = effAzI - azBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerUp = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerUp > PRF / 2)
    DopplerUp[ind] -= PRF
    ind = nonzero(DopplerUp < -PRF / 2)
    DopplerUp[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    newAzI = effAzI + azBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerDown = \
        ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerDown > PRF / 2)
    DopplerDown[ind] -= PRF
    ind = nonzero(DopplerDown < -PRF / 2)
    DopplerDown[ind] += PRF
    return DopplerCen, DopplerUp, DopplerDown, grazeOverRanges


def computeGrazingAngles(effAzIR, grazeIR, antPos, theRange, origin):
    # initialize the pointing vector to first range bin
    Rvec = array([cos(grazeIR) * sin(effAzIR),
                  cos(grazeIR) * cos(effAzIR),
                  -sin(grazeIR)])

    groundPoint = antPos + Rvec * theRange
    nlat, nlon, alt = enu2llh(*groundPoint, origin)
    # look up the height of the surface below the aircraft
    surfaceHeight = getElevationMap(nlat, nlon)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - groundPoint[2, :] + origin[2]

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while any(abs(elevDiff) > heightDiffThresh) and numIterations < iterationThresh:
        hAgl = antPos[2, :] + origin[2] - surfaceHeight
        newGrazeR = arcsin(hAgl / theRange)
        if any(isnan(newGrazeR)) or any(isinf(newGrazeR)):
            print('NaN or inf found.')
        Rvec = array([cos(newGrazeR) * sin(effAzIR),
                      cos(newGrazeR) * cos(effAzIR),
                      -sin(newGrazeR)])
        groundPoint = antPos + Rvec * theRange
        nlat, nlon, alt = enu2llh(*groundPoint, origin)
        surfaceHeight = getElevationMap(nlat, nlon)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - alt
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations


def computeGrazingAngle(effAzIR, grazeIR, antPos, theRange, origin):
    # initialize the pointing vector to first range bin
    Rvec = array([cos(grazeIR) * sin(effAzIR),
                  cos(grazeIR) * cos(effAzIR),
                  -sin(grazeIR)])

    groundPoint = antPos + Rvec * theRange
    nlat, nlon, alt = enu2llh(*groundPoint, origin)
    # look up the height of the surface below the aircraft
    surfaceHeight = getElevation((nlat, nlon), False)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - alt

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while abs(elevDiff) > heightDiffThresh and numIterations < iterationThresh:
        hAgl = antPos[2] + origin[2] - surfaceHeight
        newGrazeR = arcsin(hAgl / theRange)
        if isnan(newGrazeR) or isinf(newGrazeR):
            print('NaN or inf found.')
        Rvec = array([cos(newGrazeR) * sin(effAzIR),
                      cos(newGrazeR) * cos(effAzIR),
                      -sin(newGrazeR)])
        groundPoint = antPos + Rvec * theRange
        nlat, nlon, alt = enu2llh(*groundPoint, origin)
        surfaceHeight = getElevation((nlat, nlon), False)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - alt
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations


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


def detectExoClutterMoversRVMap(rangeVel, upMDV, lowMDV, Pfa, wrapVel):
    # Compute the velocity bin spacing
    velPerBin = wrapVel * 2.0 / rangeVel.shape[1]
    upMDVInd = int(ceil(upMDV / velPerBin))
    lowMDVInd = int(floor((wrapVel * 2.0 + lowMDV) / velPerBin)) + 1

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
        cpiDetections, ant_pos, boresight, ranges, origin, ant_az):
    """Converts a 2D binary detection output for the ranges to a dictionary
    of movers with their average range bin and Velocity bin.
    Inputs:
        cpiDetections - 2D binary numpy array, size = number of range bins x 
            number of Velocity bins
        responseDB - 2D float numpy array containing the range-Velocity map
            response in dB
    """
    # get a list of all of the indices for the pixel detections
    rangeBins, velBins = nonzero(cpiDetections)
    targets = []
    if any(rangeBins):
        DopFFTLength = cpiDetections.shape[1]
        numDetectedPixels = rangeBins.size
        # convert the range and Doppler bins to floats
        # create an array of TargetDetection objects and iniitalize the first
        # one with the first pixel index in our list
        targets = [Target(velBins[0], rangeBins[0], ant_az[rangeBins[0], velBins[0]])]
        targets[0].calc(ant_pos, boresight, ranges, origin, velBins[0], DopFFTLength)
        # initialize the number of targets
        numTargets = 1

        # loop through all the detected pixels and assign them to a target
        for iP in range(1, numDetectedPixels):
            new_target = False
            # attempt to add the index to the existing targets
            for iT in reversed(range(numTargets)):
                new_target = targets[iT].accept(rangeBins[iP], velBins[iP], 2, 5)
                if new_target:
                    targets[iT].calc(ant_pos, boresight, ranges, origin, velBins[iP], DopFFTLength)
                    break

            # Check to see if the index was added to a target, if not we need
            # to create a new target and add the index to it
            if not new_target:
                t = Target(velBins[iP], rangeBins[iP], ant_az[rangeBins[iP], velBins[iP]])
                t.calc(ant_pos, boresight, ranges, origin, velBins[iP], DopFFTLength)
                targets.append(t)
                numTargets += 1
    return targets


def getExoClutterDetectedMoversRVBlob(
        cpiDetections, ant_pos, boresight, ranges, dopplers, ant_vel, fc, origin, ant_az):
    """Converts a 2D binary detection output for the ranges to a dictionary
    of movers with their average range bin and Velocity bin.
    Inputs:
        cpiDetections - 2D binary numpy array, size = number of range bins x
            number of Velocity bins
        responseDB - 2D float numpy array containing the range-Velocity map
            response in dB
    """
    # get a list of all of the indices for the pixel detections
    # Quick image processing of detection map
    det_map_eroded = binary_erosion(binary_dilation(cpiDetections))
    dets = feature.blob_doh(det_map_eroded)
    rangeBins = dets[:, 0].astype(int)
    velBins = dets[:, 1].astype(int)
    sizes = dets[:, 2]
    targets = []
    if any(rangeBins):
        DopFFTLength = cpiDetections.shape[1]
        numDetectedPixels = rangeBins.size
        # convert the range and Doppler bins to floats
        # create an array of TargetDetection objects and iniitalize the first
        # one with the first pixel index in our list
        targets = [Target(velBins[0], rangeBins[0], ant_az[rangeBins[0], velBins[0]], sizes[0])]
        targets[0].calc(ant_pos, boresight, ranges, origin, dopplers, fc, ant_vel)
        # initialize the number of targets
        numTargets = 1

        # loop through all the detected pixels and assign them to a target
        for iP in range(1, numDetectedPixels):
            new_target = False
            # attempt to add the index to the existing targets
            for iT in reversed(range(numTargets)):
                new_target = targets[iT].accept(rangeBins[iP], velBins[iP], boresight, ant_pos, origin, ranges, dopplers, fc)
                if new_target:
                    targets[iT].calc(ant_pos, boresight, ranges, origin, dopplers, fc, ant_vel)
                    break

            # Check to see if the index was added to a target, if not we need
            # to create a new target and add the index to it
            if not new_target:
                t = Target(velBins[iP], rangeBins[iP], ant_az[rangeBins[iP], velBins[iP]], sizes[iP])
                t.calc(ant_pos, boresight, ranges, origin, dopplers, fc, ant_vel)
                targets.append(t)
                numTargets += 1
    return targets


def getStanagDwellSegmentData(target_report_count, cpi_count, total_cpi_count, ant_pos_enu, ant_vel, ant_att, origin,
                              boresight, mid_range, az_half_beamwidth, scan_rate):
    """Returns an array (or list) with all of the expected items for a proper
    dwell segment according to the STANAG 4607 format."""
    # The array we return needs to be length 31, but not all of items in the array
    # need to be populated. The following indices within the array need to be
    # populated accordingly:
    #  0 - existence mask
    #  1 - revisit index (int16)
    #  2 - dwell index (int16)
    #  3 - last dwell of revisit (uint8 or char, this is binary)
    #  4 - target report count (int16)
    #  5 - dwell time (in milliseconds as int32)
    #  6 - latitude (deg as SA32)
    #  7 - longitude (deg as BA32)
    #  8 - altitude (cm as S32)
    #  14 - sensor track heading (degrees CW from true north as BA16)
    #  15 - sensor speed (mm/s as int32)
    #  16 - sensor vertical velocity (dm/s as int8)
    #  20 - platform azimuth (deg CW from north as BA16)
    #  21 - platform pitch (deg as SA16)
    #  22 - platform roll (deg as SA16)
    #  23 - dwell area center lat (deg as SA32)
    #  24 - dwell area center lon (deg as BA32)
    #  25 - range half extent (km a B16)
    #  26 - dwell angle half extent (deg as BA16)

    dwellData = [0.0] * 31
    dwellData[0] = MoverPositionData.existenceMask
    # Not sure yet how I'm going to to the revisit index (for now just make it 0)
    dwellData[1] = 0
    dwellData[2] = cpi_count
    dwellData[3] = 0
    if cpi_count == total_cpi_count:
        dwellData[3] = 1
    dwellData[4] = target_report_count
    dwellData[5] = self.CPITime * 1e3
    # let's grab the geocoordinates
    ant_pos = enu2llh(*ant_pos_enu, origin)
    dwellData[6] = ant_pos[0]
    dwellData[7] = ant_pos[1]
    dwellData[8] = ant_pos[2] * 1e2
    # use the velocity information to get the track heading and speed and
    # vertical velocity
    # first the heading
    sensorHeading = arctan2(ant_vel[0], ant_vel[1]) / DTR
    sensorHeading = sensorHeading + 360.0 if sensorHeading < 0.0 else sensorHeading
    dwellData[14] = sensorHeading
    # then the ground speed
    groundSpeed = sqrt(ant_vel[0] ** 2 + ant_vel[1] ** 2) * 1e3
    dwellData[15] = groundSpeed
    # then the vertical velocity
    dwellData[16] = ant_vel[2] * 1e1
    # Get the attitude information of the platform
    dwellData[20] = ant_att[0] / DTR
    dwellData[21] = ant_att[1] / DTR
    dwellData[22] = ant_att[2] / DTR
    # Compute the dwell area. To do this, we need to get the boresight vector
    # and then do some computations to determine where it is pointing on the
    # ground. Technically, we should use the DTED to get that information most
    # accurately. But, maybe an approximation would be fine so as to avoid
    # computationally expensive DTED searches.
    sceneCen = enu2llh(*(ant_pos_enu + boresight * mid_range), origin)
    dwellData[23] = sceneCen[0]
    dwellData[24] = sceneCen[1]
    # we need to compute the hAGL at the point of the aircraft
    hAgl = ant_pos[2] - getElevation((dwellData[6], dwellData[7]))
    farGroundRange = sqrt(self.farRange ** 2 - hAgl ** 2)
    nearGroundRange = sqrt(self.nearRange ** 2 - hAgl ** 2)
    dwellData[25] = ((farGroundRange - nearGroundRange) / 2.0) / 1e3
    dwellData[26] = az_half_beamwidth + scan_rate / 2

    return dwellData
