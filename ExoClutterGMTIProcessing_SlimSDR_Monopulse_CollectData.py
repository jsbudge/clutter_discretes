# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:16:17 2017
Updated on 10/11/2019

@author: josh

@purpose: Exo-clutter GMTI processing.
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
# from scipy.signal import chebwin, hanning, hamming, nuttall, parzen, triang
from STAP_helper import getDopplerLine, window_taylor, \
    getWindowedFFTRefPulseUpdated, rangeCompressData, getRotationOffsetMatrix
from SlimSDRDebugDataGMTIParserModule import SlimSDRGMTIDataParser
# import SDRParsing
# from TrackBeforeDetectModule import DwellTrackManager
from ClutterCompensatedExoClutterGMTIModule import computeExoMDV, \
    detectExoClutterMoversRVMap, getExoClutterDetectedMoversRV
from MoverTruthDataModule import MoverTruthData
from ExoConfigParserModule import ExoConfiguration
from DTEDManagerModule import DTEDManager
# import ctypes
import time
from stanaggenerator import StanagGenerator
import csv
from matplotlib.pyplot import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

# Constants
c0 = 299792458.0
kb = 1.3806503e-23
T0 = 290.0

# Conversions
DTR = pi / 180

# load in the user defined processing parameters
config = ExoConfiguration()
print(config)

# grab the config parameters
Ncpi = config.Ncpi
nearRange_partial_pulse_percent = config.nearRangePartialPulsePercent
partial_pulse_percent = config.farRangePartialPulsePercent
range_interp_factor = config.rangeInterpFactor
Dop_interp_factor = config.dopInterpFactor
Kexo = config.exoBroadeningFactor
Pfa = config.FAFactor
fDelay = config.fDelay
numChan = 2
TRUTH_EXISTS = config.truthExists

# processing presum value
Presum = 1

# %%
"""Generate the filenames we will use"""
rawDirectory = config.rawDir
debugDirectory = config.debugDir
stanagDirectory = config.stanagDir
videoDirectory = config.videoDir
truthDirectory = config.truthDir
month = config.month
day = config.day
year = config.year
dateString = config.dateString
sarName = config.sarFilename

# generate the name of the file with SAR data in the ARTEMIS format
xmlfilename = '%s/%s/%s.xml' % (rawDirectory, dateString, sarName)
basename = '%s/%s/%s' % (debugDirectory, dateString, sarName)
stanagName = '%s/%s/%s' % (stanagDirectory, dateString, sarName)
truthName = '%s/%d/%s/GroundMoversTruthGPS_%s.dat' % (
    truthDirectory, year, dateString, dateString)
truthCollectionFilename = \
    '%s/%s/%s_DetectionTruthData_crap.csv' \
    % (rawDirectory, dateString, sarName)
if not TRUTH_EXISTS:
    truthName = ''

# %%
""" Open up our data files for reading"""
# open the moverInjectedData file for reading binary data
radar = SlimSDRGMTIDataParser(basename, xmlfilename, numChan, Ncpi)

# print radar info
print(radar)

# open the Truth data if it exists
if TRUTH_EXISTS:
    truthData = MoverTruthData(
        truthName, radar.gpsData[0].latConv, radar.gpsData[0].lonConv,
        radar.xmlData.channel.wavelengthM, radar.xmlData.PRF)
    print(truthData)

# initialize the DTED manager, which will then be passed around to everywhere
#   that needs to look-up DTED data
dtedManager = DTEDManager(config.dtedDir)

sg = StanagGenerator()
sg.openFileForWriting(stanagName)
# [mission_plan, flight_plan, platform_type, ]
flightPlan = '%02d%02d%02d%06d' % (month, day, year - 2000, config.colTime)
missionData = \
    ['SlimSARLCMCX', flightPlan, 255, 'WBBellyPod', year, month, day]
sg.writeMissionSegment(missionData)

# %%
"""Calculate some parameters for processing"""
# define the rotation offsets from the IMU to the mounted gimbal frame
rotationOffset = getRotationOffsetMatrix(
    radar.xmlData.gimbalSettings.rollOffsetD * DTR,
    radar.xmlData.gimbalSettings.pitchOffsetD * DTR,
    radar.xmlData.gimbalSettings.yawOffsetD * DTR)

# Grab the latitude and longitude conversion factors
latConv = radar.gpsData[0].latConv
lonConv = radar.gpsData[0].lonConv

# calculate near range and some other items
hAgl = radar.xmlData.configAlt
PRF = radar.xmlData.PRF
lamda = radar.xmlData.channel.wavelengthM
srate = radar.xmlData.channel.srateHz
dt = 1 / srate
tp = radar.xmlData.channel.pulseLengthS
Nchan = radar.TxNchan * radar.RxNchan
# get this from the noise figure we use for the radar
F = 10.0 ** (3.0 / 10.0)
N0 = kb * T0 * F
NoisePow = N0 * srate
sigma_n = sqrt(NoisePow)
MPP = c0 / srate / range_interp_factor

# determine near and far range information
RxOnS = radar.xmlData.channel.RxOnTAC / radar.xmlData.TAC
RxOffS = radar.xmlData.channel.RxOffTAC / radar.xmlData.TAC
TxOnS = radar.xmlData.channel.TxOnTAC / radar.xmlData.TAC
TxOffS = radar.xmlData.channel.TxOffTAC / radar.xmlData.TAC

configNearRange = (RxOnS - TxOnS - fDelay / radar.xmlData.TAC) * c0 / 2.0
desiredNearRange = (
                           RxOnS - TxOnS - fDelay / radar.xmlData.TAC - tp * nearRange_partial_pulse_percent) * c0 / 2.0
configFarRange = (RxOffS - TxOnS - tp) * c0 / 2.0
desiredFarRange = (
                          RxOffS - TxOnS - tp * (1.0 - partial_pulse_percent)) * c0 / 2.0

# the length of the total convolution output
convOutputLength = radar.Nsam + radar.pulseLengthN - 1
# this is the number of bins with a full convolution
NFullConvBins = radar.Nsam - radar.pulseLengthN + 1

# calculate the bin index for the desired near range
rxPulseLeadingZeroPadLength = int(tp * nearRange_partial_pulse_percent / dt)

# calculate the number of bins from the desiredNearRangeOffsetInd
numRangeSamples = int((
                              RxOffS - RxOnS - tp * (1.0 - partial_pulse_percent)
                              + tp * nearRange_partial_pulse_percent) / dt) + 1

# let's calculate the last valid range index
# Nrv = NFullConvBins * range_interp_factor
Nrv = numRangeSamples * range_interp_factor

# we will get the FastTimeFFTlength from the matched filter
FastTimeFFTlength = radar.ftFFTLength
# and inverse FFT length to get the desired range interpolation factor
FastTimeIFFTlength = FastTimeFFTlength * range_interp_factor
# and the Doppler FFT length
DopFFTlength = int(2 ** (ceil(log2(Dop_interp_factor * Ncpi))))
dopStep = PRF / DopFFTlength
velStep = dopStep * lamda / 2.0
# Nrv = FastTimeIFFTlength

# Nrv = FastTimeIFFTlength
# make an array of the range bins we want to look at
myRanges = \
    (desiredNearRange * 2.0 + arange(Nrv, dtype='float32') * MPP) / 2.0
# re-assign the far range
actualFarRange = myRanges[-1]
# compute the middle range
midRange = (myRanges[-1] + myRanges[0]) / 2.0

radar.setProcParameters(desiredNearRange, actualFarRange)

# compute the CPI sample time
Tcpi = radar.CPILength * radar.xmlData.PRI

# %%
"""
Pre-compute the Doppler window used to reduce sidelobes in the Doppler spectrum
"""
# myDopWindow = zeros(Ncpi, dtype='float32', order='C')
myDopWindow = window_taylor(Ncpi, 11, -70)
slowTimeWindow = myDopWindow.reshape((Ncpi, 1)).dot(ones((1, Nrv)))

# %%
"""Loop over the CPI's looking for detections"""

# allocate memory for recording all of the detection results
truthRanges = []
truthDops = []
truthPositions = []
truthVelocities = []
truthRadVels = []
truthAzs = []
noisePowers = []
noisePowerLPFs = []
noisePowerVarLPFs = []
ONCE = 40

# Open a file for writing the data to a CSV file
DO_HEADER = True
csvID = open(truthCollectionFilename, 'w')
csvWriter = csv.writer(csvID)
csvHeader = [
    'Time', 'AirLat', 'AirLon', 'AirEle', 'Yaw', 'Pitch', 'Roll', 'Pan', 'Tilt',
    'TruthLat', 'TruthLon', 'TruthEle', 'TruthR', 'TruthRDot', 'LatHat',
    'LonHat', 'EleHat', 'RHat', 'RDotHat', 'AzimuthDOA']
if DO_HEADER:
    csvWriter.writerow(csvHeader)

# Instantiate our DwellTrackManager
# dtManager = DwellTrackManager(radar)

# Compute the separation between antenna phase centers
antSep = abs(
    radar.xmlData.CCS.antSettings[0].portSettings[1].xOffsetM \
    - radar.xmlData.CCS.antSettings[0].portSettings[2].xOffsetM)
antSep = 19.6e-2
# antSep = 1.0
detectionCount = 0
ONCE = 397
# ONCE = 390
ONCE = 1

# Initialize a low-pass filter version of the noisePowerLPF
LPFCutoffFreqHz = 0.25
LPFTimeConstant = 1 / (2 * pi * LPFCutoffFreqHz)
LPFDecayFactor = 1 - exp(-radar.CPITime / LPFTimeConstant)
noisePowerLPF = 0
noisePowerVarLPF = 0.02
timeSinceValidCPI = 0
RFICounter = 0

startTime = time.time()
# while radar.nextCPIExists():
while( ONCE ):
    timeSinceValidCPI += radar.CPITime
    # grab the data for the current CPI
    # rawdata, nextCPI = radar.getNextCPIData()
    rawdata = radar.getCPIData( ONCE )
    nextCPI = 0
    ONCE = 0
    # ONCE -= 1
    # ONCE += 1

    # Create an empty array to which we will append the data for a row
    cpiData = []

    # Get the position of the airlane and calculate the boresight Vec
    antPos, antVel = radar.getPlatformPosVel()
    # antPos, antVel = radar.getAntennaPosVel()
    boreSightVec, effGrazeI, effAzI = radar.getCPIBoresightVector()
    headingI = arctan2(antVel.item(0), antVel.item(1))

    """ Detection ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"""
    # Detect the movers in the exo-clutter. In the exo-clutter, in the absence 
    #   of a moving target, the radar returns are expected to be noise. As we 
    #   have a pretty good idea of the noise figure of the radar, we can use 
    #   this to set a threshold based on a false alarm rate.
    # Remember that we added in extra noise to hide the effects of the Hilbert 
    #   Transform

    dopCenLine, dopUpLine, dopDownLine, grazeOverRanges = getDopplerLine(
        effAzI, myRanges, antVel, antPos, radar, dtedManager, lamda,
        radar.azBeamwidth / 2.0, PRF)
    radVelErrors = abs(dopUpLine - dopDownLine) * lamda / 2.0
    # The high-fidelity Doppler line is based on the high-fidelity grazing 
    #   angle computation. Which should probably be reserved probably for when 
    #   doing either GMTI Backprojection or when better precision is needed for 
    #   the slow-time phase compensation.
    #    (dopCenLine, dopUpLine, dopDownLine, grazeOverRanges,
    #         surfaceHeights, numIterations) = getDopplerLineHiFi( effAzI, myRanges,
    #            antVel, antPos, radar, dtedName, dtedCorrection, lamda,
    #            azNullBeamwidthHalf, PRF)

    # check to make sure that we did not have an error
    if not any(rawdata[0]):
        print("We had a problem getting the CPI data for CPI #%d" % nextCPI)
        break

    if (nextCPI - 1) % 10 == 0:
        endTime = time.time()
        print("endTime - startTime = %0.9f" % (endTime - startTime))
        startTime = time.time()
        print("CPI #%d of %d" % (nextCPI - 1, radar.NumberOfCPIs))

    """Range compress the data from each channel for the CPI"""
    slowtimes = arange(Ncpi).reshape((Ncpi, 1)) / PRF
    slowtimePhases = \
        exp(1j * 2 * pi * slowtimes.dot(-dopCenLine.reshape(1, Nrv)))
    #    slowtimePhases = 1.0

    # Python call for range compression of the CPI data
    rcdopdata = []
    for ci in range(Nchan):
        # zero out the DC frequencies in the matched filter
        myMatchedFilter = radar.matchedFilter[ci] + 0.0
        myMatchedFilter[:3] = 0
        myMatchedFilter[-3:] = 0
        cpidataP = rangeCompressData(
            rawdata[ci], myMatchedFilter, FastTimeFFTlength,
            FastTimeIFFTlength, Nrv, Ncpi)
        # Doppler FFT
        rcdopdata.append(fft(
            cpidataP * slowTimeWindow * slowtimePhases, DopFFTlength, 0).T)

    # Now compute the sum of the channels and the phase
    chanSum = rcdopdata[0] + rcdopdata[1]
    magData = abs(chanSum)
    antAz = \
        arcsin(angle(rcdopdata[0].conj() * rcdopdata[1]) \
               * lamda / (2 * pi * antSep))
    #    antAz = zeros_like( antAz )
    #    monoPhase = angle( rcdopdata[ 0 ].conj() * rcdopdata[ 1 ] )

    # Compute the upper and lower MDV along with the wrap velocity
    wrapVel = PRF * lamda / 4.0
    MDV, MDVapproach, MDVrecede = computeExoMDV(
        antVel, radar.azBeamwidth, grazeOverRanges[-1], effAzI,
        headingI, Kexo)
    threshVel = max(MDV, radar.radVelRes)
    (detMap, noisePower, thresh) = detectExoClutterMoversRVMap(
        magData, threshVel, -threshVel, Pfa, radar)
    # Store the noise powers
    noisePowers.append(noisePower)
    if not noisePowerLPF:
        noisePowerLPF = 10 * log10(noisePower)
        noisePowerVarLPF = 0.02
        timeSinceValidCPI = 0
    else:
        noisePowerdB = 10 * log10(noisePower)
        # Only update the LPF if a valid CPI was encountered
        if nextCPI < 100000:
            if noisePowerdB < noisePowerLPF + 1.0:
                LPFDecayFactor = 1 - exp(-timeSinceValidCPI / LPFTimeConstant)
                deltaNoisePower = noisePowerdB - noisePowerLPF
                #            noisePowerLPF = ( 1 - LPFDecayFactor ) * noisePowerLPF \
                #                + LPFDecayFactor * noisePowerdB
                noisePowerLPF = noisePowerLPF + LPFDecayFactor * deltaNoisePower
                noisePowerVarLPF = (1 - LPFDecayFactor) \
                                   * (noisePowerVarLPF + LPFDecayFactor * deltaNoisePower ** 2)
                #            noisePowerVarLPF = ( 1 - LPFDecayFactor ) * noisePowerVarLPF \
                #                + LPFDecayFactor * deltaNoisePower**2
                timeSinceValidCPI = 0
            else:
                print("CPI with likely RFI encountered.")
                RFICounter += 1
        else:
            if (noisePowerdB < noisePowerLPF + 5.0 * sqrt(noisePowerVarLPF)):
                LPFDecayFactor = 1 - exp(-timeSinceValidCPI / LPFTimeConstant)
                deltaNoisePower = noisePowerdB - noisePowerLPF
                #            noisePowerLPF = ( 1 - LPFDecayFactor ) * noisePowerLPF \
                #                + LPFDecayFactor * noisePowerdB
                noisePowerLPF = noisePowerLPF + LPFDecayFactor * deltaNoisePower
                noisePowerVarLPF = (1 - LPFDecayFactor) \
                                   * (noisePowerVarLPF + LPFDecayFactor * deltaNoisePower ** 2)
                #            noisePowerVarLPF = ( 1 - LPFDecayFactor ) * noisePowerVarLPF \
                #                + LPFDecayFactor * deltaNoisePower**2
                timeSinceValidCPI = 0
            else:
                print("CPI with likely RFI encountered.")
                RFICounter += 1
    noisePowerLPFs.append(noisePowerLPF + 0.0)
    noisePowerVarLPFs.append(noisePowerVarLPF + 0.0)

    """ Segmentation |||||||||||||||||||||||||||||||||||||||||||||||||||||||"""
    targetList = getExoClutterDetectedMoversRV(
        detMap, magData, antAz, lamda, wrapVel, PRF, desiredNearRange, MPP,
        noisePower, radar, radVelErrors, velStep)
    # now have all of the detections go through and compute their position and 
    # range rate

    # Add the detections we got to the dwell track manager
    # nonRedundantNumTrackUpdates, updateKeys = dtManager.addNewDetections(
    #    targetList, radar)
    # numReportableTracks, reportableTrackIndices =\
    #    dtManager.getReportableTrackNumber()

    """ Parameter Estimation |||||||||||||||||||||||||||||||||||||||||||||||"""
    # First, compute the hAgl (to get closer to the actual hAgl, we should
    #   lookup the DTED value around the center of the swath)
    cenSwathPointHat = antPos + boreSightVec * midRange
    cenlat = cenSwathPointHat.item(1) / latConv
    cenlon = cenSwathPointHat.item(0) / lonConv
    hAglHat = antPos.item(2) - dtedManager.getDTEDPoint(cenlat, cenlon)

    detectionCount += len(targetList)
    # begin the loop over the targets and pass along to them all of the 
    #   information they should need to compute their position and radial
    #   velocity
    for i in range(len(targetList)):
        # parameters: hAgl, airPos_i, airVel_i, boresightVec, dtedName, 
        #   dtedCorrection
        tPosHat, tRangeHat, tRadVelHat, tAntAzR, tAzI = \
            targetList[i].estimateParameters(
                hAglHat, antPos, antVel, boreSightVec, effAzI, dtedManager)

    # We need to generate the STANAG info and write it out to the stream. We get 
    #   the data from the targets themselves for the target reports, and from 
    #   the MoverPositionData object for the dwell
    dwellData = radar.getStanagDwellSegmentData(
        len(targetList), dtedManager)
    #    dwellData = radar.getStanagDwellSegmentData(numReportableTracks)
    sg.writeDwellSegment(dwellData)
    #    dtManager.generateStanagReports(sg, hAglHat, radar, dtedName,
    #                                    dtedCorrection)
    tarNumber = 0
    for i in range(len(targetList)):
        targetReportData = \
            targetList[i].getStanagTargetReportData(tarNumber)
        sg.writeTargetReport(targetReportData)
        tarNumber += 1

    #    if (detectionCount):
    #        break
    # let's grab the truth data's ranges and Dopplers
    if TRUTH_EXISTS:
        radarTime = radar.getRadarTime()
        truthR, truthDop, truthRadVel, truthTarRadVel, truthEle, truthAz = \
            truthData.getRangeDopplerAntennaAnglesForTime(
                radarTime, antPos, antVel, effAzI, effGrazeI, boreSightVec)
        # truthTargetReports = truthData.getStanagTargetTruthReportData(
        #    tarNumber, radarTime, antPos, radar)
        truthPos, truthVel = truthData.getPositionVelocityAtTime(radarTime)

        # save the truthData to arrays for plotting later
        truthRanges.append(truthR)
        truthDops.append(truthDop)
        truthAzs.append(truthAz)
        truthPositions.append(truthPos)
        truthVelocities.append(truthVel)
        truthRadVels.append(truthRadVel)
        # determine if the target was in the beam or not
        truthInBeam = False
        for i in range(truthData.numTracks):
            # sg.writeTargetReport( truthTargetReports[i] )
            if (abs(truthAz[i]) < radar.azBeamwidth
                    and truthR[i] < myRanges[-1]
                    and truthR[i] > myRanges[0]):
                truthInBeam = True
        if (truthInBeam):
            # print( "Azimuth: {}, Range: {}".format(
            #    truthAz * 180 / pi, truthR ) )
            # rangeInd = int( around( ( truthR - myRanges[ 0 ] ) * 2 / MPP ) )
            # velInd = int( around( truthRadVel / velStep ) )
            # azEst = antAz[ rangeInd, velInd ] / DTR
            # print( "velInd: %d, rangeInd: %d, estimated Az: %0.4f deg" % (
            #    velInd, rangeInd, azEst ) )

            # Need to loop through the detections and find the one that
            #   corresponds to the truth data so that we can get the phase and
            #   other information from the range-Doppler map and spit it out to
            #   the CSV record. This is preferrable to simplying using the ind
            #   based on the truth data to grab it from the range-Doppler map
            #   because there could be errors in the truth data, and it also
            #   would be a single pixel look up instead of an average for all
            #   the pixels that would correspond to the target response there.
            rangeDeltaThresh = radar.rngRes * 7
            velDeltaThresh = radar.radVelRes * 1
            closeTargets = []
            if (len(targetList) > 0):
                for i in range(len(targetList)):
                    # We know there is only one truth target and so we are going
                    #   to forgo an additional for-loop over the truth targets
                    deltaRange = abs(truthR[0] - targetList[i].rangeM)
                    #                    print( 'deltaRange: %0.2f, truthVel: %0.2f, estVel: %0.2f' \
                    #                          % ( deltaRange, truthRadVel[ 0 ],
                    #                             targetList[ i ].radVelMPerS ) )
                    deltaVel = \
                        abs(truthRadVel[0] - targetList[i].radVelMPerS)

                    if (deltaRange < rangeDeltaThresh
                            and deltaVel < velDeltaThresh):
                        euclideanDist = sqrt(deltaRange ** 2 + deltaVel ** 2)
                        closeTargets.append(
                            (i, deltaRange, deltaVel, euclideanDist,
                             targetList[i].maxMag))
                # Let's choose the target that is close and brightest

                if (not closeTargets):
                    continue

                # Now, let's compute the euclidian distance. If any of them are
                #   close, we will take the one with the largest magnitude
                minEuclideanDist = 0
                maxMagnitude = 0
                closestCloseTarget = 0
                brightestCloseTarget = 0
                for i in range(len(closeTargets)):
                    if (i == 0):
                        minEuclideanDist = closeTargets[i][3]
                        closestCloseTarget = i
                        maxMagnitude = closeTargets[i][4]
                        brightestCloseTarget = i
                        continue
                    if (closeTargets[i][3] < minEuclideanDist):
                        minEuclideanDist = closeTargets[i][3]
                        closestCloseTarget = i
                    if (closeTargets[i][4] > maxMagnitude):
                        maxMagnitude = closeTargets[i][4]
                        brightestCloseTarget = i
                if (closestCloseTarget != brightestCloseTarget):
                    print("The closest target is not the brightest target!")
                    print("\tClosest target is #%d, with deltaR:%0.2f m, deltaVel:%0.2f m/s, magnitude:%0.2f dB" \
                          % (closeTargets[closestCloseTarget][0],
                             closeTargets[closestCloseTarget][1],
                             closeTargets[closestCloseTarget][2],
                             20 * log10(closeTargets[closestCloseTarget][4])))
                    print("\tBrightest target is #%d, with deltaR:%0.2f m, deltaVel:%0.2f m/s, magnitude:%0.2f dB" \
                          % (closeTargets[brightestCloseTarget][0],
                             closeTargets[brightestCloseTarget][1],
                             closeTargets[brightestCloseTarget][2],
                             20 * log10(closeTargets[brightestCloseTarget][4])))
                    print("\tWe will be taking the brightest target.")
                closestTarget = closeTargets[brightestCloseTarget][0]
                print("Detection #%d matches the truth data." \
                      % (closestTarget))
                # Append all of the data of record to the cpi data record
                #   array
                yawR, pitchR, rollR = radar.getPlatformAttitude()
                panR, tiltR = radar.getGimbalPanTilt()
                cpiData.append(radarTime)
                cpiData.append(antPos.item(1) / latConv)
                cpiData.append(antPos.item(0) / lonConv)
                cpiData.append(antPos.item(2))
                cpiData.append(yawR[0])
                cpiData.append(pitchR[0])
                cpiData.append(rollR[0])
                cpiData.append(panR)
                cpiData.append(tiltR)
                cpiData.append(truthPos[0].item(1) / latConv)
                cpiData.append(truthPos[0].item(0) / lonConv)
                cpiData.append(truthPos[0].item(2))
                cpiData.append(truthR[0])
                cpiData.append(truthTarRadVel[0])
                cpiData.append(
                    targetList[closestTarget].posI[1, 0] / latConv)
                cpiData.append(
                    targetList[closestTarget].posI[0, 0] / lonConv)
                cpiData.append(
                    targetList[closestTarget].posI[2, 0])
                cpiData.append(targetList[closestTarget].rangeM)
                cpiData.append(targetList[closestTarget].tarRadVelMPerS)
                # Note that in this case the antAzR is actually the phase
                #   and not the antenna azimuth angle
                cpiData.append(targetList[closestTarget].antAzR)
                csvWriter.writerow(cpiData)
                # break

csvID.close()
## close the stanag file
sg.closeFileForWriting()

noisePowerdBs = 10 * log10(abs(array(noisePowers)))
thresholds = array(noisePowerLPFs) + 1.0  # 5.0 * sqrt(array(noisePowerVarLPFs))
print("%d CPI's with RFI encountered." % (RFICounter))
#
#
velStep = radar.wrapVel * 2 / DopFFTlength
# Plot some of the results
# magData = fftshift(20*log10(abs(rcdopdata.T)), axes=1)
# magdopdata = fftshift(abs(rcdopdata.T), axes=1)
maxMagdata = magData.max()
detInd = nonzero(detMap)
detRanges = detInd[0] * MPP / 2 + desiredNearRange
detVels = -detInd[1] * radar.wrapVel * 2 / DopFFTlength
# magData -= maxlogdata
# get the indices for any of the detections
# detInd = nonzero(detMap)
# detRanges = detInd[0] * MPP/2 + desiredNearRange
# detDops = detInd[1] * (PRF / DopFFTlength) - PRF/2
# figure();imshow( antAz / DTR, cmap='jet', origin='lower',
#    extent=[ 0.5 * velStep, -( radar.wrapVel * 2 - velStep / 2.0 ), 
#            myRanges[ 0 ] - MPP / 4, myRanges[ -1 ] + MPP / 4 ] )
# colorbar()
# axis( 'tight' );xlabel( 'Doppler (Hz)' );ylabel( 'Range (m)' )
# xlabel( 'Radial Velocity (m/s)' )

figure()
imshow(20 * log10(abs(magData)), cmap='jet', origin='lower',
       extent=[0.5 * velStep, (radar.wrapVel * 2 - velStep / 2.0),
               myRanges[0] - MPP / 4, myRanges[-1] + MPP / 4])
# plot the clutter boundary lines
axvline(x=threshVel, color='red')
axvline(x=radar.wrapVel * 2 - threshVel, color='red')
# plot( dopCenLine * lamda / 2.0, myRanges,'k--', dopUpLine * lamda / 2.0,
#     myRanges, 'r-.', dopDownLine * lamda / 2.0, myRanges, 'r-.' )
# plot( detVels, detRanges, 'y^' )
# for i in range(len(targetList)):
#    detRange, detVel, maxLogVal = targetList[i].getRangeVelocity()
#    plot(detVel, detRange, 'ys')
# for i in range(len(truthRanges)):
#    plot(truthDops[i][0], truthRanges[i][0], 'ro', markersize=15, 
#         fillstyle='none')
if (TRUTH_EXISTS):
    for i in range(truthData.numTracks):
        plot(truthRadVel[i], truthR[i], 'ro', markersize=15,
             fillstyle='none')
    title('Range-Dop Tar in beam:%d, ant Az:%0.2f deg, CPI #%d (pulses %d-%d)' \
          % (truthInBeam, truthAz[0] * 180 / pi, radar.CPICounter - 1,
             (radar.CPICounter - 1) * Ncpi, radar.CPICounter * Ncpi - 1))
else:
    title('Range-Doppler CPI #%d (pulses %d-%d)' % (
        radar.CPICounter - 1, (radar.CPICounter - 1) * Ncpi,
        radar.CPICounter * Ncpi - 1))
colorbar()
axis('tight')
xlabel('Doppler (Hz)')
ylabel('Range (m)')
xlabel('Radial Velocity (m/s)')
clim([20 * log10(maxMagdata) - 60, 20 * log10(maxMagdata)])

print("We have finished all %d CPIs!!!" % radar.CPICounter)
