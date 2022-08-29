# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:16:17 2017
Updated on 10/11/2019

@author: josh

@purpose: Exo-clutter GMTI processing.
"""
from pylab import *
from pylab import fromfile, ceil, close, log2, log10, arange, zeros, ones, \
    fft, fftshift, kaiser, ifft, figure, imshow, colorbar, xlabel, ylabel, \
    title, legend, eye, pinv, axis, exp, linspace, kron, pi, floor, clim, \
    cos, sin, arccos, arcsin, tan, arctan, swapaxes, any, diag, norm, real, \
    array, sqrt, arctan2, zeros_like, randn, genfromtxt, axvline, plot, \
    nonzero, angle, around, axes, random
#from scipy.signal import chebwin, hanning, hamming, nuttall, parzen, triang
from STAP_helper import getDopplerLine, window_taylor, \
    getWindowedFFTRefPulseUpdated, rangeCompressData, getRotationOffsetMatrix
from SlimSDRDebugDataGMTIParserModule import SlimSDRGMTIDataParser
#from TrackBeforeDetectModule import DwellTrackManager
from ClutterCompensatedExoClutterGMTIModule import computeExoMDV, \
    detectExoClutterMoversRVMap, getExoClutterDetectedMoversRV
from MoverTruthDataModule import MoverTruthData
from ExoConfigParserModule import ExoConfiguration
from DTEDManagerModule import DTEDManager
#import ctypes
import time
from stanaggenerator import StanagGenerator
import matplotlib.animation as animation

# Constants
c0 = 299792458.0
kb = 1.3806503e-23
T0 = 290.0

# Conversions
DTR = pi / 180

# load in the user defined processing parameters
config = ExoConfiguration()
print( config )

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

#%%
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
xmlfilename = '%s/%s/%s.xml' % ( rawDirectory, dateString, sarName )
basename = '%s/%s/%s' % ( debugDirectory, dateString, sarName )
stanagName = '%s/%s/%s' % ( stanagDirectory, dateString, sarName )
truthName = '%s/%d/%s/GroundMoversTruthGPS_%s.dat' % (
    truthDirectory, year, dateString, dateString )
if (not TRUTH_EXISTS):
    truthName = ''

#%%
""" Open up our data files for reading"""
# open the moverInjectedData file for reading binary data
radar = SlimSDRGMTIDataParser( basename, xmlfilename, numChan, Ncpi )

# print radar info
print( radar )

# open the Truth data if it exists
if( TRUTH_EXISTS ):
    truthData = MoverTruthData(
        truthName, radar.gpsData[ 0 ].latConv, radar.gpsData[ 0 ].lonConv,
        radar.xmlData.channel.wavelengthM, radar.xmlData.PRF )
    
    print( truthData )

# initialize the DTED manager, which will then be passed around to everywhere
#   that needs to look-up DTED data
dtedManager = DTEDManager( config.dtedDir )

sg = StanagGenerator()
sg.openFileForWriting( stanagName )
# [mission_plan, flight_plan, platform_type, ]
flightPlan = '%02d%02d%02d%06d' % ( month, day, year - 2000, config.colTime )
missionData =\
    [ 'SlimSARLCMCX', flightPlan, 255, 'WBBellyPod', year, month, day ]
sg.writeMissionSegment( missionData )    

#%%
"""Calculate some parameters for processing"""
# define the rotation offsets from the IMU to the mounted gimbal frame
rotationOffset = getRotationOffsetMatrix(
    radar.xmlData.gimbalSettings.rollOffsetD * DTR,
    radar.xmlData.gimbalSettings.pitchOffsetD * DTR, 
    radar.xmlData.gimbalSettings.yawOffsetD * DTR )

# Grab the latitude and longitude conversion factors
latConv = radar.gpsData[ 0 ].latConv
lonConv = radar.gpsData[ 0 ].lonConv

# define the near, far, and center grazing angles (this is from the simulation)
cen_graze = radar.xmlData.gimbalSettings.depressionD / radar.RTD

# record the flight path information contained in the XML
fp_startN = radar.xmlData.flightLine.startLatD * latConv
fp_startN = radar.gpsData[ 0 ].rxNorthingM[ 0 ]
fp_startE = radar.xmlData.flightLine.startLonD * lonConv
fp_startE = radar.gpsData[ 0 ].rxEastingM[ 0 ]
fp_stopN = radar.xmlData.flightLine.stopLatD * latConv
fp_stopN = radar.gpsData[ 0 ].rxNorthingM[ -1 ]
fp_stopE = radar.xmlData.flightLine.stopLonD * lonConv
fp_stopE = radar.gpsData[ 0 ].rxEastingM[ -1 ]

# create a unit vector pointing in the direction of the flight path
flightPath = array( [ [ fp_stopE - fp_startE ], [ fp_stopN - fp_startN ] ] )
flightPath = flightPath / sqrt( flightPath.T.dot( flightPath ) )
fpHeading = arctan2( flightPath[ 0, 0 ], flightPath[ 1, 0 ] )

# calculate near range and some other items
hAgl = radar.xmlData.configAlt
PRF = radar.xmlData.PRF
lamda = radar.xmlData.channel.wavelengthM
srate = radar.xmlData.channel.srateHz
dt = 1 / srate
tp = radar.xmlData.channel.pulseLengthS
Nchan = radar.TxNchan * radar.RxNchan
# get this from the noise figure we use for the radar
F = 10.0**( 3.0 / 10.0 )
N0 = kb * T0 * F
NoisePow = N0 * srate
sigma_n = sqrt( NoisePow )
MPP = c0 / srate / range_interp_factor

# determine near and far range information
RxOnS = radar.xmlData.channel.RxOnTAC / radar.xmlData.TAC
RxOffS = radar.xmlData.channel.RxOffTAC / radar.xmlData.TAC
TxOnS = radar.xmlData.channel.TxOnTAC / radar.xmlData.TAC
TxOffS = radar.xmlData.channel.TxOffTAC / radar.xmlData.TAC

configNearRange = ( RxOnS - TxOnS - fDelay / radar.xmlData.TAC ) * c0 / 2.0
desiredNearRange = (
    RxOnS - TxOnS - fDelay / radar.xmlData.TAC \
        - tp * nearRange_partial_pulse_percent ) * c0 / 2.0
configFarRange = ( RxOffS - TxOnS - tp ) * c0 / 2.0
desiredFarRange = (
    RxOffS - TxOnS - tp * ( 1.0 - partial_pulse_percent ) ) * c0 / 2.0

# the length of the total convolution output
convOutputLength = radar.Nsam + radar.pulseLengthN - 1
# this is the number of bins with a full convolution
NFullConvBins = radar.Nsam - radar.pulseLengthN + 1

# calculate the bin index for the desired near range
rxPulseLeadingZeroPadLength = int( tp * nearRange_partial_pulse_percent / dt )

# calculate the number of bins from the desiredNearRangeOffsetInd
numRangeSamples = int( (
    RxOffS - RxOnS - tp * ( 1.0 - partial_pulse_percent ) \
        + tp * nearRange_partial_pulse_percent ) / dt ) + 1

# let's calculate the last valid range index
#Nrv = NFullConvBins * range_interp_factor
Nrv = numRangeSamples * range_interp_factor

# we will get the FastTimeFFTlength from the matched filter
FastTimeFFTlength = radar.ftFFTLength
# and inverse FFT length to get the desired range interpolation factor
FastTimeIFFTlength = FastTimeFFTlength * range_interp_factor
# and the Doppler FFT length
DopFFTlength = int( 2**( ceil( log2( Dop_interp_factor * Ncpi ) ) ) )
dopStep = PRF / DopFFTlength
velStep = dopStep * lamda / 2.0
#Nrv = FastTimeIFFTlength

#Nrv = FastTimeIFFTlength
# make an array of the range bins we want to look at
myRanges =\
    ( desiredNearRange * 2.0 + arange( Nrv, dtype='float32' ) * MPP ) / 2.0
# re-assign the far range
actualFarRange = myRanges[ -1 ]
# compute the middle range
midRange = ( myRanges[ -1 ] + myRanges[ 0 ] ) / 2.0

radar.setProcParameters( desiredNearRange, actualFarRange )

# compute the CPI sample time
Tcpi = radar.CPILength * radar.xmlData.PRI

myDopWindow = window_taylor( Ncpi, 11, -70 )
slowTimeWindow = myDopWindow.reshape( ( Ncpi, 1 ) ).dot( ones( ( 1, Nrv ) ) )


#%%
"""Set up the plot figure for the animation"""
velStep = radar.wrapVel * 2 / DopFFTlength
# Compute the separation between antenna phase centers
if( numChan > 1 ):
    antSep = abs(
        radar.xmlData.CCS.antSettings[ 0 ].portSettings[ 1 ].xOffsetM \
            - radar.xmlData.CCS.antSettings[ 0 ].portSettings[ 2 ].xOffsetM )
else:
    antSep = 1.0

magData = zeros( ( Nrv, Ncpi ), dtype='float64' )
# Grab the CPI of data

rawdata = radar.getCPIData( 100 )

# Get the position of the airlane and calculate the boresight Vec
antPos, antVel = radar.getPlatformPosVel()
#antPos, antVel = radar.getAntennaPosVel()
boreSightVec, effGrazeI, effAzI = radar.getCPIBoresightVector()
headingI = arctan2( antVel.item( 0 ), antVel.item( 1 ) )

# Compute the Doppler line along the center of the beam
dopCenLine, dopUpLine, dopDownLine, grazeOverRanges = getDopplerLine(
    effAzI, myRanges, antVel, antPos, radar, dtedManager, lamda,
    radar.azBeamwidth / 2.0, PRF )

"""Range compress the data from each channel for the CPI"""
slowtimes = arange( Ncpi ).reshape( ( Ncpi, 1 ) ) / PRF
slowtimePhases =\
    exp( 1j * 2 * pi * slowtimes.dot( -dopCenLine.reshape( 1, Nrv ) ) )
#slowtimePhases = 1.0
#phaseMat = slowtimePhases.reshape((Ncpi,1)).dot(ones((1, Nrv)))

# Python call for range compression of the CPI data
rcdopdata = []
for ci in range( Nchan ):
    cpidataP = rangeCompressData(
        rawdata[ ci ], radar.matchedFilter[ ci ], FastTimeFFTlength, 
        FastTimeIFFTlength, Nrv, Ncpi )
    # Doppler FFT
    rcdopdata.append( fft(
        cpidataP * slowTimeWindow * slowtimePhases, DopFFTlength, 0 ).T )

# Now compute the sum of the channels and the phase
chanSum = rcdopdata[ 0 ] + 0.0
if( Nchan > 1 ):
    chanSum += rcdopdata[ 1 ]
magData = 20 * log10( abs( chanSum ) )
if( Nchan > 1 ):
    antAz =\
        arcsin( angle( rcdopdata[ 0 ].conj() * rcdopdata[ 1 ] ) \
            * lamda / ( 2 * pi * antSep ) )

minMagData = magData.min()
maxMagData = magData.max()

fig = figure()
ax = axes(
    xlim = [ 0.5 * velStep, ( radar.wrapVel * 2 - velStep / 2.0) ],
    ylim = [ myRanges[ 0 ] - MPP / 4, myRanges[ -1 ] + MPP / 4 ] )
imageDat = ax.imshow(
    magData, cmap='jet', origin='lower', interpolation='nearest',
    extent=[ 0.5 * velStep, ( radar.wrapVel * 2 - velStep / 2.0 ), 
        myRanges[ 0 ] - MPP / 4, myRanges[ -1 ] + MPP / 4 ], aspect='auto',
    vmin=maxMagData - 60, vmax = maxMagData )
ax.set_xlabel( 'Radial Velocity (m/s)' )
ax.set_ylabel( 'Range (m)' )
ax.set_title( 'Nuts' )
# plot the clutter boundary lines
lowerVThreshDat = ax.axvline( x=1, color='red' )
upperVThreshDat = ax.axvline( x=5, color='red' )
#plot( detVels, detRanges, 'y^' )


truthPlotDat, =\
    ax.plot( 0, myRanges[ 0 ], 'ro', markersize=15, fillstyle='none' )

# zero out the DC frequencies in the matched filter
myMatchedFilter = []
myMatchedFilter.append( radar.matchedFilter[ 0 ] + 0.0 )
myMatchedFilter[ 0 ][:3] = 0
myMatchedFilter[ 0 ][-3:] = 0
myMatchedFilter.append( radar.matchedFilter[ 1 ] + 0.0 )
myMatchedFilter[ 1 ][:3] = 0
myMatchedFilter[ 1 ][-3:] = 0

startTime = time.time()

#%% Create the init and animate functions
def init():
    # Set the image data with random noise
    imageDat.set_array( random( ( Nrv, Ncpi ) ) )
    lowerVThreshDat.set_xdata( 20 )
    upperVThreshDat.set_xdata( 60 )
    truthPlotDat.set_xdata( 0 )
    truthPlotDat.set_ydata( myRanges[ 0 ] )
        
    return imageDat, lowerVThreshDat, upperVThreshDat, truthPlotDat

# animation function
def animate( i ):
    if( i % 10 == 0 ):
        print( "CPI %d / %d" % ( i, radar.NumberOfCPIs ) )
    # Grab the CPI of data
    rawdata = radar.getCPIData( i )
    
    # Get the position of the airlane and calculate the boresight Vec
    antPos, antVel = radar.getPlatformPosVel()
    #antPos, antVel = radar.getAntennaPosVel()
    boreSightVec, effGrazeI, effAzI = radar.getCPIBoresightVector()
    headingI = arctan2( antVel.item( 0 ), antVel.item( 1 ) )
    
    # Compute the Doppler line along the center of the beam
    dopCenLine, dopUpLine, dopDownLine, grazeOverRanges = getDopplerLine(
        effAzI, myRanges, antVel, antPos, radar, dtedManager, lamda,
        radar.azBeamwidth / 2.0, PRF )
    
    """Range compress the data from each channel for the CPI"""
    slowtimes = arange( Ncpi ).reshape( ( Ncpi, 1 ) ) / PRF
    slowtimePhases =\
        exp( 1j * 2 * pi * slowtimes.dot( -dopCenLine.reshape( 1, Nrv ) ) )
    #slowtimePhases = 1.0
    #phaseMat = slowtimePhases.reshape((Ncpi,1)).dot(ones((1, Nrv)))
    
    # Python call for range compression of the CPI data
    rcdopdata = []
    for ci in range( Nchan ):
        cpidataP = rangeCompressData(
            rawdata[ ci ], myMatchedFilter[ ci ], FastTimeFFTlength, 
            FastTimeIFFTlength, Nrv, Ncpi )
        # Doppler FFT
        rcdopdata.append( fft(
            cpidataP * slowTimeWindow * slowtimePhases, DopFFTlength, 0 ).T )
    
    # Now compute the sum of the channels and the phase
    chanSum = rcdopdata[ 0 ] + 0.0
#    antAz = 1.0
    if( Nchan > 1 ):
        chanSum += rcdopdata[ 1 ]
#        antAz =\
#            arcsin( angle( rcdopdata[ 0 ].conj() * rcdopdata[ 1 ] ) \
#                * lamda / ( 2 * pi * antSep ) )
    magData = 20 * log10( abs( chanSum ) )
        
    # Compute the upper and lower MDV along with the wrap velocity
#    wrapVel = PRF * lamda / 4.0
    MDV, MDVapproach, MDVrecede = computeExoMDV(
        antVel, radar.azBeamwidth, grazeOverRanges[ -1 ], effAzI, 
        headingI, Kexo )
    threshVel = max( MDV, radar.radVelRes )
    
    # let's grab the truth data's ranges and Dopplers
    radarTime = radar.getRadarTime()
    if( TRUTH_EXISTS ):
        truthR, truthDop, truthRadVel, tarRadVel, truthEle, truthAz =\
            truthData.getRangeDopplerAntennaAnglesForTime(
                radarTime, antPos, antVel, effAzI, effGrazeI, boreSightVec )
        #truthTargetReports = truthData.getStanagTargetTruthReportData(
        #    tarNumber, radarTime, antPos, radar)
        truthPos, truthVel = truthData.getPositionVelocityAtTime( radarTime )
    
    # Set the plot data
    imageDat.set_array( magData )
    lowerVThreshDat.set_xdata( threshVel + velStep )
    upperVThreshDat.set_xdata( radar.wrapVel * 2 - threshVel  + velStep)
    if( TRUTH_EXISTS ):
        truthPlotDat.set_xdata( truthRadVel[ 0 ] )
        truthPlotDat.set_ydata( truthR[ 0 ] )
    ax.set_title( 'CPINum: %d' % i )
#    maxImage = magData.max()
#    minImage = magData.min()
#    imageDat.set_clim( [ maxImage - 50, maxImage  ] )
    
    return imageDat, lowerVThreshDat, upperVThreshDat, truthPlotDat

## close the stanag file
sg.closeFileForWriting()

""" Run the animation """
anim = animation.FuncAnimation(
    fig, animate, init_func = init, frames = radar.NumberOfCPIs - 1,
    interval = radar.CPITime / 1e-3, blit = True )

# Save the animation as mp4 video file
anim.save( '%s/%s/%s_TruthRDVideo.mp4' \
    % ( videoDirectory, dateString, sarName ) )




