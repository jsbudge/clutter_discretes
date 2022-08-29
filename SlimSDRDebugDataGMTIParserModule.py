# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:01:47 2019

@author: Josh

@purpose: This module provides the classes for reading SlimSAR DEBUG data output
  for GMTI processing.
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from numpy import any as npany
from STAP_helper import getRotationOffsetMatrix, getBoresightVector, \
    getEffectiveInertialAzimuthAndGraze, gimbalToBody, bodyToInertial, \
    window_taylor
import xml.etree.ElementTree as ET
import datetime as dt
from dataclasses import dataclass, field
from scipy.io import loadmat
# from DataCorrection import getIdealChirp, getInverseFunction, \
#     getTransferFunction, getTimeDelayS, getCorrectionMatchedFilter
import matched_filters as mf
import SDRParsing

# Define constants
c0 = 299792458.0
ft2mConv = 0.3048
knots2mpers = 0.514444
DTR = pi / 180.0


@dataclass
class StriderInfo:
    version: str = ''
    buildDate: str = ''
    buildTime: str = ''


@dataclass
class LowPassFilter:
    decimation: int = 1
    passBandMHz: float = 0.0
    stopBandMHz: float = 0.0
    lpfCoefficients: list = field(default_factory=list)


@dataclass
class PortSettings:
    peakTxPowerW: float = 0.0
    maxGaindB: float = 0.0
    xOffsetM: float = 0.0
    yOffsetM: float = 0.0
    zOffsetM: float = 0.0


@dataclass
class AntennaSettings:
    side: str = ''
    squintD: float = 0.0
    azBeamwidthD: float = 0.0
    elBeamwidthD: float = 0.0
    dopplerBeamwidthD: float = 0.0
    depressionD: float = 0.0
    numAzimuthPhaseCenters: int = 1
    numElevationPhaseCenters: int = 1
    isSplitWeightAntenna: bool = False
    numPorts: int = 1
    portSettings: list = field(default_factory=list)


@dataclass
class FlightLine:
    lineType: str = ''
    altitudeM: float = 0.0
    startLatD: float = 0.0
    startLonD: float = 0.0
    stopLatD: float = 0.0
    stopLonD: float = 0.0


@dataclass
class CommonChannelSettings:
    configAltFt: float = 0
    velKnots: float = 0.0
    numCalPulses: int = 0
    circleSAREnabled: str = ''
    PRFBroadeningFactor: float = 0.0
    GMTI: bool = False
    PRITAC: int = 0
    CPITAC: int = 0
    altPRITAC: int = 0
    transmitPRFHz: float = 0.0
    numDigitalChannels: int = 0
    lpfInfo: LowPassFilter = LowPassFilter()
    numAntennas: int = 0
    antSettings: list = field(default_factory=list)


@dataclass
class SpotlightCenter:
    latD: float = 0.0
    lonD: float = 0.0
    altM: float = 0.0


@dataclass
class GimbalSettings:
    model: str = ''
    stabilizationMode: str = ''
    cutoffFreqHz: float = 0.0
    scanRateDPS: float = 0.0
    spotlightCenter: SpotlightCenter = SpotlightCenter()
    lookSide: str = ''
    panLimitsD: float = 0.0
    tiltLimitsD: float = 0.0
    updateRateHz: float = 0.0
    depressionD: float = 0.0
    squintD: float = 0.0
    model: str = ''
    xOffsetM: float = 0.0
    yOffsetM: float = 0.0
    zOffsetM: float = 0.0
    rollOffsetD: float = 0.0
    pitchOffsetD: float = 0.0
    yawOffsetD: float = 0.0
    initialCourseR: float = 0.0


@dataclass
class AGCSettings:
    initialAttenuationdB: int = 0
    maxThresholdADC: int = 0
    minThresholdADC: int = 0
    maxThresholdCount: int = 0
    minThresholdCount: int = 0


@dataclass
class ChannelSettings:
    freqBand: str = ''
    mode: int = 0
    waveformUpload: bool = False
    chirpDirection: str = ''
    cenFreqHz: float = 0.0
    wavelengthM: float = 0.0
    bandwidthHz: float = 0.0
    offsetVideoEnabled: bool = False
    DCOffsetHz: float = 0.0
    NCOHz: float = 0.0
    nearRangeD: float = 0.0
    farRangeD: float = 0.0
    pulseLengthPercent: float = 0.0
    presum: int = 1
    attenuationMode: str = ''
    agcSettings: AGCSettings = AGCSettings()
    RFIMitEnabled: bool = False
    upperBandEnabled: bool = False
    receiveOnly: bool = False
    LPFNumber: int = 0
    srateHz: float = 0.0
    upconverterSlot: int = 0
    receiverSlot: int = 0
    receiverSlice: int = 0
    transmitPort: int = 0
    receivePort: int = 0
    pulseLengthS: float = 0.0
    pulseLengthN: int = 0
    dutyCycle: float = 0.0
    dopplerPRFHz: float = 0.0
    effectivePRFHz: float = 0.0
    swathM: float = 0.0
    RxOnTAC: int = 0
    RxOffTAC: int = 0
    TxOnTAC: int = 0
    TxOffTAC: int = 0
    dataRateMBPS: float = 0.0
    simultaneousRxModes: int = 0
    chirpRateHzPerS: float = 0


class SDRXMLParser(object):

    def __init__(self, filename):
        # define the epoch
        epoch = dt.datetime(1980, 1, 6)

        self.filename = filename

        # get the date string from the filename
        ind = self.filename.rfind('_') - 8
        dtString = self.filename[ind: ind + 8]
        # define the TAC clock
        self.TAC = 125e6

        e = ET.parse(self.filename).getroot()
        # Get the Strider info
        striderInfoNode = e.findall('Strider_Info')[0]
        self.striderInfo = StriderInfo()
        self.striderInfo.version = \
            striderInfoNode.findall('Version')[0].text
        self.striderInfo.buildDate = \
            striderInfoNode.findall('Build_Date')[0].text
        self.striderInfo.buildTime = \
            striderInfoNode.findall('Build_Time')[0].text

        # for brevity get the slimsar info tree
        slimInfo = e.findall('SlimSDR_Info')[0]
        # Grab the SlimSDR model
        self.slimSDRModel = slimInfo.findall('SlimSDR_Model')[0].text
        self.slimSDRVersion = slimInfo.findall('Version')[0].text

        """ Parse the flight line info """
        self.flightLine = FlightLine()
        if slimInfo.findall('Flight_Line'):
            tempTree = slimInfo.findall('Flight_Line')[0]
            self.flightLine.lineType = \
                tempTree.findall('Flight_Line_Type')[0].text
            self.flightLine.altitudeM = \
                float(tempTree.findall('Flight_Line_Altitude_M')[0].text)
            if self.flightLine.lineType.lower() == 'straight':
                self.flightLine.startLatD = \
                    float(tempTree.findall('Start_Latitude_D')[0].text)
                self.flightLine.startLonD = \
                    float(tempTree.findall('Start_Longitude_D')[0].text)
                self.flightLine.stopLatD = \
                    float(tempTree.findall('Stop_Latitude_D')[0].text)
                self.flightLine.stopLonD = \
                    float(tempTree.findall('Stop_Longitude_D')[0].text)

        """ Parse the common channel settings """
        self.CCS = CommonChannelSettings()
        if (slimInfo.findall('System_Mode')):
            if (slimInfo.findall('System_Mode')[0].text.lower() == 'gmti'):
                self.CCS.GMTI = True
        tempTree = slimInfo.findall('Common_Channel_Settings')[0]
        self.CCS.configAltFt = \
            float(tempTree.findall('Altitude_Ft')[0].text)
        self.CCS.velKnots = \
            float(tempTree.findall('Velocity_Knots')[0].text)
        self.CCS.numCalPulses = \
            int(tempTree.findall('Num_Cal_Pulses')[0].text)
        self.CCS.circleSARMode = \
            tempTree.findall('Circle_SAR_Enabled')[0].text
        self.CCS.PRFBroadeningFactor = \
            float(tempTree.findall('PRF_Broadening_Factor')[0].text)
        if (tempTree.findall('GMTI')):
            self.CCS.GMTI = \
                tempTree.findall('GMTI')[0].text.lower() == 'true'
        elif (tempTree.findall('Configuration_Mode')):
            if (tempTree.findall('Configuration_Mode')[0].text.lower() \
                    == 'gmti'):
                self.CCS.GMTI = True
        self.CCS.PRITAC = \
            int(tempTree.findall('PRI_TAC')[0].text)
        self.CCS.CPITAC = \
            int(tempTree.findall('CPI_TAC')[0].text)
        self.CCS.altPRITAC = \
            int(tempTree.findall('ALT_PRI_TAC')[0].text)
        self.CCS.transmitPRFHz = \
            float(tempTree.findall('Transmit_PRF_Hz')[0].text)
        self.CCS.numDigitalChannels = \
            int(tempTree.findall('Num_Digital_Channels')[0].text)

        if (tempTree.findall('Low_Pass_Filter_1')):
            altTree = tempTree.findall('Low_Pass_Filter_1')[0]
            self.CCS.lpfInfo.decimation = \
                int(altTree.findall('Decimation_Factor')[0].text)
            self.CCS.lpfInfo.passBandMHz = \
                float(altTree.findall('Pass_Band_Freq_MHz')[0].text)
            self.CCS.lpfInfo.stopBandMHz = \
                float(altTree.findall('Stop_Band_Freq_MHz')[0].text)
            self.CCS.lpfInfo.lpfCoefficients += \
                altTree.findall('LPF_Coefficients')[0].text.split(' ')

        # Parse the antenna settings
        antTree = tempTree.findall('Antenna_Settings')[0]
        for k in range(4):
            antStr = 'Antenna_%d' % (k)
            if (antTree.findall(antStr)):
                altTree = antTree.findall(antStr)[0]
                self.CCS.antSettings.append(
                    AntennaSettings())
                self.CCS.numAntennas += 1
                self.CCS.antSettings[k].side = \
                    altTree.findall('Side')[0].text
                self.CCS.antSettings[k].squintD = \
                    float(altTree.findall(
                        'Antenna_Squint_Angle_D')[0].text)
                self.CCS.antSettings[k].azBeamwidthD = \
                    float(altTree.findall(
                        'Azimuth_Beamwidth_D')[0].text)
                self.CCS.antSettings[k].elBeamwidthD = \
                    float(altTree.findall(
                        'Elevation_Beamwidth_D')[0].text)
                self.CCS.antSettings[k].dopplerBeamwidthD = \
                    float(altTree.findall(
                        'Doppler_Beamwidth_D')[0].text)
                self.CCS.antSettings[k].depressionD = \
                    float(altTree.findall(
                        'Antenna_Depression_Angle_D')[0].text)
                # Check if the following fields exist, otherwise don't try to 
                #   read them or we will get an error
                isOldXML = True
                if (altTree.findall('Num_Azimuth_Phase_Centers')):
                    self.CCS.antSettings[k].numAzimuthPhaseCenters = \
                        int(altTree.findall(
                            'Num_Azimuth_Phase_Centers')[0].text)
                    isOldXML = False
                if (altTree.findall('Num_Elevation_Phase_Centers')):
                    self.CCS.antSettings[k].numElevationPhaseCenters = \
                        int(altTree.findall(
                            'Num_Elevation_Phase_Centers')[0].text)
                    isOldXML = False
                if (altTree.findall('Is_Split_Weight_Antenna')):
                    self.CCS.antSettings[k].isSplitWeightAntenna = \
                        altTree.findall('Is_Split_Weight_Antenna')[
                            0].text.lower() == 'true'
                # Check if the common antenna setting has the gain and power
                gainIsCommon = False
                if (altTree.findall('Peak_Transmit_Power_dB')):
                    self.CCS.antSettings[k].portSettings[0].peakTxPowerW = \
                        float(altTree.findall(
                            'Peak_Transmit_Power_dB')[0].text)
                    gainIsCommon = True
                if (altTree.findall('Max_Gain_dB')):
                    self.CCS.antSettings[k].portSettings[0].maxGaindB = \
                        float(altTree.findall('Max_Gain_dB')[0].text)
                    gainIsCommon = True

                if (isOldXML):
                    self.CCS.antSettings[k].portSettings.append(
                        PortSettings())
                    self.CCS.antSettings[k].portSettings[0].xOffsetM = \
                        float(altTree.findall(
                            'Antenna_X_Offset_M')[0].text)
                    self.CCS.antSettings[k].portSettings[0].yOffsetM = \
                        float(altTree.findall(
                            'Antenna_Y_Offset_M')[0].text)
                    self.CCS.antSettings[k].portSettings[0].zOffsetM = \
                        float(altTree.findall(
                            'Antenna_Z_Offset_M')[0].text)
                    self.CCS.antSettings[k].portSettings[0].peakTxPowerW = \
                        float(altTree.findall(
                            'Peak_Transmit_Power_dB')[0].text)
                    self.CCS.antSettings[k].portSettings[0].maxGaindB = \
                        float(altTree.findall('Max_Gain_dB')[0].text)
                else:
                    # Compute the number of ports
                    self.CCS.antSettings[k].numPorts = \
                        self.CCS.antSettings[k].numAzimuthPhaseCenters \
                        * self.CCS.antSettings[k].numElevationPhaseCenters
                    for pn in range(self.CCS.antSettings[k].numPorts):
                        # Generate Port name
                        portTag = 'Antenna_Port_%d' % (pn)
                        portTree = antTree.findall(portTag)[0]
                        self.CCS.antSettings[k].portSettings.append(
                            PortSettings())
                        self.CCS.antSettings[k].portSettings[pn].xOffsetM = \
                            float(portTree.findall(
                                'Port_X_Offset_M')[0].text)
                        self.CCS.antSettings[k].portSettings[pn].yOffsetM = \
                            float(portTree.findall(
                                'Port_Y_Offset_M')[0].text)
                        self.CCS.antSettings[k].portSettings[pn].zOffsetM = \
                            float(portTree.findall(
                                'Port_Z_Offset_M')[0].text)
                        # Check if the gain was common
                        if (not gainIsCommon):
                            self.CCS.antSettings[k].portSettings[
                                pn].peakTxPowerW = \
                                float(portTree.findall(
                                    'Peak_Transmit_Power_dB')[0].text)
                            self.CCS.antSettings[k].portSettings[
                                pn].maxGaindB = \
                                float(portTree.findall(
                                    'Max_Gain_dB')[0].text)

        # Convert some of the values from common channel settings to more
        #   useful units
        self.configVel = self.CCS.velKnots * knots2mpers
        self.configAlt = self.CCS.configAltFt * ft2mConv
        self.srate = 2e9
        if (self.CCS.lpfInfo.decimation):
            self.srate /= self.CCS.lpfInfo.decimation
        self.dt = 1.0 / self.srate

        """ Parse the gimbal settings """
        self.gimbalSettings = GimbalSettings()
        if (tempTree.findall('Gimbal_Settings')):
            gimTree = tempTree.findall('Gimbal_Settings')[0]
            self.gimbalSettings.model = \
                gimTree.findall('Gimbal_Model')[0].text
            self.gimbalSettings.stabilizationMode = \
                gimTree.findall('Stabilization_Mode')[0].text
            if (gimTree.findall('Stripmap_Settings')):
                self.gimbalSettings.cutoffFreqHz = \
                    float(gimTree.findall('Stripmap_Settings')[0].findall(
                        'LPF_Cut-Off_Frequency_Hz')[0].text)
            if (gimTree.findall('GMTI_Scanning_Settings')):
                self.gimbalSettings.scanRateDPS = \
                    float(gimTree.findall(
                        'GMTI_Scanning_Settings')[0].findall(
                        'Scan_Rate_Degrees_Per_Second')[0].text)
            if (gimTree.findall('Spotlight_Settings')):
                altTree = gimTree.findall('Spotlight_Settings')[0]
                self.gimbalSettings.spotlightCenter.latD = \
                    float(altTree.findall(
                        'Phase_Center_Latitude_D')[0].text)
                self.gimbalSettings.spotlightCenter.lonD = \
                    float(altTree.findall(
                        'Phase_Center_Longitude_D')[0].text)
                self.gimbalSettings.spotlightCenter.altM = \
                    float(altTree.findall(
                        'Phase_Center_Altitude_M')[0].text)
            self.gimbalSettings.lookSide = \
                gimTree.findall('Gimbal_Look_Side')[0].text
            self.gimbalSettings.panLimitsD = \
                float(gimTree.findall('Pan_Limits_D')[0].text)
            self.gimbalSettings.tiltLimitsD = \
                float(gimTree.findall('Tilt_Limits_D')[0].text)
            self.gimbalSettings.updateRateHz = \
                float(gimTree.findall('Update_Rate_Hz')[0].text)
            self.gimbalSettings.depressionD = \
                float(gimTree.findall(
                    'Gimbal_Depression_Angle_D')[0].text)
            self.gimbalSettings.squintD = \
                float(gimTree.findall('Squint_Angle_D')[0].text)
            self.gimbalSettings.xOffsetM = \
                float(gimTree.findall('Gimbal_X_Offset_M')[0].text)
            self.gimbalSettings.yOffsetM = \
                float(gimTree.findall('Gimbal_Y_Offset_M')[0].text)
            self.gimbalSettings.zOffsetM = \
                float(gimTree.findall('Gimbal_Z_Offset_M')[0].text)
            self.gimbalSettings.rollOffsetD = \
                float(gimTree.findall('Roll_D')[0].text)
            self.gimbalSettings.pitchOffsetD = \
                float(gimTree.findall('Pitch_D')[0].text)
            self.gimbalSettings.yawOffsetD = \
                float(gimTree.findall('Yaw_D')[0].text)
            if (gimTree.findall('Initial_Course_Angle_R')):
                self.gimbalSettings.initialCourseR = \
                    float(gimTree.findall(
                        'Initial_Course_Angle_R')[0].text)

        """ Parse the channel information """
        self.numTxChannels = 1
        # Only reading one channel until this is fixed
        self.channel = ChannelSettings()
        tempTree = slimInfo.findall('Channel_0')[0]
        self.channel.freqBand = tempTree.findall('Freq_Band')[0].text
        self.channel.mode = \
            int(tempTree.findall('Mode')[0].text, 16)
        self.channel.waveformUpload = \
            tempTree.findall('Waveform_Upload')[0].text.lower() == 'true'
        self.channel.chirpDirection = \
            tempTree.findall('Chirp_Direction')[0].text
        self.channel.cenFreqHz = \
            float(tempTree.findall('Center_Frequency_Hz')[0].text)
        self.channel.bandwidthHz = \
            float(tempTree.findall('Bandwidth_Hz')[0].text)
        self.channel.offsetVideoEnabled = \
            tempTree.findall('Offset_Video_Enabled')[0].text.lower() \
            == 'true'
        if (self.channel.offsetVideoEnabled):
            self.channel.DCOffsetHz = \
                float(tempTree.findall('DC_Offset_MHz')[0].text) * 1e6
        self.channel.NCOHz = float(tempTree.findall('NCO_Hz')[0].text)
        self.channel.nearRangeD = \
            float(tempTree.findall('Near_Range_D')[0].text)
        self.channel.farRangeD = \
            float(tempTree.findall('Far_Range_D')[0].text)
        self.channel.pulseLengthPercent = \
            float(tempTree.findall('Pulse_Length_Percent')[0].text)
        # Check if the presum factor is present
        if (tempTree.findall('Presum_Factor')):
            self.channel.presum = \
                int(tempTree.findall('Presum_Factor')[0].text)
        self.channel.attenuationMode = \
            tempTree.findall('Attenuation_Mode')[0].text

        # Parse the AGC settings
        if (self.channel.attenuationMode == 'AGC'):
            altTree = tempTree.findall('AGC_Settings')[0]
            self.channel.agcSettings.initialAttenuationdB = \
                int(altTree.findall('Initial_Attenuation_dB')[0].text)
            self.channel.agcSettings.maxThresholdADC = \
                int(altTree.findall('Max_Threshold_Percent')[0].text)
            self.channel.agcSettings.minThresholdADC = \
                int(altTree.findall('Min_Threshold_Percent')[0].text)
            self.channel.agcSettings.maxThresholdCount = \
                int(altTree.findall('Max_Threshold_Count')[0].text)
            self.channel.agcSettings.minThresholdCount = \
                int(altTree.findall('Min_Threshold_Count')[0].text)
        self.channel.RFIMitEnabled = \
            tempTree.findall('RFI_Mitigation_Enabled')[0].text.lower() \
            == 'true'
        self.channel.upperBandEnabled = \
            tempTree.findall('Upper_Band_Enabled')[0].text.lower() == 'true'
        self.channel.receiveOnly = \
            tempTree.findall('Receive_Only')[0].text.lower() == 'true'
        if (tempTree.findall('Channel_Low_Pass_Filter_Number')):
            self.channel.LPFNumber = \
                int(tempTree.findall(
                    'Channel_Low_Pass_Filter_Number')[0].text)
        self.channel.srateHz = \
            float(tempTree.findall('Sampling_Frequency_Hz')[0].text)
        self.channel.upconverterSlot = \
            int(tempTree.findall('Upconverter_Slot')[0].text)
        self.channel.receiverSlot = \
            int(tempTree.findall('Receiver_Slot')[0].text, 16)
        self.channel.receiverSlice = \
            int(tempTree.findall('Receiver_Slice')[0].text, 16)
        self.channel.transmitPort = \
            int(tempTree.findall('Transmit_Port')[0].text, 16)
        self.channel.receivePort = \
            int(tempTree.findall('Receive_Port')[0].text, 16)
        self.channel.pulseLengthS = \
            float(tempTree.findall('Pulse_Length_S')[0].text)
        self.channel.dutyCycle = \
            float(tempTree.findall('Duty_Cycle')[0].text) / 100
        self.channel.dopplerPRFHz = \
            float(tempTree.findall('Doppler_PRF_Hz')[0].text)
        self.channel.effectivePRFHz = \
            float(tempTree.findall('Effective_PRF_Hz')[0].text)
        self.channel.swathM = float(tempTree.findall('Swath_M')[0].text)
        self.channel.RxOnTAC = \
            int(tempTree.findall('Receive_On_TAC')[0].text)
        self.channel.RxOffTAC = \
            int(tempTree.findall('Receive_Off_TAC')[0].text)
        self.channel.TxOnTAC = \
            int(tempTree.findall('Transmit_On_TAC')[0].text)
        self.channel.TxOffTAC = \
            int(tempTree.findall('Transmit_Off_TAC')[0].text)
        self.channel.dataRateMBPS = \
            float(tempTree.findall('Data_Rate_MBPS')[0].text)
        if (tempTree.findall('Simultaneous_Receive_Modes')):
            self.channel.simultaneousRxModes = int(
                tempTree.findall('Simultaneous_Receive_Modes')[0].text,
                16)

        # Compute the more accurate pulse length, chirp rate, PRF
        # compute the wavelength
        self.channel.wavelengthM = c0 / self.channel.cenFreqHz
        self.channel.pulseLengthS = \
            (self.channel.TxOffTAC - self.channel.TxOnTAC) / self.TAC
        self.channel.pulseLengthN = \
            int(ceil(self.channel.pulseLengthS * self.channel.srateHz))
        self.channel.chirpRateHzPerS = \
            self.channel.bandwidthHz / self.channel.pulseLengthS
        if (self.channel.chirpDirection.lower() == 'down'):
            self.channel.chirpRateHzPerS *= -1
        self.PRI = \
            self.CCS.PRITAC * self.channel.presum / self.TAC
        self.PRF = 1 / self.PRI

        # Determine the mixdown frequency
        self.mixDownHz = 1e9
        if (self.channel.freqBand.lower() == 'x-band'):
            self.mixDownHz = 8e9
        if (self.channel.freqBand.lower() == 'ka-band'):
            self.mixDownHz = 33e9
        if (self.channel.upperBandEnabled):
            self.mixDownHz += 1e9

        # compute the number of GPS weeks
        gpxDT = dt.datetime.strptime(dtString, '%m%d%Y')
        self.numWeeks = int((gpxDT - epoch).days / 7)
        # now calculate the datetime for the beginning of the week
        begWeek = epoch + dt.timedelta(self.numWeeks * 7)
        gpsWeekDelta = gpxDT - begWeek
        self.gpsWeekSecDiff = gpsWeekDelta.total_seconds()

    def __str__(self):
        printString = "||||||||||||||||||||||||||||||||||||||||||||\n"
        printString += "||||||                                ||||||\n"
        printString += "||||||  Collection Settings from XML  ||||||\n"
        printString += "||||||                                ||||||\n"
        printString += "||||||||||||||||||||||||||||||||||||||||||||\n\n"
        printString += "  Center Frequency: %0.1f MHz\n" \
                       % (self.channel.cenFreqHz / 1e6)
        printString += "  Bandwidth: %0.1f MHz\n" \
                       % (self.channel.bandwidthHz / 1e6)
        printString += "  Sample Rate: %0.1f MHz\n" \
                       % (self.channel.srateHz / 1e6)
        printString += "  PRF: %0.2f Hz\n" % (self.PRF)
        printString += "  Effective PRI: %0.3f us\n" % (self.PRI * 1e6)
        printString += "  Pulse Length: %0.3f us\n" \
                       % (self.channel.pulseLengthS * 1e6)
        printString += "  Near Range Angle: %0.1f deg\n" \
                       % (self.channel.nearRangeD)
        printString += "  Far Range Angle: %0.1f deg\n" \
                       % (self.channel.farRangeD)
        printString += "  Chirp Rate: %0.2f GHz/s\n" \
                       % (self.channel.chirpRateHzPerS / 1e9)
        printString += "  Duty Cycle: %0.2f %%\n" \
                       % (self.channel.dutyCycle * 100)
        printString += "  Presum: %d\n" % (self.channel.presum)
        printString += "  Ant Side: %s\n" \
                       % (self.CCS.antSettings[0].side)
        if (self.CCS.antSettings[0].side.lower() \
                == 'gimballed'):
            printString += "    Gimbal Side: %s\n" \
                           % (self.gimbalSettings.lookSide)
        printString += "  Config Velocity: %0.1f m/s\n" % (self.configVel)
        printString += "  Config Altitude: %0.1f m\n" % (self.configAlt)
        printString += "  Gimbal Scan Rate: %0.1f deg/s\n" \
                       % (self.gimbalSettings.scanRateDPS)
        printString += "  Gimbal Depression: %0.1f deg\n" \
                       % (self.gimbalSettings.depressionD)
        printString += "  Gimbal Offset:\n"
        printString += "    X: %0.4f m\n    Y: %0.4f m\n    Z: %0.4f m\n" % (
            self.gimbalSettings.xOffsetM, self.gimbalSettings.yOffsetM,
            self.gimbalSettings.zOffsetM)
        printString += "  Gimbal Rotation Offset:\n"
        printString += "    Roll: %0.4f deg\n    Pitch: %0.4f deg\n" % (
            self.gimbalSettings.rollOffsetD, self.gimbalSettings.pitchOffsetD)
        printString += "    Yaw: %0.4f deg\n" \
                       % (self.gimbalSettings.yawOffsetD)
        for i in range(self.CCS.numAntennas):
            printString += "  Antenna %d Settings:\n" % (i)
            printString += "    Squint Angle: %0.1f deg\n" \
                           % (self.CCS.antSettings[i].squintD)
            printString += "    Azimuth Beamwidth: %0.1f deg\n" \
                           % (self.CCS.antSettings[i].azBeamwidthD)
            printString += "    Elevation Beamwidth: %0.1f deg\n" \
                           % (self.CCS.antSettings[i].elBeamwidthD)
            printString += "    Doppler Beamwidth: %0.1f deg\n" \
                           % (self.CCS.antSettings[i].dopplerBeamwidthD)
            printString += "    Antenna Depression Angle: %0.1f deg\n" \
                           % (self.CCS.antSettings[i].depressionD)
            printString += "    Number of azimuth phase centers: %d\n" \
                           % (self.CCS.antSettings[i].numAzimuthPhaseCenters)
            printString += "    Number of elevation phase centers: %d\n" \
                           % (self.CCS.antSettings[i].numElevationPhaseCenters)
            printString += "    Is Split-weight antenna: %d\n" \
                           % (int(self.CCS.antSettings[i].isSplitWeightAntenna))
            printString += "    Antenna %d has %d ports\n" \
                           % (i, self.CCS.antSettings[i].numPorts)
            for k in range(self.CCS.antSettings[i].numPorts):
                printString += "    Port %d Settings:\n" % (k)
                printString += "      X Offset: %0.4f m\n" \
                               % (self.CCS.antSettings[i].portSettings[k].xOffsetM)
                printString += "      Y Offset: %0.4f m\n" \
                               % (self.CCS.antSettings[i].portSettings[k].yOffsetM)
                printString += "      Z Offset: %0.4f m\n" \
                               % (self.CCS.antSettings[i].portSettings[k].zOffsetM)
                printString += "      Peak Transmit Power: %0.2f dB\n" \
                               % (self.CCS.antSettings[i].portSettings[
                                      k].peakTxPowerW)
                printString += "      Max Gain: %0.2f dB\n" \
                               % (self.CCS.antSettings[i].portSettings[k].maxGaindB)

        printString += "  Flight Line Start:\n"
        printString += "    Lat: %0.8f Deg\n    Lon: %0.8f Deg\n" \
                       % (self.flightLine.startLatD, self.flightLine.startLonD)
        printString += "  Flight Line Stop:\n"
        printString += "    Lat: %0.8f Deg\n    Lon: %0.8f Deg\n" \
                       % (self.flightLine.stopLatD, self.flightLine.stopLonD)
        printString += "  Flight Line Altitude: %0.3f m\n" \
                       % (self.flightLine.altitudeM)
        printString += "  GPS Week: %d\n" % (self.numWeeks)
        printString += "  Time delta for GPS week %0.1f sec\n\n" \
                       % (self.gpsWeekSecDiff)

        return printString


class SlimSDRGPSParser(object):

    def __init__(self, debugGPSName, chunkSize):
        # record the basename and CPILength
        self.fileName = debugGPSName
        self.chunkSize = chunkSize

        # open the uninterpolated GPS data for reading
        fid = open(self.fileName, 'rb')
        # parse all of the pre-jump corrections GPS data
        self.numINSFrames = int(fromfile(fid, 'uint32', 1, '')[0])
        self.latD = fromfile(fid, 'float64', self.numINSFrames, '')
        self.lonD = fromfile(fid, 'float64', self.numINSFrames, '')
        self.altM = fromfile(fid, 'float64', self.numINSFrames, '')
        self.nVelMPerS = fromfile(fid, 'float64', self.numINSFrames, '')
        self.eVelMPerS = fromfile(fid, 'float64', self.numINSFrames, '')
        self.uVelMPerS = fromfile(fid, 'float64', self.numINSFrames, '')
        self.rollR = fromfile(fid, 'float64', self.numINSFrames, '')
        self.pitchR = fromfile(fid, 'float64', self.numINSFrames, '')
        self.aziX = fromfile(fid, 'float64', self.numINSFrames, '')
        self.aziY = fromfile(fid, 'float64', self.numINSFrames, '')
        self.sec = fromfile(fid, 'float64', self.numINSFrames, '')
        self.systemTimeTAC = fromfile(fid, 'float64', self.numINSFrames, '')
        fid.close()

        # initialize the chunk counter
        self.chunkCounter = 0

    def __str__(self):
        printString = "GPS Data Parser Info:\n"
        printString += "-------------------------------\n"
        printString += "  numINSFrames: %d\n" % (self.numINSFrames)
        return printString

    def initializePPS(
            self, firstPPSTimeS, firstPPSSystemTimeTAC, systemTimeSlope):
        # record the PPS times and slope
        self.systemTimeSlope = systemTimeSlope
        self.beginFrame = nonzero(self.sec >= firstPPSTimeS)[0][0]
        # Now we need to set the current system time
        self.firstPPSTimeS = floor(firstPPSTimeS)
        self.firstPPSSystemTimeTAC = \
            (self.firstPPSTimeS - firstPPSTimeS) * systemTimeSlope \
            + firstPPSSystemTimeTAC

    def getNextChunk(self):
        ind1 = self.beginFrame + self.chunkSize * self.chunkCounter
        ind2 = self.beginFrame + self.chunkSize * (self.chunkCounter + 1)
        latD = self.latD[ind1: ind2]
        lonD = self.lonD[ind1: ind2]
        altM = self.altM[ind1: ind2]
        nVelMPerS = self.nVelMPerS[ind1: ind2]
        eVelMPerS = self.eVelMPerS[ind1: ind2]
        uVelMPerS = self.uVelMPerS[ind1: ind2]
        rollR = self.rollR[ind1: ind2]
        pitchR = self.pitchR[ind1: ind2]
        aziX = self.aziX[ind1: ind2]
        aziY = self.aziY[ind1: ind2]
        sec = self.sec[ind1: ind2]
        systemTimeTAC = self.systemTimeTAC[ind1: ind2]
        PPSTimeS, PPSSystemTimeTAC = self.getPPS(sec[-1])
        self.chunkCounter += 1

        # The prior chunk's last index
        priorLastIndex = maximum(0, ind1 - 1)

        return latD, lonD, altM, nVelMPerS, eVelMPerS, uVelMPerS, rollR, \
               pitchR, aziX, aziY, sec, systemTimeTAC, PPSTimeS, \
               PPSSystemTimeTAC, self.systemTimeTAC[priorLastIndex]

    def getPPS(self, latestTime):
        # check to see that 1 second has elapsed
        firstPPSTimeDiff = floor(latestTime - self.firstPPSTimeS)
        PPSTimeS = self.firstPPSTimeS + firstPPSTimeDiff
        PPSSystemTimeTAC = \
            firstPPSTimeDiff * self.systemTimeSlope + self.firstPPSSystemTimeTAC
        return PPSTimeS, PPSSystemTimeTAC

    def getAzimuth(self):
        return arctan2(self.aziX, self.aziY)

    def getAzimuthAt(self, ind):
        return arctan2(self.aziX[ind], self.aziY[ind])

    def getPosition(self, ind):
        return array([
            [self.lonD[ind]],
            [self.latD[ind]],
            [self.altM[ind]]])

    def getVelocity(self, ind):
        return array([
            [self.eVelMPerS[ind]],
            [self.nVelMPerS[ind]],
            [self.uVelMPerS[ind]]])

    def getPosVel(self, ind):
        return self.getPosition(ind), self.getVelocity(ind)


class SlimSDRGimbalParser(object):

    def __init__(self, debugGimbalName, chunkSize):
        # record the basename and CPILength
        self.fileName = debugGimbalName
        self.chunkSize = chunkSize

        # open the uninterpolated GPS data for reading
        fid = open(self.fileName, 'rb')
        # parse all of the pre-jump corrections GPS data
        self.numGimbalFrames = int(fromfile(fid, 'uint32', 1, '')[0])
        self.pan = fromfile(fid, 'float64', self.numGimbalFrames, '')
        self.tilt = fromfile(fid, 'float64', self.numGimbalFrames, '')
        self.systemTimeTAC = \
            fromfile(fid, 'float64', self.numGimbalFrames, '')
        fid.close()

        # initialize the chunk counter
        self.chunkCounter = 0

    def __str__(self):
        printString = "Gimbal Data Parser Info:\n"
        printString += "-------------------------------\n"
        printString += "  numINSFrames: %d\n" % (self.numINSFrames)
        return printString


class SlimSDRIntGPSParser(object):

    def __init__(self, basename, CPILength=128):
        # record the basename and CPILength
        self.basename = basename
        self.CPILength = CPILength

        # construct the filenames for the GPS data
        self.preCorrectionName = self.basename + "_preCorrectionsGPSData.dat"
        self.postCorrectionName = self.basename + "_postCorrectionsGPSData.dat"
        print("%s" % (self.preCorrectionName))

        # open the pre-corrections interpolated GPS data for reading
        fid = open(self.preCorrectionName, 'rb')
        # parse all of the pre-correction data
        self.numPreFrames = int(fromfile(fid, 'uint32', 1, '')[0])
        fid.seek(self.numPreFrames * 8 * 3, 1)
        # self.lat = fromfile(fid, 'float64', self.numPreFrames, '')
        # self.lon = fromfile(fid, 'float64', self.numPreFrames, '')
        # self.altPre = fromfile(fid, 'float64', self.numPreFrames, '')
        self.nVel = fromfile(fid, 'float64', self.numPreFrames, '')
        self.eVel = fromfile(fid, 'float64', self.numPreFrames, '')
        self.uVel = fromfile(fid, 'float64', self.numPreFrames, '')
        self.rollR = fromfile(fid, 'float64', self.numPreFrames, '')
        self.pitchR = fromfile(fid, 'float64', self.numPreFrames, '')
        self.aziR = fromfile(fid, 'float64', self.numPreFrames, '')
        # self.msecPre = fromfile(fid, 'float64', self.numPreFrames, '')
        fid.close()

        # open the post-corrections
        fid = open(self.postCorrectionName, 'rb')
        # parse all of the post-correction data
        self.numPostFrames = int(fromfile(fid, 'uint32', 1, '')[0])
        self.latConv = fromfile(fid, 'float64', 1, '')[0]
        self.lonConv = fromfile(fid, 'float64', 1, '')[0]
        self.rxEastingM = fromfile(fid, 'float64', self.numPostFrames, '')
        self.rxEastingM[abs(self.rxEastingM) < self.lonConv] *= self.lonConv
        self.rxNorthingM = fromfile(fid, 'float64', self.numPostFrames, '')
        self.rxNorthingM[abs(self.rxNorthingM) < self.latConv] *= \
            self.latConv
        self.rxAltM = fromfile(fid, 'float64', self.numPostFrames, '')
        fid.seek(self.numPostFrames * 8 * 3, 1)
        # self.txEastingM = fromfile(fid, 'float64', self.numPostFrames, '')
        # self.txNorthingM = fromfile(fid, 'float64', self.numPostFrames, '')
        # self.txAltM = fromfile(fid, 'float64', self.numPostFrames, '')
        self.aziPostR = fromfile(fid, 'float64', self.numPostFrames, '')
        self.sec = fromfile(fid, 'float64', self.numPostFrames, '')
        fid.close()

    def __str__(self):
        printString = "GPS Data Parser Info:\n"
        printString += "-------------------------------\n"
        printString += "  numPreFrames: %d\n  numPostFrames: %d\n" \
                       % (self.numPreFrames, self.numPostFrames)
        printString += "  latConv: %0.8f\n  lonConv: %0.8f\n\n" \
                       % (self.latConv, self.lonConv)
        return printString

    def getCPIAttitudeData(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength

        return self.aziR[begInd:endInd], self.pitchR[begInd:endInd], \
               self.rollR[begInd:endInd]

    def getCPIMeanAttitude(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength
        # calculate the mean of the attitude data
        meanCosAz = cos(self.aziR[begInd:endInd]).mean()
        meanSinAz = sin(self.aziR[begInd:endInd]).mean()
        meanYaw = arctan2(meanSinAz, meanCosAz)
        meanPitch = self.pitchR[begInd:endInd].mean()
        meanRoll = self.rollR[begInd:endInd].mean()

        return array([[meanYaw], [meanPitch], [meanRoll]])

    def getCPIPos(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength

        return self.rxEastingM[begInd:endInd], \
               self.rxNorthingM[begInd:endInd], self.rxAltM[begInd:endInd]

    def getCPIMeanVel(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength
        # calculate the means of the velocity
        meanVel = zeros((3, 1))
        meanVel[0, 0] = self.eVel[begInd:endInd].mean()
        meanVel[1, 0] = self.nVel[begInd:endInd].mean()
        meanVel[2, 0] = self.uVel[begInd:endInd].mean()

        return meanVel

    def getCPIMeanPosVel(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength
        # calculate the means of the position
        meanPosition = zeros((3, 1))
        meanPosition[0, 0] = self.rxEastingM[begInd:endInd].mean()
        meanPosition[1, 0] = self.rxNorthingM[begInd:endInd].mean()
        meanPosition[2, 0] = self.rxAltM[begInd:endInd].mean()
        # calculate the means of the velocity
        meanVel = zeros((3, 1))
        meanVel[0, 0] = self.eVel[begInd:endInd].mean()
        meanVel[1, 0] = self.nVel[begInd:endInd].mean()
        meanVel[2, 0] = self.uVel[begInd:endInd].mean()

        return meanPosition, meanVel

    def getTime(self, cpiInd):
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength

        midTime = (self.sec[endInd] + self.sec[begInd]) / 2

        return midTime


class SlimSDRIntGimbalParser(object):

    def __init__(self, basename, CPILength=128):
        # record the filename and CPILength
        self.filename = basename + "_GimbalData.dat"
        self.CPILength = CPILength
        # open the file for reading binary
        fid = open(self.filename, 'rb')
        # parse all of the gimbal data
        self.numFrames = int(fromfile(fid, 'uint32', 1, '')[0])
        self.pan = fromfile(fid, 'float64', self.numFrames, '')
        self.tilt = fromfile(fid, 'float64', self.numFrames, '')

    def __str__(self):
        printString = "Gimbal Data Parser Info:\n"
        printString += "-------------------------------\n"
        printString += "  numFrames: %d\n\n" % (self.numFrames)

        return printString

    def getCPIRotationData(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength

        return self.pan[begInd:endInd], self.tilt[begInd:endInd]

    def getCPIMeanRotationData(self, cpiInd):
        # compute beginning and ending indices into the data
        begInd = cpiInd * self.CPILength
        endInd = (cpiInd + 1) * self.CPILength

        return self.pan[begInd:endInd].mean(), \
               self.tilt[begInd:endInd].mean()


class SlimSDRGMTIDataParser(object):
    headerBytes = 4 + 4
    # existenceMask = uint64(0xFF071FC39E000000)
    # existence mask with uncertainties included
    existenceMask = uint64(0xFF071FC39E780000)
    adc2Volts = 800.0 * 1e-3 / 4096

    def __init__(self, collectName, xmlFilename, numChan=2, CPILength=128):
        self.RTD = 180 / pi
        # Simplify things by assuming a start offset index for the pulse
        self.startOffset = 120

        # record the basename and CPILength
        self.collectName = collectName
        self.xmlFilename = xmlFilename
        # number of channels
        self.TxNchan = 1
        self.RxNchan = numChan
        self.CPILength = CPILength

        # parse the XML file data
        self.xmlData = SDRXMLParser(self.xmlFilename)

        # Use Jeff's SlimSDR Parser for python to get the channel data for coputing
        #   the advanced matched filter using the RF waveform
        sdr = SDRParsing.load(self.xmlFilename[:-4], import_pickle=False)

        # To make this this class compatible with previous Python code in
        #   STAP_helper that was designed to work with HeaderParser, we need to
        #   define a number member variables directly in the SDRXMLParser
        self.fc = self.xmlData.channel.cenFreqHz
        self.lamda = self.xmlData.channel.wavelengthM
        self.BW = self.xmlData.channel.bandwidthHz
        self.freqOffset = 0
        if (self.xmlData.channel.offsetVideoEnabled):
            self.freqOffset = self.xmlData.channel.DCOffsetHz
        self.T_r = self.xmlData.PRI
        self.PRI = self.xmlData.PRI
        self.PRF = self.xmlData.PRF
        self.adc_on = self.xmlData.channel.RxOnTAC / self.xmlData.TAC
        self.tp = self.xmlData.channel.pulseLengthS
        self.kr = self.xmlData.channel.chirpRateHzPerS
        self.NCO = self.xmlData.channel.NCOHz
        self.pulseLengthS = self.xmlData.channel.pulseLengthS
        self.pulseLengthN = self.xmlData.channel.pulseLengthN
        self.srateHz = self.xmlData.channel.srateHz
        self.chirpRateHzPerS = self.xmlData.channel.chirpRateHzPerS
        self.refDelayN = 120 // self.xmlData.CCS.lpfInfo.decimation
        # Determine the RF mixdown frequency
        self.mixDownHz = self.xmlData.mixDownHz
        self.azBeamwidth = self.xmlData.CCS.antSettings[0].azBeamwidthD * DTR
        self.nearRangeD = self.xmlData.channel.nearRangeD

        # precompute the rotation offset matrix
        self.rotBtoMG = \
            getRotationOffsetMatrix(
                self.xmlData.gimbalSettings.rollOffsetD * DTR,
                self.xmlData.gimbalSettings.pitchOffsetD * DTR,
                self.xmlData.gimbalSettings.yawOffsetD * DTR)

        # create a 3x1 vector of the IMU to gimbal axes rotation center offset
        self.gimbalOffsetM = array([
            [self.xmlData.gimbalSettings.xOffsetM],
            [self.xmlData.gimbalSettings.yOffsetM],
            [self.xmlData.gimbalSettings.zOffsetM]])

        # record if left or right side based on xml
        self.isLeft = 0
        if (self.xmlData.gimbalSettings.lookSide.lower() == "left"):
            self.isLeft = 1

        # calculate the CPI Time
        self.CPITime = self.xmlData.PRI * self.CPILength
        # Get the scan rate in radians per second
        scanRateRPS = self.xmlData.gimbalSettings.scanRateDPS * DTR
        # calculate the angle covered during a scan
        self.scanAngleR = self.CPITime * scanRateRPS
        # Get the azimuth beamwidth in radians
        azBeamwidthR = \
            self.xmlData.CCS.antSettings[0].azBeamwidthD \
            * DTR
        # compute the azimuth uncertainty
        self.azUncertainty = azBeamwidthR + self.scanAngleR
        # calculate the range resolution of the system
        rangeBroadeningFactor = 1.25
        self.rngRes = rangeBroadeningFactor * c0 \
                      / (2.0 * self.xmlData.channel.bandwidthHz)
        # calculate the radial Velocity resolution
        dopplerBroadeningFactor = 2.5
        self.radVelRes = dopplerBroadeningFactor * c0 * self.xmlData.PRF / \
                         (self.xmlData.channel.cenFreqHz * self.CPILength)
        self.dopRes = \
            dopplerBroadeningFactor * self.xmlData.PRF / self.CPILength
        self.wrapVel = self.xmlData.PRF * self.xmlData.channel.wavelengthM / 4.0
        # compute the time to scan the beamwidth
        if (scanRateRPS != 0):
            self.beamwidthScanTime = azBeamwidthR / scanRateRPS

        # I need to loop through the number of channels and put everything into
        #   lists
        self.gpsData = []
        self.gimbalData = []
        self.rawDataName = []
        self.fid = []
        self.refPulse = []
        self.priorMatchedFilter = []
        self.matchedFilter = []
        self.ITF = []
        self.agcData = []
        self.sysTimeTAC = []
        self.Nsam = 0
        self.idealChirp = 0
        self.ftFFTLength = 0
        self.numFullConvBins = 0
        for i in range(self.RxNchan):
            # Generate the channel string
            channelName = 'Channel_%d_%s_%d_GHz' % (
                i + 1, self.xmlData.channel.freqBand,
                self.xmlData.mixDownHz / 1e9)
            # Generate the basename per channel
            baseName = '%s_%s_VV' % (self.collectName, channelName)
            # Get the interpolated GPS data
            self.gpsData.append(
                SlimSDRIntGPSParser(baseName, self.CPILength))
            # Get the interpolated Gimbal data
            self.gimbalData.append(
                SlimSDRIntGimbalParser(baseName, self.CPILength))
            # Record the raw data names
            self.rawDataName.append('%s_RawData.dat' % baseName)

            # Load in the file, get the ID and assign relevant parameters
            self.fid.append(open(self.rawDataName[i], 'rb'))
            # read the header information
            self.Npulses = \
                int(fromfile(self.fid[i], 'uint32', 1, '')[0])
            self.Nsam = int(fromfile(self.fid[i], 'uint32', 1, '')[0])
            # read the AGC data and system time
            self.agcData.append(
                fromfile(self.fid[i], 'uint8', self.Npulses, ''))
            self.sysTimeTAC.append(
                fromfile(self.fid[i], 'float64', self.Npulses, ''))
            # The rest of the data is stored frame by frame, so I can load those
            #   in only when I need them

            # Compute the number of full convolutions and the FFT length
            self.numFullConvBins = self.Nsam - self.pulseLengthN + 1
            self.ftFFTLength = \
                int(2 ** ceil(log2(self.Nsam + self.pulseLengthN - 1)))

            # Get the advanced matched filter using the RF waveform
            advMatchedFilter = mf.GetAdvMatchedFilter(
                sdr[i], nbar=5, SLL=-35)
            self.refPulse.append(sdr[i].cal_chirp)
            # Save the matched filter
            self.matchedFilter.append(advMatchedFilter)

        # get the lat and lon conversion factor from the gpsData
        self.latConv = self.gpsData[0].latConv
        self.lonConv = self.gpsData[0].lonConv

        # record the total bytes before the pulse data
        self.leadingBytes = \
            SlimSDRGMTIDataParser.headerBytes + self.Npulses \
            * (self.agcData[0].itemsize + self.sysTimeTAC[0].itemsize)

        # determine the total number of CPI's for the collection
        self.NumberOfCPIs = self.Npulses // self.CPILength
        # initialize the CPI counter
        self.CPICounter = 0

        # assign the length of a
        self.datatype = dtype('int16')
        # calculate the expected size of one pulse
        self.singleChanPulseSize = self.Nsam
        # the expected size of one pulse for all channels
        self.combChanPulseSize = \
            self.RxNchan * self.TxNchan * self.singleChanPulseSize
        # the expected size of all the pulses and channels in a CPI
        self.CPIDataSize = self.CPILength * self.singleChanPulseSize * 2
        self.firstRadarTimeS = self.gpsData[0].getTime(0)

        # set all of the cpi position and attitude related fields to zero
        self.cpiPlatformVel = 0
        self.cpiPlatformPos = 0
        self.cpiAntennaPos = 0
        self.cpiAttitude = 0
        self.cpiBoresightVec = 0
        self.cpiAzI = 0
        self.cpiGrazeI = 0
        self.time = 0

        self.rotGPtoI = zeros((3, 3))

    def __str__(self):
        printString = self.xmlData.__str__()
        printString += "Raw Data Info:\n"
        printString += "-------------------------------\n"
        printString += "  Number of Pulses: %d\n" % (self.Npulses)
        printString += "  Samples per pulse: %d\n" % (self.Nsam)
        printString += "  Number of CPIs  %d\n" % (self.NumberOfCPIs)
        printString += "  Left over pulses: %d\n\n" \
                       % (self.Npulses - self.NumberOfCPIs * self.CPILength)
        printString += self.gpsData.__str__()
        printString += self.gimbalData.__str__()

        return printString

    def setProcParameters(self, nearRange, farRange):
        self.nearRange = nearRange
        self.farRange = farRange
        self.midRange = (nearRange + farRange) / 2.0

        # compute the half beamwidth in degrees
        self.halfBeamwidth = \
            self.xmlData.CCS.antSettings[0].azBeamwidthD / 2
        self.CPIScanRateD = \
            self.CPITime * self.xmlData.gimbalSettings.scanRateDPS

    def nextCPIExists(self):
        """This function returns true if another CPI still exists."""
        return (self.CPICounter < self.NumberOfCPIs)

    def getNextCPIData(self):
        """This function simply returns all of the data for the next CPI
        if it is not beyond the last CPI. Otherwise, it returns a zero.
        It also returns the current CPI Index.
        """
        retData = []
        # check to make sure that we are not beyond our last CPI
        if (self.CPICounter < self.NumberOfCPIs):
            for i in range(self.RxNchan):
                data = fromfile(
                    self.fid[i], self.datatype, self.CPIDataSize, '')
                data = data.astype('float32')
                data = data[::2] + 1j * data[1::2]
                # reshape the data febore returning it
                data.resize((self.CPILength, self.Nsam))
                # we need to undo the attenuation and convert to floats
                begInd = self.CPICounter * self.CPILength
                endInd = (self.CPICounter + 1) * self.CPILength
                attenuation = \
                    10.0 ** (self.agcData[i][begInd:endInd] / 20.0)
                attens = attenuation.reshape((self.CPILength, 1)).dot(
                    ones((1, self.Nsam)))
                retData.append(data * attens)

            # increment the CPICounter
            self.CPICounter += 1

        # set all of the cpi position and attitude related fields to zero
        self.cpiPlatformVel = 0
        self.cpiPlatformPos = 0
        self.cpiAntennaPos = 0
        self.cpiAttitude = 0
        self.cpiBoresightVec = 0
        self.cpiAzI = 0
        self.cpiGrazeI = 0
        self.time = 0

        return retData, self.CPICounter

    def getCPIData(self, cpiIndex):
        """This functions returns the requested CPI's data if it is within the
        CPI range of the collection. Otherwise, it returns a zero.
        """
        retData = []
        if (cpiIndex >= 0 and cpiIndex < self.NumberOfCPIs):
            self.CPICounter = cpiIndex
            for i in range(self.RxNchan):
                # seek the file to the correct position before reading the data
                seekPoint = self.CPICounter \
                            * (self.CPIDataSize * self.datatype.itemsize) \
                            + self.leadingBytes
                self.fid[i].seek(seekPoint)
                data = fromfile(
                    self.fid[i], self.datatype, self.CPIDataSize, '')
                data = data.astype('float32')
                data = data[::2] + 1j * data[1::2]
                # reshape the data before returning it
                data.resize((self.CPILength, self.Nsam))

                # we need to undo the attenuation and convert to floats
                begInd = self.CPICounter * self.CPILength
                endInd = (self.CPICounter + 1) * self.CPILength
                attenuation = \
                    10.0 ** (self.agcData[i][begInd:endInd] / 20.0)
                attens = attenuation.reshape((self.CPILength, 1)).dot(
                    ones((1, self.Nsam)))
                retData.append(data * attens)
            # increment the CPI Counter so that it points to the next one
            self.CPICounter += 1

        # set all of the cpi position and attitude related fields to zero
        self.cpiPlatformVel = 0
        self.cpiPlatformPos = 0
        self.cpiAntennaPos = 0
        self.cpiAttitude = 0
        self.cpiBoresightVec = 0
        self.cpiAzI = 0
        self.cpiGrazeI = 0
        self.time = 0

        return retData

    def getPlatformPosVel(self):
        """Returns the average easting, northing, and altitude of the aircraft
        for the pulses of the designated CPI. 
        return - 3x1 numpy array of platform position in easting, northing, 
            and altitude in meters, and 3x1 numpy array of easting, northing, 
            and altitude velocity of the platform in meters/sec"""
        if (not npany(self.cpiPlatformPos) \
                or not npany(self.cpiPlatformVel)):
            self.cpiPlatformPos, self.cpiPlatformVel = \
                self.gpsData[0].getCPIMeanPosVel(self.CPICounter - 1)

        return self.cpiPlatformPos, self.cpiPlatformVel

    def getAntennaPosVel(self):
        """Returns the average easting, northing, and altitude of the antenna
        phase center for the pulses of the designated CPI. 
        return - 3x1 numpy array of antenna phase center position in easting, 
            northing, and altitude in meters, and 3x1 numpy array of easting, 
            northing, and altitude velocity of the platform in meters/sec"""
        if (not npany(self.cpiAntennaPos)):
            # NOTE: To simplify, I'm just going to use the zeroth channel
            # grab all of the yaw, pitch, and roll values for the CPI
            yaw, pitch, roll = \
                self.gpsData[0].getCPIAttitudeData(self.CPICounter - 1)
            # grab all of the pan and tilt values for the CPI
            pan, tilt = \
                self.gimbalData[0].getCPIRotationData(self.CPICounter - 1)
            # grab all of the IMU position information for the CPI
            north, east, alt = \
                self.gpsData[0].getCPIPos(self.CPICounter - 1)
            # we are going to compute the average position of the antenna phase
            #   center for the entire CPI. Initialize it to zero here.
            aveEast = 0.0
            aveNorth = 0.0
            aveAlt = 0.0
            # loop through all the pulses in the CPI
            for pulse in range(self.CPILength):
                # get the phase center offset for the pulse
                pcDelta_i = self.getPhaseCenterInertialCorrection(
                    pan[pulse], tilt[pulse], yaw[pulse], pitch[pulse],
                    roll[pulse])
                aveEast += east[pulse] + pcDelta_i[0, 0]
                aveNorth += north[pulse] + pcDelta_i[1, 0]
                aveAlt += alt[pulse] + pcDelta_i[2, 0]
            # finalize the average computation by dividing by the CPILength
            self.cpiAntennaPos = \
                array([[aveEast], [aveNorth], [aveAlt]]) \
                / self.CPILength
        if (not npany(self.cpiPlatformVel)):
            self.cpiPlatformVel = \
                self.gpsData[0].getCPIMeanVel(self.CPICounter - 1)

        return self.cpiAntennaPos, self.cpiPlatformVel

    def getPhaseCenterInertialCorrection(
            self, alpha_az, alpha_el, yaw, pitch, roll):
        """Returns the 3x1 numpy array with the inertial translational
        correction to be applied to the INS position value to have the position
        of the antenna phase center."""

        # rotate the antenna offsets (provided in the gimbal frame) into the
        #   the body frame of the INS
        antDelta_b = gimbalToBody(
            self.rotBtoMG, alpha_az, alpha_el,
            self.xmlData.CCS.antSettings[0].xOffsetM,
            self.xmlData.CCS.antSettings[0].yOffsetM,
            self.xmlData.CCS.antSettings[0].zOffsetM)
        # add the antenna offset in the body frame to the offsets measured from
        #   the INS center to the gimbal axes center of rotation
        totDelta_b = antDelta_b + self.gimbalOffsetM
        # finish the rotation of the combined offset from the body frame into
        #   the inertial frame
        delta_i = bodyToInertial(
            yaw, pitch, roll, totDelta_b[0, 0], totDelta_b[1, 0],
            totDelta_b[2, 0])

        # return the inertial correction
        return delta_i

    def getPlatformAttitude(self):
        """Returns the average azimuth, pitch, and roll of the aircraft for the
        pulses of the designated CPI.
        return - 3x1 numpy array of azimuth, pitch, roll in degrees"""
        if (not npany(self.cpiAttitude)):
            self.cpiAttitude = \
                self.gpsData[0].getCPIMeanAttitude(self.CPICounter - 1)

        return self.cpiAttitude

    def getGimbalPanTilt(self):
        """Return the average pan and tilt of the gimbal for the CPI"""
        return self.gimbalData[0].getCPIMeanRotationData(
            self.CPICounter - 1)

    def getCPIBoresightVector(self):
        """Return the average inertial boresight pointing vector of the antenna
        for the pulses of the designated CPI.
        return - 3x1 numpy array with the normalized boresight vector (easting, 
            northing, altitude)"""
        if (not npany(self.cpiBoresightVec) and not npany(self.cpiGrazeI) \
                and not npany(self.cpiAzI)):
            yaw, pitch, roll = \
                self.gpsData[0].getCPIAttitudeData(self.CPICounter - 1)
            pan, tilt = \
                self.gimbalData[0].getCPIRotationData(self.CPICounter - 1)
            self.cpiBoresightVec = array([[0.0], [0.0], [0.0]])
            self.rotGPtoI = zeros((3, 3))
            for pulse in range(self.CPILength):
                tempBoresightVec = getBoresightVector(
                    self.rotBtoMG, pan[pulse], tilt[pulse], yaw[pulse],
                    pitch[pulse], roll[pulse])
                self.cpiBoresightVec += tempBoresightVec
            # finalize the average computation by dividing by the CPILength
            self.cpiBoresightVec /= self.CPILength

            effGrazeI, effAzI = \
                getEffectiveInertialAzimuthAndGraze(self.cpiBoresightVec)
            self.cpiGrazeI = effGrazeI
            self.cpiAzI = effAzI

        return self.cpiBoresightVec, self.cpiGrazeI, self.cpiAzI

    def getRadarTime(self):
        if (not self.time):
            self.time = self.gpsData[0].getTime(self.CPICounter - 1)

        return self.time

    def incrementCPI(self):
        if (self.CPICounter < self.NumberOfCPIs):
            self.CPICounter += 1
        self.cpiPlatformVel = 0
        self.cpiPlatformPos = 0
        self.cpiAntennaPos = 0
        self.cpiAttitude = 0
        self.cpiBoresightVec = 0
        self.cpiAzI = 0
        self.cpiGrazeI = 0
        self.time = 0
        return self.CPICounter

    def getStanagDwellSegmentData(self, targetReportCount, dtedManager):
        """
        Returns an array (or list) with all of the expected items for a proper
        dwell segment according to the STANAG 4607 format.
        """
        # The array we return needs to be length 31, but not all of items in the 
        #   array need to be populated. The following indices within the array
        #   need to be populated accordingly:
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
        dwellData[0] = SlimSDRGMTIDataParser.existenceMask
        # Not sure yet how I'm going to to the revisit index (for now just make
        #   it 0)
        dwellData[1] = 0
        dwellData[2] = self.CPICounter
        dwellData[3] = 0
        if (self.CPICounter == self.NumberOfCPIs):
            dwellData[3] = 1
        dwellData[4] = targetReportCount
        # get the radar time in seconds since the beginning of the GPS week and
        #   convert it to seconds since the beginning of the day
        currentTime = self.getRadarTime() - self.xmlData.gpsWeekSecDiff
        dwellData[5] = currentTime * 1e3
        # let's grab the geocoordinates
        pos, vel = self.getPlatformPosVel()
        dwellData[6] = pos.item(1) / self.latConv
        dwellData[7] = pos.item(0) / self.lonConv
        dwellData[8] = pos.item(2) * 1e2
        # use the velocity information to get the track heading and speed and
        # vertical velocity
        # first the heading
        sensorHeading = arctan2(vel.item(0), vel.item(1)) * self.RTD
        if (sensorHeading < 0.0):
            sensorHeading += 360.0
        dwellData[14] = sensorHeading
        # then the ground speed
        groundSpeed = sqrt(vel.item(0) ** 2 + vel.item(1) ** 2) * 1e3
        dwellData[15] = groundSpeed
        # then the vertical velocity
        dwellData[16] = vel.item(2) * 1e1
        # Get the attitude information of the platform
        attitude = self.getPlatformAttitude()
        dwellData[20] = attitude.item(0) * self.RTD
        dwellData[21] = attitude.item(1) * self.RTD
        dwellData[22] = attitude.item(2) * self.RTD
        # Compute the dwell area. To do this, we need to get the boresight
        #   vector and then do some computations to determine where it is
        #   pointing on the ground. Technically, we should use the DTED to get
        #   that information most accurately. But, maybe an approximation would
        #   be fine so as to avoid computationally expensive DTED searches.
        bsvec, grazeI, azI = self.getCPIBoresightVector()
        sceneCen = pos + bsvec * self.midRange
        dwellData[23] = sceneCen.item(1) / self.latConv
        dwellData[24] = sceneCen.item(0) / self.lonConv
        # we need to compute the hAGL at the point of the aircraft
        hAgl = pos.item(2) \
               - dtedManager.getDTEDPoint(dwellData[6], dwellData[7])
        farGroundRange = sqrt(self.farRange ** 2 - hAgl ** 2)
        nearGroundRange = sqrt(self.nearRange ** 2 - hAgl ** 2)
        dwellData[25] = ((farGroundRange - nearGroundRange) / 2.0) / 1e3
        dwellData[26] = self.halfBeamwidth + self.CPIScanRateD / 2

        return dwellData

    def close(self):
        """Closes the data file for reading"""
        self.fid.close()
