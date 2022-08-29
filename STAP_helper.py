# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:50:17 2015

@author: Josh Bradley

@purpose: this is module with various helper functions in it
pertaining to STAP radar data simulation and analyzation
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from numpy import any as npany
from numpy import power
# from scipy.signal import chebwin, hanning, hamming
from scipy.linalg import toeplitz
from scipy.special import i0
from osgeo import gdal
# from Particle_Filtering import Particles
from scipy.ndimage import convolve
from scipy.io import loadmat
import numpy as np

DTR = pi / 180.0
c0 = 299792458.0


def window_taylor(N, nbar=4, sll=-30):
    """Taylor tapering window
    Taylor windows allows you to make tradeoffs between the
    mainlobe width and sidelobe level (sll).
    Implemented as described by Carrara, Goodman, and Majewski
    in 'Spotlight Synthetic Aperture Radar: Signal Processing Algorithms'
    Pages 512-513
    :param N: window length
    :param float nbar:
    :param float sll:
    The default values gives equal height
    sidelobes (nbar) and maximum sidelobe level (sll).
    .. warning:: not implemented
    .. seealso:: :func:`create_window`, :class:`Window`
    """
    B = 10 ** (-sll / 20)
    A = np.log(B + np.sqrt(B ** 2 - 1)) / np.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)

    def calc_Fm(m):
        numer = (-1) ** (m + 1) \
                * np.prod(1 - m ** 2 / s2 / (A ** 2 + (ma - 0.5) ** 2))
        denom = 2 * np.prod([1 - m ** 2 / j ** 2 for j in ma if j != m])
        return numer / denom

    Fm = np.array([calc_Fm(m) for m in ma])

    def W(n):
        return 2 * np.sum(
            Fm * np.cos(2 * np.pi * ma * (n - N / 2 + 1 / 2) / N)) + 1

    w = np.array([W(n) for n in range(N)])
    # normalize (Note that this is not described in the original text)
    scale = W((N - 1) / 2)
    w /= scale
    return w


class HeaderParser(object):

    def __init__(self, fid):
        self.Npulses = fromfile(fid, 'int32', 1, '').item()
        self.Nsam = fromfile(fid, 'int32', 1, '').item()
        self.RxNchan = fromfile(fid, 'int32', 1, '').item()
        self.TxNchan = fromfile(fid, 'int32', 1, '').item()
        self.RxLen = fromfile(fid, 'float64', 1, '').item()
        self.TxLen = fromfile(fid, 'float64', 1, '').item()
        self.fc = fromfile(fid, 'float64', 1, '').item()
        self.BW = fromfile(fid, 'float64', 1, '').item()
        self.freq_offset = fromfile(fid, 'float64', 1, '').item()
        self.T_r = fromfile(fid, 'float64', 1, '').item()
        self.adc_on = fromfile(fid, 'float64', 1, '').item()
        self.tp = fromfile(fid, 'float64', 1, '').item()
        self.dt = fromfile(fid, 'float64', 1, '').item()
        self.k = fromfile(fid, 'float64', 1, '').item()
        self.kr = fromfile(fid, 'float64', 1, '').item()
        self.sigma_n = fromfile(fid, 'float64', 1, '').item()
        self.w0 = fromfile(fid, 'float64', 1, '').item()

        # define the header length in bytes
        self._headerLengthBytes = dtype('int32').itemsize * 4 \
                                  + dtype('float64').itemsize * 13

    def __str__(self):
        string = "  Header Contents:\n  {\n"
        string += "    Npulses:     %d\n" % (self.Npulses)
        string += "    Nsam:        %d\n" % (self.Nsam)
        string += "    RxNchan:     %d\n" % (self.RxNchan)
        string += "    TxNchan:     %d\n" % (self.TxNchan)
        string += "    RxLen:       %0.3f m\n" % (self.RxLen)
        string += "    TxLen:       %0.3f m\n" % (self.TxLen)
        string += "    fc:          %0.3f GHz\n" % (self.fc * 1e-9)
        string += "    BW:          %0.3f MHz\n" % (self.BW * 1e-6)
        string += "    freq_offset: %0.3f MHz\n" % (self.freq_offset * 1e-6)
        string += "    T_r:         %0.3e s\n" % (self.T_r)
        string += "    adc_on:      %0.3e s\n" % (self.adc_on)
        string += "    tp:          %0.3e s\n" % (self.tp)
        string += "    dt:          %0.3e s\n" % (self.dt)
        string += "    k:           %0.3f\n" % (self.k)
        string += "    kr:          %0.3f GHz/s\n" % (self.kr * 1e-9)
        string += "    sigma_n:     %0.3f dB\n" % (20 * log10(self.sigma_n))
        string += "    w0:          %0.3f\n  }\n" % (self.w0)

        return string

    def getHeaderLengthBytes(self):
        return self._headerLengthBytes


class BinaryRadarData(object):

    def __init__(self, filename, CPILength=128):
        # record the CPILength
        self.CPILength = CPILength
        # open the file for reading binary
        self.fid = open(filename, 'rb')

        # parse the header information
        self.header = HeaderParser(self.fid)
        # after reading the header, get the current position of the file reader
        self.filePos = self.fid.tell()

        # determine the total number of CPI's for the collection
        self.NumberOfCPIs = self.header.Npulses // self.CPILength
        # initialize the CPI counter
        self.CPICounter = 0

        # assign the length of a
        self.datatype = dtype('float32')
        # calculate the expected size of one pulse
        self.singleChanPulseSize = self.header.Nsam
        # the expected size of one pulse for all channels
        self.combChanPulseSize = \
            self.header.RxNchan * self.header.TxNchan * self.singleChanPulseSize
        # the expected size of all the pulses and channels in a CPI
        self.CPIDataSize = self.CPILength * self.combChanPulseSize

    def __str__(self):
        string = "Radar Data Contents:\n{\n"
        string += self.header.__str__()
        string += "\n"
        string += "  CPI Length:     %d\n" % (self.CPILength)
        string += "  Number of CPIs  %d\n}\n" % (self.NumberOfCPIs)
        return string

    def nextCPIExists(self):
        """This function returns true if another CPI still exists."""
        return (self.CPICounter < self.NumberOfCPIs)

    def getNextCPIData(self):
        """This function simply returns all of the data for the next CPI
        if it is not beyond the last CPI. Otherwise, it returns a zero.
        It also returns the current CPI Index.
        """
        data = 0
        # check to make sure that we are not beyond our last CPI
        if self.nextCPIExists():
            data = fromfile(self.fid, self.datatype, self.CPIDataSize, '')
            # reshape the data febore returning it
            data.resize(
                (self.CPILength, self.header.RxNchan, self.header.Nsam))

            # increment the CPICounter
            self.CPICounter += 1

        return data, self.CPICounter

    def getCPIData(self, cpiIndex):
        """This functions returns the requested CPI's data if it is within the
        CPI range of the collection. Otherwise, it returns a zero.
        """
        data = 0
        if (cpiIndex >= 0 and cpiIndex < self.NumberOfCPIs):
            self.CPICounter = cpiIndex
            # seek the file to the correct position before reading the data
            seekPoint = \
                self.CPICounter \
                * (self.CPIDataSize * self.datatype.itemsize) \
                + self.header.getHeaderLengthBytes()
            self.fid.seek(seekPoint)
            data = fromfile(self.fid, self.datatype, self.CPIDataSize, '')
            # reshape the data before returning it
            data.resize(
                (self.CPILength, self.header.RxNchan, self.header.Nsam))
            # increment the CPI Counter so that it points to the next one
            self.CPICounter += 1

        return data

    def close(self):
        """Closes the data file for reading"""
        self.fid.close()


class BinaryRadarDataGeneral(object):

    def __init__(self, filename, dataType='complex64', CPILength=128):
        # record the CPILength
        self.CPILength = CPILength
        # open the file for reading binary
        self.fid = open(filename, 'rb')

        # parse the header information
        self.header = HeaderParser(self.fid)
        # after reading the header, get the current position of the file reader
        self.filePos = self.fid.tell()

        # determine the total number of CPI's for the collection
        self.NumberOfCPIs = self.header.Npulses // self.CPILength
        # initialize the CPI counter
        self.CPICounter = 0

        # assign the length of a
        self.datatype = dtype(dataType)
        # calculate the expected size of one pulse
        self.singleChanPulseSize = self.header.Nsam
        # the expected size of one pulse for all channels
        self.combChanPulseSize = \
            self.header.RxNchan * self.header.TxNchan * self.singleChanPulseSize
        # the expected size of all the pulses and channels in a CPI
        self.CPIDataSize = self.CPILength * self.combChanPulseSize

    def __str__(self):
        string = "Radar Data Contents:\n{\n"
        string += self.header.__str__()
        string += "\n"
        string += "  CPI Length:     %d\n" % (self.CPILength)
        string += "  Number of CPIs  %d\n}\n" % (self.NumberOfCPIs)
        return string

    def nextCPIExists(self):
        """This function returns true if another CPI still exists."""
        return (self.CPICounter < self.NumberOfCPIs)

    def getNextCPIData(self):
        """This function simply returns all of the data for the next CPI
        if it is not beyond the last CPI. Otherwise, it returns a zero.
        It also returns the current CPI Index.
        """
        data = 0
        # check to make sure that we are not beyond our last CPI
        if self.nextCPIExists():
            data = fromfile(self.fid, self.datatype, self.CPIDataSize, '')
            # reshape the data febore returning it
            data.resize((self.CPILength, self.header.Nsam))

            # increment the CPICounter
            self.CPICounter += 1

        return data, self.CPICounter

    def getCPIData(self, cpiIndex):
        """This functions returns the requested CPI's data if it is within the
        CPI range of the collection. Otherwise, it returns a zero.
        """
        data = 0
        if (cpiIndex >= 0 and cpiIndex < self.NumberOfCPIs):
            self.CPICounter = cpiIndex
            # seek the file to the correct position before reading the data
            seekPoint = \
                self.CPICounter \
                * (self.CPIDataSize * self.datatype.itemsize) \
                + self.header.getHeaderLengthBytes()
            self.fid.seek(seekPoint)
            data = fromfile(self.fid, self.datatype, self.CPIDataSize, '')
            # reshape the data before returning it
            data.resize(
                (self.CPILength, self.header.RxNchan, self.header.Nsam))
            # increment the CPI Counter so that it points to the next one
            self.CPICounter += 1

        return data

    def close(self):
        """Closes the data file for reading"""
        self.fid.close()


class MoverPositionData(object):
    # define the number of data points per target
    dataPointsPerTarget = 5
    numRadarDataPoints = 9

    # define the data positions of the data for the radar
    timeIndex = 0
    antEastIndex = 1
    antNorthIndex = 2
    antAltIndex = 3
    yawIndex = 4
    pitchIndex = 5
    rollIndex = 6
    gimbalPanIndex = 7
    gimbalTiltIndex = 8

    # define the data positions of the data for the movers
    tarRangeIndex = 0
    tarAntGainIndex = 1
    tarEastIndex = 2
    tarNorthIndex = 3
    tarAltIndex = 4

    existenceMask = uint64(0xFF071FC39E000000)

    def __init__(self, filename, txtFilename, CPILength, NumCPIs, T_r,
                 gimbalRollOffset, gimbalPitchOffset, gimbalYawOffset):
        self.RTD = 180.0 / pi

        # calculate the gimbal rotation offset matrix
        self.rotationOffsetMat = getRotationOffsetMatrix(gimbalRollOffset, gimbalPitchOffset, gimbalYawOffset)

        # read in the RCS values from the movers.txt file
        with open(txtFilename) as f:
            tmp = f.readlines()
        f.close()
        self.rcs_data = zeros(len(tmp))
        for idx, i in enumerate(tmp):
            ttt = i.split(',')
            self.rcs_data[idx] = float(ttt[-2])

        fid = open(filename, 'rb')
        # record the pulse repetition interval
        self.T_r = T_r

        # parse out the header information
        self.numMovers = int(fromfile(fid, 'float64', 1, '').item())
        self.lonConv = fromfile(fid, 'float64', 1, '').item()
        self.latConv = fromfile(fid, 'float64', 1, '').item()

        # parse the rest of the data out of the file
        self.data = fromfile(fid, 'float64', -1, '')
        fid.close()
        # reshape the data
        self.data = self.data.reshape((-1, self.numMovers * MoverPositionData.dataPointsPerTarget \
                                       + MoverPositionData.numRadarDataPoints), order='C')

        # check to make sure that the number of CPI's matches
        self.CPILength = CPILength
        self.CPITime = (self.CPILength * self.T_r)
        self.NPulses = self.data.shape[0]
        self.NumberOfCPIs = self.NPulses / self.CPILength
        if (self.NumberOfCPIs != NumCPIs):
            print("The number of CPI's present in the mover position data differs from the raw data.")
            print("Mover Position file number of CPI's = %d" % (self.NumberOfCPIs))
        else:
            print("CPI numbers in MoverPositionData matches that for pulse data.")
        # initialize the CPI counter
        self.CPICounter = 0
        self.startPulse = 0
        self.stopPulse = self.startPulse + self.CPILength

        # initialize the position, velocity and boresight vector of the plane for
        # the CPI
        self.cpiPos = 0
        self.cpiVel = 0
        self.cpiBoresightVec = 0
        self.cpiAttitude = 0
        self.cpiGrazeI = 0
        self.cpiAzI = 0

    def __str__(self):
        string = "Mover Position Content:\n{\n"
        string += "  NumMovers:      %d\n" % (self.numMovers)
        string += "  latConv:        %0.6f\n" % (self.latConv)
        string += "  lonConv:        %0.6f\n" % (self.lonConv)
        string += "  CPI Length:     %d\n" % (self.CPILength)
        string += "  Number of CPIs  %d\n}\n" % (self.NumberOfCPIs)
        return string

    def setRadarParameters(self, nearRange, farRange, grazAngle, azimuthBeamwidth, lamda):
        # record the ranges
        self.nearRange = nearRange
        self.farRange = farRange
        self.midRange = (nearRange + farRange) / 2.0
        self.grazAngle = grazAngle * self.RTD
        self.lamda = lamda

        # record the half beamwidth
        self.halfBeamwidth = (azimuthBeamwidth / 2.0) * self.RTD

        # compute the CPI time
        self.scanRate = 6.0  # deg/s
        self.CPIScanRate = self.scanRate * self.CPITime

    def setCPI(self, cpiNum):
        """Sets the CPI counter to point to the correct CPI and pulse for
        indexing into the data.  This must be performed before attempting to
        extract data to avoid erroneous output."""
        self.CPICounter = cpiNum
        self.startPulse = self.CPICounter * self.CPILength
        self.stopPulse = self.startPulse + self.CPILength
        # reset all of the CPI data
        self.cpiPos = 0
        self.cpiVel = 0
        self.cpiBoresightVec = 0
        self.cpiAttitude = 0
        self.cpiGrazeI = 0
        self.cpiAzI = 0

    def getAntennaPosVel(self):
        """Returns the average easting, northing, and altitude of the aircraft
        for the pulses of the designated CPI. 
        return - 3x1 numpy array of easting, northing, and altitude in meters
        and 3x1 numpy array of easting, northing, and altitude velocity in meters/sec"""
        if (not npany(self.cpiPos) and not npany(self.cpiVel)):
            self.cpiPos = self.data[self.startPulse: self.stopPulse,
                          MoverPositionData.antEastIndex: MoverPositionData.antAltIndex + 1]
            self.cpiVel = mean((self.cpiPos[1:, :] - self.cpiPos[:-1, :]) / self.T_r, axis=0).reshape((3, 1))
            self.cpiPos = mean(self.cpiPos, axis=0).reshape((3, 1), order='C')

        return self.cpiPos, self.cpiVel

    def getPlatformAttitude(self):
        """Returns the average azimuth, pitch, and roll of the aircraft for the
        pulses of the designated CPI.
        return - 3x1 numpy array of azimuth, pitch, roll in degrees"""
        if (not npany(self.cpiAttitude)):
            self.cpiAttitude = self.data[self.startPulse: self.stopPulse,
                               MoverPositionData.yawIndex: MoverPositionData.rollIndex + 1]
            self.cpiAttitude = mean(self.cpiAttitude, axis=0).reshape((3, 1), order='C')

        return self.cpiAttitude

    def getBoresightVector(self):
        """Return the average inertial boresight pointing vector of the antenna
        for the pulses of the designated CPI.
        return - 3x1 numpy array with the normalized boresight vector (easting, northing, altitude)"""
        if (not npany(self.cpiBoresightVec) and not npany(self.cpiGrazeI) and not npany(self.cpiAzI)):
            attitude = self.data[self.startPulse: self.stopPulse,
                       MoverPositionData.yawIndex: MoverPositionData.rollIndex + 1]
            gimbalRotation = self.data[self.startPulse: self.stopPulse,
                             MoverPositionData.gimbalPanIndex: MoverPositionData.gimbalTiltIndex + 1]
            self.cpiBoresightVec = array([[0.0], [0.0], [0.0]])
            for pulse in range(self.CPILength):
                self.cpiBoresightVec += getBoresightVector(self.rotationOffsetMat,
                                                           gimbalRotation[pulse, 0], gimbalRotation[pulse, 1],
                                                           attitude[pulse, 0], attitude[pulse, 1], attitude[pulse, 2])
            # finalize the average computation by dividing by the CPILength
            self.cpiBoresightVec /= self.CPILength

            effGrazeI, effAzI = getEffectiveInertialAzimuthAndGraze(self.cpiBoresightVec)
            self.cpiGrazeI = effGrazeI
            self.cpiAzI = effAzI

        return self.cpiBoresightVec, self.cpiGrazeI, self.cpiAzI

    def getTargetPosVel(self, tarNum):
        """Returns the average easting, northing, and elevation of the target
        designated for the pulses of the designated CPI.
        The return value is a 3x1 numpy array."""
        # calculate the index into the data for the target
        targetDataStartIndex = MoverPositionData.numRadarDataPoints \
                               + MoverPositionData.dataPointsPerTarget * tarNum \
                               + MoverPositionData.tarEastIndex
        targetDataStopIndex = MoverPositionData.numRadarDataPoints \
                              + MoverPositionData.dataPointsPerTarget * tarNum \
                              + MoverPositionData.tarAltIndex

        tarPos = self.data[self.startPulse: self.stopPulse,
                 targetDataStartIndex: targetDataStopIndex + 1]
        tarVel = mean((tarPos[1:, :] - tarPos[:-1, :]) / self.T_r, axis=0)
        tarPos = mean(tarPos, axis=0)
        # reshape the output to a column vector
        return tarPos.reshape((3, 1), order='C'), tarVel.reshape((3, 1), order='C')

    def getTargetGainCombo(self, tarNum):
        """Returns the average antenna gain target during the current CPI for
        the designated target."""
        tarAntGain = self.data[self.startPulse: self.stopPulse,
                     MoverPositionData.numRadarDataPoints + MoverPositionData.dataPointsPerTarget * tarNum \
                     + MoverPositionData.tarAntGainIndex].mean()

        return 10 * log10(tarAntGain * tarAntGain * self.rcs_data[tarNum])

    def getTargetRange(self, tarNum):
        """Returns the average target range during the current CPI for the 
        designated target."""
        tarRange = self.data[self.startPulse: self.stopPulse,
                   MoverPositionData.numRadarDataPoints + MoverPositionData.dataPointsPerTarget * tarNum \
                   + MoverPositionData.tarRangeIndex].mean()

        return tarRange

    def computeTargetError(self, tarNum, posHat, radVelHat, rangeHat):
        """For the specified target number compute the errors between the actual
        position and velocities and the estimates."""
        rangeError = 0.0
        groundRangeError = 0.0
        crossRangeError = 0.0
        heightError = 0.0
        azimuthError = 0.0
        distError = 0.0
        radVelError = 0.0

        tpp, tvv = self.getTargetPosVel(tarNum)
        truthRange = self.getTargetRange(tarNum)

        # The errors need to be projected into a local tangent reference frame
        #   centered about the actual target location, with y being in the
        #   ground range direction, z being aligned opposite of gravity, and
        #   x being in the direction 90 deg left-handed from y rotated about z
        # In order to determine this we need to compute the range vector from
        #   the antenna phase center to the target position and then compute the
        #   azimuth angle and then rotate the error vector by that value.
        antPos, antVel = self.getAntennaPosVel()
        truthRangeVec = tpp - antPos
        # Turn the truthRangeVec into a unit pointing vector
        truthRangeVec /= sqrt(truthRangeVec.T.dot(truthRangeVec)).item(0)
        truthAzR = arctan2(truthRangeVec.item(0), truthRangeVec.item(1))
        cTA = cos(truthAzR)
        sTA = sin(truthAzR)
        R_i_tar = array([[cTA, -sTA, 0],
                         [sTA, cTA, 0],
                         [0, 0, 1]])
        # Compute the position error in the target local frame
        posError = R_i_tar.dot(posHat - tpp)
        # Calculate the absolute distance error (or separation error)
        distError = sqrt(posError.T.dot(posError)).item(0)
        # Calculate the actual radial velocity
        truthRadVel = -truthRangeVec.T.dot(tvv).item(0)
        # Compute the radial velocity error
        radVelError = truthRadVel - radVelHat
        # compute the ranging error
        rangeError = truthRange - rangeHat
        # Save the ground and cross range error and height error
        groundRangeError = posError.item(1)
        crossRangeError = posError.item(0)
        heightError = posError.item(2)

        # Finally compute the azimuth error using the dot product cosine rule
        estRangeVec = posHat - antPos
        estRangeVec /= sqrt(estRangeVec.T.dot(estRangeVec)).item(0)
        estV2 = estRangeVec[0:2, :] / sqrt(
            estRangeVec[0:2, :].T.dot(estRangeVec[0:2, :])).item(0)
        truthV2 = truthRangeVec[0:2, :] / sqrt(
            truthRangeVec[0:2, :].T.dot(truthRangeVec[0:2, :])).item(0)
        azimuthError = arccos(truthV2.T.dot(estV2).item(0))
        # modify the azimuth Error by the sign of the cross-product
        if (estV2[0, 0] * truthV2[1, 0] - estV2[1, 0] * truthV2[0, 0] < 0):
            azimuthError *= -1

        return groundRangeError, crossRangeError, heightError, rangeError, \
               azimuthError, distError, radVelError, tpp

    def getStanagDwellSegmentData(self, targetReportCount):
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
        dwellData[2] = self.CPICounter
        dwellData[3] = 0
        if (self.CPICounter == self.NumberOfCPIs):
            dwellData[3] = 1
        dwellData[4] = targetReportCount
        dwellData[5] = self.CPITime * 1e3
        # let's grab the geocoordinates
        pos, vel = self.getAntennaPosVel()
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
        # Compute the dwell area. To do this, we need to get the boresight vector
        # and then do some computations to determine where it is pointing on the
        # ground. Technically, we should use the DTED to get that information most
        # accurately. But, maybe an approximation would be fine so as to avoid
        # computationally expensive DTED searches.
        bsvec = self.getBoresightVector()
        sceneCen = pos + bsvec * self.midRange
        dwellData[23] = sceneCen.item(1) / self.latConv
        dwellData[24] = sceneCen.item(0) / self.lonConv
        # we need to compute the hAGL at the point of the aircraft
        dtedName = getDTEDName(dwellData[6], dwellData[7])
        dtedCorrection = getDTEDCorrection(dwellData[6], dwellData[7])
        hAgl = pos.item(2) - getDTEDPoint(dwellData[6], dwellData[7], dtedName, dtedCorrection)
        farGroundRange = sqrt(self.farRange ** 2 - hAgl ** 2)
        nearGroundRange = sqrt(self.nearRange ** 2 - hAgl ** 2)
        dwellData[25] = ((farGroundRange - nearGroundRange) / 2.0) / 1e3
        dwellData[26] = self.halfBeamwidth + self.CPIScanRate / 2

        return dwellData


def computeGrazingAngle(
        effAzIR, grazeIR, antPos, theRange, lonConv, latConv, dtedManager):
    # initialize the pointing vector to first range bin
    Rvec = array([[cos(grazeIR) * sin(effAzIR)],
                  [cos(grazeIR) * cos(effAzIR)],
                  [-sin(grazeIR)]])

    groundPoint = antPos + Rvec * theRange

    nlat = groundPoint[1, 0] / latConv
    nlon = groundPoint[0, 0] / lonConv
    # look up the height of the surface below the aircraft
    surfaceHeight = dtedManager.getDTEDPoint(nlat, nlon)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - groundPoint[2, 0]

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while (abs(elevDiff) > heightDiffThresh and numIterations < iterationThresh):
        hAgl = antPos[2, 0] - surfaceHeight
        newGrazeR = arcsin(hAgl / theRange)
        if (isnan(newGrazeR) or isinf(newGrazeR)):
            print('hAgl: %0.3f, theRange: %0.3f' % (hAgl, theRange))
        Rvec = array([[cos(newGrazeR) * sin(effAzIR)],
                      [cos(newGrazeR) * cos(effAzIR)],
                      [-sin(newGrazeR)]])
        groundPoint = antPos + Rvec * theRange
        nlat = groundPoint[1, 0] / latConv
        nlon = groundPoint[0, 0] / lonConv
        surfaceHeight = dtedManager.getDTEDPoint(nlat, nlon)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - groundPoint[2, 0]
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations


def getDopplerLine(
        effAzI, rangeBins, antVel, antPos, radar, dtedManager, lamda,
        azBeamwidthHalf, PRF, lonConv=0, latConv=0):
    """Compute the expected Doppler vs range for the given platform geometry"""
    if latConv == 0 and lonConv == 0:
        lonConv = radar.lonConv
        latConv = radar.latConv

    # Get the grazing angle in the near range
    nearRangeGrazeR = radar.nearRangeD * DTR

    # compute the grazing angle for the near range to start
    (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
        effAzI, nearRangeGrazeR, antPos, rangeBins[0], lonConv, latConv,
        dtedManager)
    #    print( 'surfaceHeight: %f' % surfaceHeight )
    #    print( "nearRangeGrazeR: %f" % nearRangeGrazeR )

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = arcsin((antPos[2, 0] - surfaceHeight) / rangeBins)

    # this is a special version of Rvec (it is not 3x1, it is 3xNrv)
    Rvec = array([
        cos(grazeOverRanges) * sin(effAzI),
        cos(grazeOverRanges) * cos(effAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Doppler
    DopplerCen = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerCen > radar.PRF / 2)
    DopplerCen[ind] = DopplerCen[ind] - radar.PRF
    ind = nonzero(DopplerCen < -radar.PRF / 2)
    DopplerCen[ind] = DopplerCen[ind] + radar.PRF

    # generate the radial vector for the forward beamwidth edge 
    # (NOTE!!!: this is dependent
    # on the antenna pointing vector attitude with respect to the aircraft heading.
    # if on the left side, negative azimuth will be lower Doppler, and positive
    # azimuth will be higher, but on the right side, it will be the opposite, one
    # could use the sign of the cross-product to determine which it is.)
    if (radar.xmlData.gimbalSettings.lookSide.lower() == 'left'):
        azBeamwidthHalf *= -1.0

    newAzI = effAzI - azBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerUp = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerUp > radar.PRF / 2)
    DopplerUp[ind] = DopplerUp[ind] - radar.PRF
    ind = nonzero(DopplerUp < -radar.PRF / 2)
    DopplerUp[ind] = DopplerUp[ind] + radar.PRF

    # generate the radial vector for the forward beamwidth edge
    newAzI = effAzI + azBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerDown = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerDown > radar.PRF / 2)
    DopplerDown[ind] = DopplerDown[ind] - radar.PRF
    ind = nonzero(DopplerDown < -radar.PRF / 2)
    DopplerDown[ind] = DopplerDown[ind] + radar.PRF
    return DopplerCen, DopplerUp, DopplerDown, grazeOverRanges


def getDopplerLineHiFi(
        effAzI, rangeBins, antVel, antPos, radar, dtedManager, lamda,
        azNullBeamwidthHalf, PRF, lonConv=0, latConv=0):
    """Compute the expected Doppler vs range for the given platform geometry"""
    # record the lat and lon conversion factors
    if (lonConv == 0 and latConv == 0):
        latConv = radar.latConv
        lonConv = radar.lonConv

    # Get the grazing angle in the near range
    nearRangeGrazeR = radar.nearRangeD * DTR

    # compute the grazing angle for the near range to start
    (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
        effAzI, nearRangeGrazeR, antPos, rangeBins[0], lonConv, latConv,
        dtedManager)

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = zeros(len(rangeBins))
    grazeOverRanges[0] = nearRangeGrazeR
    grazeAngleStep = \
        radar.xmlData.CCS.antSettings[0].elBeamwidthD * DTR / len(rangeBins)
    surfaceHeights = zeros(len(rangeBins))
    surfaceHeights[0] = surfaceHeight
    numIterList = zeros(len(rangeBins))
    numIterList[0] = numIter
    for i in range(1, len(rangeBins)):
        grazeIR = grazeOverRanges[i - 1] - grazeAngleStep
        (grazeIR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
            effAzI, grazeIR, antPos, rangeBins[i], lonConv, latConv,
            dtedManager)
        grazeOverRanges[i] = grazeIR
        surfaceHeights[i] = surfaceHeight
        numIterList[i] = numIter

    # this is a special version of Rvec (it is not 3x1, it is 3xNrv)
    Rvec = array([
        cos(grazeOverRanges) * sin(effAzI),
        cos(grazeOverRanges) * cos(effAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Doppler
    DopplerCen = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerCen > radar.PRF / 2)
    DopplerCen[ind] = DopplerCen[ind] - radar.PRF
    ind = nonzero(DopplerCen < -radar.PRF / 2)
    DopplerCen[ind] = DopplerCen[ind] + radar.PRF

    # generate the radial vector for the forward beamwidth edge 
    # (NOTE!!!: this is dependent
    # on the antenna pointing vector attitude with respect to the aircraft heading.
    # if on the left side, negative azimuth will be lower Doppler, and positive
    # azimuth will be higher, but on the right side, it will be the opposite, one
    # could use the sign of the cross-product to determine which it is.)
    if (radar.xmlData.gimbalSettings.lookSide.lower() == 'left'):
        azNullBeamwidthHalf *= -1.0

    newAzI = effAzI - azNullBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerUp = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerUp > radar.PRF / 2)
    DopplerUp[ind] = DopplerUp[ind] - radar.PRF
    ind = nonzero(DopplerUp < -radar.PRF / 2)
    DopplerUp[ind] = DopplerUp[ind] + radar.PRF

    # generate the radial vector for the forward beamwidth edge
    newAzI = effAzI + azNullBeamwidthHalf
    Rvec = array([
        cos(grazeOverRanges) * sin(newAzI),
        cos(grazeOverRanges) * cos(newAzI),
        -sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerDown = \
        ((2.0 / radar.lamda) * Rvec.T.dot(antVel).flatten()) % radar.PRF
    # account for wrapping of the Doppler spectrum
    ind = nonzero(DopplerDown > radar.PRF / 2)
    DopplerDown[ind] = DopplerDown[ind] - radar.PRF
    ind = nonzero(DopplerDown < -radar.PRF / 2)
    DopplerDown[ind] = DopplerDown[ind] + radar.PRF

    return DopplerCen, DopplerUp, DopplerDown, grazeOverRanges, \
           surfaceHeights, numIterList


def getEffectiveInertialAzimuthAndGraze(boreSightVec):
    """Compute the effective inertial azimuth and grazing angle for the antenna
        pointing"""
    effGrazeI = arcsin(-boreSightVec.item(2))
    effAzI = arctan2(boreSightVec.item(0), boreSightVec.item(1))
    return effGrazeI, effAzI


def bodyToInertial(yaw, pitch, roll, x, y, z):
    cy = cos(yaw)
    sy = sin(yaw)
    cp = cos(pitch)
    sp = sin(pitch)
    cr = cos(roll)
    sr = sin(roll)

    # compute the inertial to body rotation matrix
    rotItoB = array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])
    newXYZ = rotItoB.T.dot(array([[x], [y], [z]]))

    return newXYZ


def gimbalToBody(rotBtoMG, pan, tilt, x, y, z):
    cp = cos(pan)
    sp = sin(pan)
    ct = cos(tilt)
    st = sin(tilt)

    rotMGtoGP = array([
        [cp, -sp, 0],
        [sp * ct, cp * ct, st],
        [-sp * st, -cp * st, ct]])

    # compute the gimbal mounted to gimbal pointing rotation matrix
    rotBtoGP = rotMGtoGP.dot(rotBtoMG)
    newXYZ = rotBtoGP.T.dot(array([[x], [y], [z]]))

    return newXYZ


def getRotationOffsetMatrix(roll0, pitch0, yaw0):
    cps0 = cos(yaw0)
    sps0 = sin(yaw0)
    cph0 = cos(roll0)
    sph0 = sin(roll0)
    cth0 = cos(pitch0)
    sth0 = sin(pitch0)

    Delta1 = cph0 * cps0 + sph0 * sth0 * sps0
    Delta2 = -cph0 * sps0 + sph0 * sth0 * cps0
    Delta3 = -sph0 * cth0
    Delta4 = cth0 * sps0
    Delta5 = cth0 * cps0
    Delta6 = sth0
    Delta7 = sph0 * cps0 - cph0 * sth0 * sps0
    Delta8 = -sph0 * sps0 - cph0 * sth0 * cps0
    Delta9 = cph0 * cth0

    ROffset = array([
        [-Delta1, -Delta2, -Delta3],
        [Delta4, Delta5, Delta6],
        [-Delta7, -Delta8, -Delta9]])

    return ROffset


def getBoresightVector(ROffset, alpha_az, alpha_el, yaw, pitch, roll):
    """Returns the a 3x1 numpy array with the normalized boresight vector in 
    the inertial frame"""
    # set the boresight pointing vector in the pointed gimbal frame
    delta_gp = array([[0], [0], [1.0]])
    # rotate these into the body frame
    delta_b = gimbalToBody(
        ROffset, alpha_az, alpha_el, delta_gp.item(0), delta_gp.item(1),
        delta_gp.item(2))
    # finish the rotation into the inertial frame
    delta_i = bodyToInertial(
        yaw, pitch, roll, delta_b.item(0), delta_b.item(1),
        delta_b.item(2))
    # return the boresight in the inertial frame
    return delta_i


def computePhaseDifference(rcd1, rcd2, struct):
    # struct = zeros((winLen, winLen))
    # struct[winLen/2,:] = ones(winLen)
    # struct = ones((winLen,winLen))
    gamma = rcd1 * rcd2.conj()
    realg = convolve(gamma.real, struct, mode='constant')
    imagg = convolve(gamma.imag, struct, mode='constant')
    gamma = realg + 1J * imagg
    i1 = sqrt(convolve(rcd1.real ** 2 + rcd1.imag ** 2, struct, mode='constant'))
    i2 = sqrt(convolve(rcd2.real ** 2 + rcd2.imag ** 2, struct, mode='constant'))
    gamma = gamma / (i1 * i2)
    # gamma = gamma.mean(axis=0)
    return gamma


def parseMovPosData(filename):
    fid = open(filename, 'rb')
    # parse the header information first
    movParams = {}
    movParams['numMovers'] = int(fromfile(fid, 'float64', 1, '').item())
    movParams['lonConv'] = fromfile(fid, 'float64', 1, '').item()
    movParams['latConv'] = fromfile(fid, 'float64', 1, '').item()
    movdata = fromfile(fid, 'float64', -1, '')
    movdata = movdata.reshape((-1, movParams['numMovers'] * 5 + 4), order='C')
    fid.close()
    return movParams, movdata


def computeLatLonConv(latitudeD):
    """Returns the values for converting from latitude and longitude to meters
    based on a reference latitude
    Inputs:
        latitudeD - the reference latitude in degrees
    Outputs:
        latConv - the value for converting from latitude to northings (meters)
        lonConv - the value for converting from longitude to eastings (meters)"""
    sinReferenceLatitude = sin(latitudeD * DTR)
    w84 = sqrt(1.0 - 0.00669437999014131699613723354004 * sinReferenceLatitude \
               * sinReferenceLatitude)
    # Calculate the latitude conversion factor
    latConv = 110574.27582159436148033428703626 / (w84 * w84 * w84)
    # Calculate the longitude conversion factor
    lonConv = 111319.49079327357264771338267056 * cos(latitudeD * DTR) / w84

    return latConv, lonConv


def associateDet2Targets(tPosHat, tRangeHat, tRadVelHat, antPos, mp,
                         gainComboThreshold, MPP, usePosError=True):
    """This function associates a "detected mover" to a target in the truth
    data list and returns the target number to which it corresponds."""

    # determine which target this is
    tarIndex = -1
    minRangeIndex = -1
    minPosIndex = -1
    falseAlarm = False
    tRangeErrorMin = 1e20
    tRadVelErrorMin = 1e20
    tPosErrorMin = 1e20
    errorNormMin = 1e20
    errorNormIndex = -1
    posThreshold = 500.0
    for j in range(mp.numMovers):
        # get all the information for the targets in the list for this CPI
        tGainCombo = mp.getTargetGainCombo(j)
        tpp, tvv = mp.getTargetPosVel(j)
        tRange = mp.getTargetRange(j)

        # now calculate the errors
        # location
        pdiff = tpp - tPosHat
        tPosError = sqrt(pdiff.T.dot(pdiff)).item(0)
        # radial velocity
        r_i = tpp - antPos
        r_i = r_i / sqrt(r_i.T.dot(r_i)).item(0)
        tRadVel = -r_i.T.dot(tvv).item(0)
        tRadVelError = abs(tRadVel - tRadVelHat)

        tRangeError = tRange - tRangeHat
        errorVec = array([tRadVelError, tRangeError])
        if (usePosError):
            posThreshold = 50.0
            errorVec = array([tRadVelError, tRangeError, tPosError])
        errorNorm = sqrt(errorVec.dot(errorVec).item(0))

        if (tRadVelError < tRadVelErrorMin and tGainCombo > gainComboThreshold):
            tRadVelErrorMin = tRadVelError

        if (errorNorm < errorNormMin and tGainCombo > gainComboThreshold):
            errorNormMin = errorNorm
            errorNormIndex = j

        if (tPosError < tPosErrorMin and tGainCombo > gainComboThreshold):
            tPosErrorMin = tPosError
            minPosIndex = j

        if (abs(tRangeError) < tRangeErrorMin and tGainCombo > gainComboThreshold):
            tRangeErrorMin = abs(tRangeError)
            minRangeIndex = j

    # if (tRangeErrorMin > MPP/2 or tPosErrorMin > 200.0 ):# or abs(tRadVelError) > 4.0):
    #    falseAlarm = True
    # else:
    # look up the target range error for the index that got the minimum pos error
    # and see if it is smaller than the MPP
    tRange = mp.getTargetRange(errorNormIndex)
    tRangeError = tRange - tRangeHat
    tpp, tvv = mp.getTargetPosVel(errorNormIndex)
    pdiff = tpp - tPosHat
    tPosError = sqrt(pdiff.T.dot(pdiff)).item(0)
    r_i = tpp - antPos
    r_i = r_i / sqrt(r_i.T.dot(r_i)).item(0)
    tRadVel = -r_i.T.dot(tvv).item(0)
    tRadVelError = tRadVel - tRadVelHat

    # tRangeError = mp.getTargetRange(minRangeIndex) - tRangeHat
    if (abs(tRangeError) < MPP / 2 and tPosError < posThreshold and abs(tRadVelError) < 4.0):
        # print('errorNormIndex: %d, Range Error: %0.5f m' \
        #     % (errorNormIndex, tRangeError) )
        tarIndex = errorNormIndex
    else:
        falseAlarm = True

    #    if( tarIndex == 10 ):
    #        print("tRangeError:%0.3f, tRadVelError:%0.3f, tPosError:%0.3f" % (
    #                tRangeError, tRadVelError, tPosError))
    # if (errorNormIndex != minRangeIndex):
    #    print("minNormIndex=%d, minPosIndex=%d, minRangeIndex=%d" % (errorNormIndex, minPosIndex, minRangeIndex))
    return falseAlarm, tarIndex


def getMoverParameters(binsData, radar, movPos, hAgl, airPos_i, airVel_i, R_i_A,
                       boreSightAz, nThetaRes, nDopRes, dtedManager, RIntFactor,
                       lonConv=0, latConv=0):
    """Returns all the relevant mover parameters such as target range, radial
    velocity, latitude, longitude, elevation
    Inputs:
        binsData - 1D numpy of length 3. Contains range bin, Doppler bin, angular freuency bin
        radar - a BinaryRadarData object
        hAgl - the intended height above ground level of the plane"""
    # snag the latitude and longitude conversion from the movPos header
    if (lonConv == 0 and latConv == 0):
        lonConv = movPos.lonConv
        latConv = movPos.latConv
    # make initial estimate of the plane's height above the target
    hRef = hAgl
    # calculate the wavelength
    lamda = c0 / radar.header.fc

    # calculate the range for the first bin, corresponding to adc_on time
    nearRange = radar.header.adc_on * c0 / 2.0
    # calculate the meters-per-sample in range
    MPP = c0 * radar.header.dt / RIntFactor
    # range corresponding to the range bin for the mover
    tRange = (nearRange * 2.0 + binsData[0] * MPP) / 2.0
    # compute the target's normalized spatial frequency
    # this has to be negated for the right side of the aircraft to match up with 
    # the left-handed rotation
    tNTheta = -(binsData[2] * nThetaRes - 0.5)
    # and the target's normalized Doppler frequency
    tNDop = binsData[1] * nDopRes
    tNDop = -(floor(tNDop / 0.5) - tNDop)
    # compensate for a lack of fftshift in the data
    if (tNDop > 0.5):
        tNDop -= 1.0
    # make the initial estimate of the target depression angle
    tDepI = arcsin(hRef / tRange)
    # make the initial estimate of the target inertial azimuth angle
    tAzI = boreSightAz
    # formulate our initial estimate of the inertial target radial vector
    r_hatI = array([[cos(tDepI) * sin(tAzI)],
                    [cos(tDepI) * cos(tAzI)],
                    [-sin(tDepI)]])

    # get initial estimate of the target position
    tarPos_i = airPos_i + tRange * r_hatI
    # grab the elevation for the target position
    tarAlt = dtedManager.getDTEDPoint(
        tarPos_i[1, 0] / latConv, tarPos_i[0, 0] / lonConv)
    tarPos_i[2, 0] = tarAlt

    # set the iteration limit
    iterLimit = 10
    # initialize the altitude error
    altError = 1e3
    # rangeError = 1e3
    iterCount = 0
    while (abs(altError) > 0.01 and iterCount < iterLimit):
        """Step 1 - look-up the DTED value for the target position estimate"""
        # update the reference height
        hRef = airPos_i[2, 0] - tarAlt

        """Step 2 - recompute the estimate of the target inertial depression angle
        and update the target inertial radial vector"""
        tDepI = arcsin(hRef / tRange)
        r_hatI = array([[cos(tDepI) * sin(tAzI)],
                        [cos(tDepI) * cos(tAzI)],
                        [-sin(tDepI)]])

        """Step 3 - Use the inertial-2-antenna frame rotation matrix to transform
        the target inertial radial vector in the antenna frame and get an estimate
        for the target antenna elevation angle"""
        r_hatA = R_i_A.dot(r_hatI)
        tElA = arcsin(r_hatA[2, 0])

        """Step 4 - Use the estimate for target antenna elevation angle to compute
        an estimate of the target antenna azimuth angle"""
        tAzA = arcsin(tNTheta * lamda / (radar.header.RxLen * cos(tElA)))

        """Step 5 - Update the target radial vector in the antenna frame"""
        r_hatA = array([[cos(tElA) * sin(tAzA)],
                        [cos(tElA) * cos(tAzA)],
                        [sin(tElA)]])
        """Step 6 - Transform the target radial vector back into the inertial frame
        to update the estimate of the target inertial radial vector, and then update
        the target position"""
        r_hatI = R_i_A.T.dot(r_hatA)
        # update the estimate of the target inertial azimuth angle
        tAzI = arctan2(r_hatI[0, 0], r_hatI[1, 0])
        # update the target position
        tarPos_i = airPos_i + tRange * r_hatI

        """Step 7 - Calculate the altitude error to test for need to iterate again"""
        # now that we have an updated target position, we need to look up the target
        # altitude for this new lat/lon and then check the error
        tarAlt = dtedManager.getDTEDPoint(
            tarPos_i[1, 0] / latConv, tarPos_i[0, 0] / lonConv)
        # determine the update altitude error
        altError = tarPos_i[2, 0] - tarAlt
        # print "After iteration %d alt-error:%0.5f, range-error:%0.5f" % (iterCount+1, altError, rangeError)
        # update the iteration count
        iterCount += 1

    if (iterCount > iterLimit):
        print("For this detection, we reached the iteration limit.")

    """ Radial Velocity Estimation """
    # Now that we have finished estimation of the target position, we can use the 
    # target inertial radial vector and the aircraft inertial velocity vector to 
    # estimate the target radial velocity

    # finalize the inertial radial vector based on the latest dted look-up
    # update the reference height
    hRef = airPos_i[2, 0] - tarAlt

    # recompute the estimate of the target inertial depression angle
    # and update the target inertial radial vector
    tDepI = arcsin(hRef / tRange)
    r_hatI = array([[cos(tDepI) * sin(tAzI)],
                    [cos(tDepI) * cos(tAzI)],
                    [-sin(tDepI)]])

    r_hatA = R_i_A.dot(r_hatI)
    tElA = arcsin(r_hatA[2, 0])
    tAzA = arcsin(tNTheta * lamda / (radar.header.RxLen * cos(tElA)))
    r_hatA = r_hatA = array([[cos(tElA) * sin(tAzA)],
                             [cos(tElA) * cos(tAzA)],
                             [sin(tElA)]])
    r_hatI = R_i_A.T.dot(r_hatA)
    tarPos_i = airPos_i + tRange * r_hatI

    # compute Doppler frequncy induced by motion of the antenna during CPI
    f_s = -(2.0 / lamda) * (r_hatI.T.dot(airVel_i)).item(0)
    # f_s = 0.0
    # estimate the total received Doppler frequency from the target
    f_r = -tNDop / radar.header.T_r
    # estimate the radial velocity based on the estimate Doppler cause by motion
    # of the target
    tRadVel = (f_r - f_s) * lamda / 2.0

    return tarPos_i, tRange, tRadVel, tElA * 180 / pi, tAzA * 180 / pi


def getDTEDCorrection(lat, lon):
    """Returns the ellipsoid correction for the DTED values so that it
    corresponds to the geocoordinates in the GPS."""
    # the filename for the correction data
    filename = "D:/Data/DTED/EGM96.DAT"
    fid = open(filename, 'rb')
    egm96Data = fromfile(fid, 'float64', -1, '')
    egm96Data = egm96Data.reshape((721, 1441), order='C')
    fid.close()

    # we want to be able to perform a bilinear interpolation for the data we
    # get out, so we will get all of the points surrounding our lat/lon point
    egN = ceil(lat / 0.25) * 0.25
    egS = floor(lat / 0.25) * 0.25
    egE = ceil(lon / 0.25) * 0.25
    egW = floor(lon / 0.25) * 0.25
    egNI = int((egN + 90.0 + 0.25) / 0.25) - 1
    egSI = int((egS + 90.0 + 0.25) / 0.25) - 1
    egEI = int((egE + 180.0 + 0.25) / 0.25) - 1
    egWI = int((egW + 180.0 + 0.25) / 0.25) - 1
    sepInv = 1.0 / ((egE - egW) * (egN - egS))

    # grab the four data
    eg01 = egm96Data[egNI, egWI]
    eg02 = egm96Data[egSI, egWI]
    eg03 = egm96Data[egNI, egEI]
    eg04 = egm96Data[egSI, egEI]

    egc = sepInv * (eg02 * (egE - lon) * (egN - lat) \
                    + eg04 * (lon - egW) * (egN - lat) \
                    + eg01 * (egE - lon) * (lat - egS) \
                    + eg03 * (lon - egW) * (lat - egS))

    return egc


def getInterpolatedDTEDGrid(eastings, northings, dtedName, correction, lonConv,
                            latConv):
    gdal.UseExceptions()
    ds = gdal.Open(dtedName)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    # pre-compute 1/elevation_grid_spacing
    elevSpacInv = 1.0 / abs(xres * yres)

    # get the min and max northings and eastings
    eBound = eastings.max() + 100.0
    wBound = eastings.min() - 100.0
    nBound = northings.max() + 100.0
    sBound = northings.min() - 100.0

    # calculate the x and y indices into the DTED data for the lat/lon
    maxX = (eBound / lonConv - ulx) / xres
    minX = (wBound / lonConv - ulx) / xres
    minY = (nBound / latConv - uly) / yres
    maxY = (sBound / latConv - uly) / yres

    # only if these x and y indices are within the bounds of the DTED, get the
    # raster band and try to read in the DTED values
    dtedInterp = ones_like(eastings) * 1e-20
    if ((minX >= 0 and maxX < ds.RasterXSize) and (minY >= 0 and maxY < ds.RasterYSize)):
        rasterBand = ds.GetRasterBand(1)
        xSize = int(maxX - minX)
        ySize = int(maxY - minY)
        dtedData = rasterBand.ReadAsArray(int(minX), int(minY), xSize, ySize)

        # use nearest-neighbor interpolation initially to get the elevation for the pixels
        longitudes = eastings / lonConv
        latitudes = northings / latConv

        # calculate the indices into the dtedData array
        px = (longitudes - ulx) / xres - int(minX)
        py = (latitudes - uly) / yres - int(minY)

        leftLon = (px.astype('int') + minX) * xres + ulx
        upLat = (py.astype('int') + minY) * yres + uly

        # pre-compute the differences for the bilinear interpolation
        rightLonDiff = (leftLon + xres) - longitudes
        upLatDiff = upLat - latitudes
        leftLonDiff = longitudes - leftLon
        lowLatDiff = latitudes - (upLat + yres)

        dtedInterp = elevSpacInv * \
                     (dtedData[floor(py).astype('int'), floor(px).astype('int')] * rightLonDiff * lowLatDiff \
                      + dtedData[floor(py).astype('int'), ceil(px).astype('int')] * leftLonDiff * lowLatDiff \
                      + dtedData[ceil(py).astype('int'), floor(px).astype('int')] * rightLonDiff * upLatDiff \
                      + dtedData[ceil(py).astype('int'), ceil(px).astype('int')] * leftLonDiff * upLatDiff)

    return dtedInterp + correction


def getDTEDPoint(lat, lon, dtedName, correction):
    """Returns the digirtal elevation value closest to a latitude and longitude"""
    # dtedName = getDTEDName(lat, lon)
    gdal.UseExceptions()
    # open DTED file for reading
    ds = gdal.Open(dtedName)

    # get the geo transform info for the dted
    # ulx is upper left corner longitude
    # xres is the resolution in the x-direction (in degrees/sample)
    # xskew is useless (0.0)
    # uly is the upper left corner latitude
    # yskew is useless (0.0)
    # yres is the resolution in the y-direction (in degrees/sample)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    # calculate the x and y indices into the DTED data for the lat/lon
    px = (lon - ulx) / xres
    py = (lat - uly) / yres

    # only if these x and y indices are within the bounds of the DTED, get the
    # raster band and try to read in the DTED values
    elevation = -1e20
    if ((px >= 0 and px < ds.RasterXSize) and (py >= 0 and py < ds.RasterYSize)):
        rasterBand = ds.GetRasterBand(1)
        elevation = rasterBand.ReadAsArray(int(px), int(py), 1, 1)

    return elevation.item(0) + correction


def getInterpolatedDTED(lat, lon, dtedName, correction):
    """Returns the digital elevation for a latitude and longitude"""
    # dtedName = getDTEDName(lat, lon)
    gdal.UseExceptions()
    # open DTED file for reading
    ds = gdal.Open(dtedName)

    # get the geo transform info for the dted
    # ulx is upper left corner longitude
    # xres is the resolution in the x-direction (in degrees/sample)
    # xskew is useless (0.0)
    # uly is the upper left corner latitude
    # yskew is useless (0.0)
    # yres is the resolution in the y-direction (in degrees/sample)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    # pre-compute 1/elevation_grid_spacing
    elevSpacInv = 1.0 / abs(xres * yres)
    # calculate the x and y indices into the DTED data for the lat/lon
    px = (lon - ulx) / xres
    py = (lat - uly) / yres

    # only if these x and y indices are within the bounds of the DTED, get the
    # raster band and try to read in the DTED values
    elevation = -1e20
    if ((px >= 0 and px < ds.RasterXSize) and (py >= 0 and py < ds.RasterYSize)):
        rasterBand = ds.GetRasterBand(1)
        dtedData = rasterBand.ReadAsArray(int(px), int(py), 2, 2)

        # use bilinear interpolation to get the elevation for the lat/lon
        leftLon = int(px) * xres + ulx
        upLat = int(py) * yres + uly

        # pre compute the differences for the bilinear interpolation
        rightLonDiff = (leftLon + xres) - lon
        upLatDiff = upLat - lat
        # lowLatDiff = lat - lowLat
        leftLonDiff = lon - leftLon
        lowLatDiff = lat - (upLat + yres)
        # upLatDiff = (lowLat + yres) - lat

        elevation = elevSpacInv * (dtedData[0, 0] * rightLonDiff * lowLatDiff \
                                   + dtedData[0, 1] * leftLonDiff * lowLatDiff \
                                   + dtedData[1, 0] * rightLonDiff * upLatDiff \
                                   + dtedData[1, 1] * leftLonDiff * upLatDiff)

    return elevation + correction


def getDTEDName(lat, lon):
    """Return the path and name of the dted to load for the given lat/lon"""
    tmplat = int(floor(lat))
    tmplon = int(floor(lon))
    direw = 'e'
    if (tmplon < 0):
        direw = 'w'
    dirns = 'n'
    if (tmplat < 0):
        dirns = 's'

    return 'D:/Data/DTED/%s%03d/%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))


def centerOfMass(x, y, mass):
    comX = x.dot(mass) / mass.sum()
    comY = y.dot(mass) / mass.sum()
    return comX, comY


def getDetectedMovers(cpiDetections, maxZdB, zValuesDB, thresh):
    """Converts a 1D binary detection output for the ranges to a dictionary
    of movers with their average range bin, Doppler bin and angle bin.
    Inputs:
        cpiDetections - 1D binary numpy array, size = number of range bins
        maxZdB - 1D float numpy array containing maximum Z value from over
            all space-time bins for each range, size = number of range bins
        zValuesDB - 3D float numpy array containing all of the Z values in dB
            for all range, Doppler, and angle bins, size = Nrv x Ncpi x NSpatBins
    """
    # now, let's consolidate range detections into moving target detections
    rangeBins = nonzero(cpiDetections)
    rangeBins = rangeBins[0]
    numRangeDetections = rangeBins.size
    rangeBinsf = rangeBins + 0.0
    dopBins = zeros(rangeBins.shape, dtype='float64')
    angBins = zeros(rangeBins.shape, dtype='float64')

    movNum = 0
    moversM = []
    moversR = []
    moversD = []
    moversA = []
    if (numRangeDetections > 0):
        # get the angle and Doppler bin for the first range detection
        dopp, angg = nonzero(zValuesDB[rangeBins[0], ...] > thresh[rangeBins[0]])
        # store the range bins and doppler and angle bins
        Zs = zValuesDB[rangeBins[0], dopp, angg]
        dopBins[0] = dopp.dot(Zs) / Zs.sum()
        # dopBins[0] = dopp.mean()
        angBins[0] = angg.dot(Zs) / Zs.sum()
        # angBins[0] = angg.mean()
        # initialize the movers average range, Doppler, and Angle bin arrays
        moversR.append(rangeBins[0])
        moversD.append(dopBins[0])
        moversA.append(angBins[0])
        moversM.append(maxZdB[rangeBins[0]])

        rangePerMover = 1

        # loop through all of the range bins with detections
        for i in range(1, numRangeDetections):
            # get the Doppler bin and angle bin
            dopp, angg = nonzero(zValuesDB[rangeBins[i], ...] > thresh[rangeBins[i]])
            Zs = zValuesDB[rangeBins[i], dopp, angg]
            dopBins[i] = dopp.dot(Zs) / Zs.sum()
            # dopBins[i] = dopp.mean()
            angBins[i] = angg.dot(Zs) / Zs.sum()
            # angBins[i] = angg.mean()
            Rdiff = rangeBins[i] - rangeBins[i - 1]
            Ddiff = dopBins[i] - dopBins[i - 1]
            Adiff = angBins[i] - angBins[i - 1]

            # if the difference is only one range bin and the Doppler and Angle difference 
            # is less than a few bins, then we assume they are the same mover, 
            # otherwise we will start a new targetthan we assume they are the same mover
            if (Rdiff <= 1 and (abs(Ddiff) <= 1 and abs(Adiff) <= 2)):
                # check if the next values are greater than the previous, if so, replace
                if (maxZdB[rangeBins[i]] > moversM[movNum]):
                    moversM[movNum] = maxZdB[rangeBins[i]]
                    moversR[movNum] = rangeBins[i]
                    moversD[movNum] = dopBins[i]
                    moversA[movNum] = angBins[i]
                rangePerMover += 1
            else:
                # finalize the stats for the previous mover
                # moversR.append( rangeL.dot(zDBL) / zDBL.sum() )
                # moversD.append( doppL.dot(zDBL) / zDBL.sum() )
                # moversA.append( anggL.dot(zDBL) / zDBL.sum() )
                # moversM.append( zDBL.mean() )
                # moversR[ movNum ] /= float(rangePerMover)
                # moversD[ movNum ] /= float(rangePerMover)
                # moversA[ movNum ] /= float(rangePerMover)

                # initialize a new mover, icrement the mover pointer, and reset the
                # rangePerMover
                moversM.append(maxZdB[rangeBins[i]])
                moversR.append(rangeBins[i])
                moversD.append(dopBins[i])
                moversA.append(angBins[i])
                movNum += 1
                rangePerMover = 1

        # finalize the stats for the last mover
        # moversR[ movNum ] /= float(rangePerMover)
        # moversD[ movNum ] /= float(rangePerMover)
        # moversA[ movNum ] /= float(rangePerMover)

        # increment the movNum one last time to represent the total number of movers
        movNum += 1

    return array([moversR, moversD, moversA]), movNum


def getDetectedMoversMax(cpiDetections, maxZdB, zValuesDB, thresh):
    """Converts a 1D binary detection output for the ranges to a dictionary
    of movers with their average range bin, Doppler bin and angle bin.
    Inputs:
        cpiDetections - 1D binary numpy array, size = number of range bins
        maxZdB - 1D float numpy array containing maximum Z value from over
            all space-time bins for each range, size = number of range bins
        zValuesDB - 3D float numpy array containing all of the Z values in dB
            for all range, Doppler, and angle bins, size = Nrv x Ncpi x NSpatBins
    """
    # now, let's consolidate range detections into moving target detections
    rangeBins = nonzero(cpiDetections)
    rangeBins = rangeBins[0]
    numRangeDetections = rangeBins.size
    dopBins = zeros(rangeBins.shape, dtype='int64')
    angBins = zeros(rangeBins.shape, dtype='int64')

    movNum = 0
    moversM = []
    moversR = []
    moversD = []
    moversA = []
    if (numRangeDetections > 0):
        # get the angle and Doppler bin for the first range detection
        dopp, angg = nonzero(zValuesDB[rangeBins[0], ...] == maxZdB[rangeBins[0]])
        # store the range bins and doppler and angle bins
        Zs = zValuesDB[rangeBins[0], dopp, angg]
        # dopBins[0] = dopp.dot( Zs ) / Zs.sum()
        dopBins[0] = dopp.mean()
        # angBins[0] = angg.dot( Zs ) / Zs.sum()
        angBins[0] = angg.mean()
        # initialize the movers average range, Doppler, and Angle bin arrays
        moversR.append(rangeBins[0])
        moversD.append(dopBins[0])
        moversA.append(angBins[0])
        moversM.append(maxZdB[rangeBins[0]])

        rangePerMover = 1

        # loop through all of the range bins with detections
        for i in range(1, numRangeDetections):
            # get the Doppler bin and angle bin
            dopp, angg = nonzero(zValuesDB[rangeBins[i], ...] == maxZdB[rangeBins[i]])
            Zs = zValuesDB[rangeBins[i], dopp, angg]
            # dopBins[i] = dopp.dot( Zs ) / Zs.sum()
            dopBins[i] = dopp.mean()
            # angBins[i] = angg.dot( Zs ) / Zs.sum()
            angBins[i] = angg.mean()
            Rdiff = rangeBins[i] - rangeBins[i - 1]
            Ddiff = dopBins[i] - dopBins[i - 1]
            Adiff = angBins[i] - angBins[i - 1]

            # if the difference is only one range bin and the Doppler and Angle difference 
            # is less than a few bins, then we assume they are the same mover, 
            # otherwise we will start a new targetthan we assume they are the same mover
            if (Rdiff <= 1 and (abs(Ddiff) <= 1 and abs(Adiff) <= 2)):
                # check if the next values are greater than the previous, if so, replace
                if (maxZdB[rangeBins[i]] > moversM[movNum]):
                    moversM[movNum] = maxZdB[rangeBins[i]]
                    moversR[movNum] = rangeBins[i]
                    moversD[movNum] = dopBins[i]
                    moversA[movNum] = angBins[i]
                rangePerMover += 1
            else:
                # finalize the stats for the previous mover
                # moversR.append( rangeL.dot(zDBL) / zDBL.sum() )
                # moversD.append( doppL.dot(zDBL) / zDBL.sum() )
                # moversA.append( anggL.dot(zDBL) / zDBL.sum() )
                # moversM.append( zDBL.mean() )
                # moversR[ movNum ] /= float(rangePerMover)
                # moversD[ movNum ] /= float(rangePerMover)
                # moversA[ movNum ] /= float(rangePerMover)

                # initialize a new mover, icrement the mover pointer, and reset the
                # rangePerMover
                moversM.append(maxZdB[rangeBins[i]])
                moversR.append(rangeBins[i])
                moversD.append(dopBins[i])
                moversA.append(angBins[i])
                movNum += 1
                rangePerMover = 1

        # finalize the stats for the last mover
        # moversR[ movNum ] /= float(rangePerMover)
        # moversD[ movNum ] /= float(rangePerMover)
        # moversA[ movNum ] /= float(rangePerMover)

        # increment the movNum one last time to represent the total number of movers
        movNum += 1

    return array([moversR, moversD, moversA]), movNum


def getDetectedMoversCOM(cpiDetections, maxZdB, zValuesDB, thresh):
    """Converts a 1D binary detection output for the ranges to a dictionary
    of movers with their average range bin, Doppler bin and angle bin.
    Inputs:
        cpiDetections - 1D binary numpy array, size = number of range bins
        maxZdB - 1D float numpy array containing maximum Z value from over
            all space-time bins for each range, size = number of range bins
        zValuesDB - 3D float numpy array containing all of the Z values in dB
            for all range, Doppler, and angle bins, size = Nrv x Ncpi x NSpatBins
    """
    # now, let's consolidate range detections into moving target detections
    rangeBins = nonzero(cpiDetections)
    rangeBins = rangeBins[0]
    numRangeDetections = rangeBins.size
    rangeBinsf = rangeBins + 0.0
    dopBins = zeros(rangeBins.shape, dtype='float64')
    angBins = zeros(rangeBins.shape, dtype='float64')

    movNum = 0
    moversM = []
    moversR = []
    moversD = []
    moversA = []
    if (numRangeDetections > 0):
        # get the angle and Doppler bin for the first range detection
        dopp, angg = nonzero(zValuesDB[rangeBins[0], ...] > thresh[rangeBins[0]])
        # store the range bins and doppler and angle bins
        Zs = zValuesDB[rangeBins[0], dopp, angg]
        doppL = dopp + 0.0
        anggL = angg + 0.0
        zDBL = Zs + 0.0
        rangeL = ones(dopp.shape[0]) * rangeBinsf[0] + 0.0
        dopBins[0] = dopp.dot(Zs) / Zs.sum()
        # dopBins[0] = dopp.mean()
        angBins[0] = angg.dot(Zs) / Zs.sum()
        # angBins[0] = angg.mean()
        # initialize the movers average range, Doppler, and Angle bin arrays
        # moversR.append(rangeBins[0])
        # moversD.append(dopBins[0])
        # moversA.append(angBins[0])
        # moversM.append( maxZdB[rangeBins[0]] )

        rangePerMover = 1

        # loop through all of the range bins with detections
        for i in range(1, numRangeDetections):
            # get the Doppler bin and angle bin
            dopp, angg = nonzero(zValuesDB[rangeBins[i], ...] > thresh[rangeBins[i]])
            Zs = zValuesDB[rangeBins[i], dopp, angg]
            dopBins[i] = dopp.dot(Zs) / Zs.sum()
            # dopBins[i] = dopp.mean()
            angBins[i] = angg.dot(Zs) / Zs.sum()
            # angBins[i] = angg.mean()
            Rdiff = rangeBins[i] - rangeBins[i - 1]
            Ddiff = dopBins[i] - dopBins[i - 1]
            Adiff = angBins[i] - angBins[i - 1]

            # if the difference is only one range bin and the Doppler and Angle difference 
            # is less than a few bins, then we assume they are the same mover, 
            # otherwise we will start a new targetthan we assume they are the same mover
            if (Rdiff <= 1 and (abs(Ddiff) <= 3 and abs(Adiff) <= 4)):
                # check if the next values are greater than the previous, if so, replace
                doppL = concatenate([doppL, dopp])
                anggL = concatenate([anggL, angg])
                zDBL = concatenate([zDBL, Zs])
                rangeL = concatenate([rangeL, ones(dopp.shape[0]) * rangeBinsf[i]])

                #                if (maxZdB[rangeBins[i]] > moversM[ movNum ]):
                #                    moversM[ movNum ] = maxZdB[rangeBins[i]]
                #                    moversR[ movNum ] = rangeBins[i]
                #                    moversD[ movNum ] = dopBins[i]
                #                    moversA[ movNum ] = angBins[i]
                rangePerMover += 1
            else:
                # finalize the stats for the previous mover
                moversR.append(rangeL.dot(zDBL) / zDBL.sum())
                moversD.append(doppL.dot(zDBL) / zDBL.sum())
                moversA.append(anggL.dot(zDBL) / zDBL.sum())
                moversM.append(zDBL.mean())
                # moversR[ movNum ] /= float(rangePerMover)
                # moversD[ movNum ] /= float(rangePerMover)
                # moversA[ movNum ] /= float(rangePerMover)

                # initialize a new mover, icrement the mover pointer, and reset the
                # rangePerMover
                doppL = dopp + 0.0
                anggL = angg + 0.0
                zDBL = Zs + 0.0
                rangeL = ones(dopp.shape[0]) * rangeBinsf[i] + 0.0
                # moversM.append( maxZdB[rangeBins[i]])
                # moversR.append( rangeBins[i] )
                # moversD.append( dopBins[i] )
                # moversA.append( angBins[i] )
                movNum += 1
                rangePerMover = 1

        # finalize the stats for the last mover
        moversR.append(rangeL.dot(zDBL) / zDBL.sum())
        moversD.append(doppL.dot(zDBL) / zDBL.sum())
        moversA.append(anggL.dot(zDBL) / zDBL.sum())
        moversM.append(zDBL.mean())

        # moversR[ movNum ] /= float(rangePerMover)
        # moversD[ movNum ] /= float(rangePerMover)
        # moversA[ movNum ] /= float(rangePerMover)

        # increment the movNum one last time to represent the total number of movers
        movNum += 1

    return array([moversR, moversD, moversA]), movNum


def rangeCompressData(
        rawCPIData, matchedFilterMat, FFTLength, IFFTLength, NRangeBins, Ncpi):
    # compute the FFT over the last dimension (dimension indices are zero based)
    fdata = fft(rawCPIData, n=FFTLength, axis=1)
    data = ifft(fdata * matchedFilterMat, n=IFFTLength, axis=1)
    return data[:, :NRangeBins].astype('complex64')


def getWindowedFFTRefPulseUpdated(
        refdata, FFTLength, myWindow, header=HeaderParser):
    # compute the number of samples in a transmit pulse
    # Nsam_in_pulse = int(header.tp / header.dt)
    # Zero-out samples in the reference pulse outside of the length of the actual pulse
    # refdata[ :dataOffset ] = 0.0
    # refdata[ dataOffset + Nsam_in_pulse: ] = 0.0
    # pre-calculate the frequency spectrum of the reference pulses
    frefpulse = fft(refdata, FFTLength)

    # generate a window in the frequency domain for reducing the effect of zeroing
    # out the negative frequencies and everything outside of our bandwidth
    # (we will be converting from a real-valued signal to a complex valued signal)
    # compute the length of the frequency window
    len_freq_win = int(floor(header.BW * FFTLength * header.dt))
    # compute the leadin length (or number of bins offset into the FFT)
    leadin_len = int(floor(header.freqOffset * FFTLength * header.dt))

    # create the window using a kaiser window (ideally we would use a Taylor window
    # -35 dB sidelobe levels)
    freq_Window = zeros(FFTLength, dtype='complex64')
    freq_Window[leadin_len:leadin_len + len_freq_win] = myWindow
    # freq_Window[ leadin_len:leadin_len + len_freq_win ] = ones(len_freq_win)

    # window the frequency spectrum of the reference pulse
    frefpulse = frefpulse * freq_Window
    return frefpulse


def getWindowedFFTRefPulse(refdata, dataOffset, FFTLength, myWindow, header=HeaderParser):
    # compute the number of samples in a transmit pulse
    Nsam_in_pulse = int(header.tp / header.dt)
    # Zero-out samples in the reference pulse outside of the length of the actual pulse
    refdata[:dataOffset] = 0.0
    refdata[dataOffset + Nsam_in_pulse:] = 0.0
    # pre-calculate the frequency spectrum of the reference pulses
    frefpulse = fft(refdata[dataOffset:], FFTLength)

    # generate a window in the frequency domain for reducing the effect of zeroing
    # out the negative frequencies and everything outside of our bandwidth
    # (we will be converting from a real-valued signal to a complex valued signal)
    # compute the length of the frequency window
    len_freq_win = int(floor(header.BW * FFTLength * header.dt))
    # compute the leadin length (or number of bins offset into the FFT)
    leadin_len = int(floor(header.freq_offset * FFTLength * header.dt))

    # create the window using a kaiser window (ideally we would use a Taylor window
    # -35 dB sidelobe levels)
    freq_Window = zeros(FFTLength, dtype='complex64')
    freq_Window[leadin_len:leadin_len + len_freq_win] = myWindow
    # freq_Window[ leadin_len:leadin_len + len_freq_win ] = ones(len_freq_win)

    # window the frequency spectrum of the reference pulse
    frefpulse = frefpulse * freq_Window
    return frefpulse


def generateFullDimensionSteeringVectors(NDopplerBins, NSpatBins, Ncpi, Nchan, a_w, b_w):
    # create our normalized spatial frequencies
    ntheta_spacing = 1.0 / NSpatBins
    ntheta = (arange(NSpatBins) - NSpatBins / 2) * ntheta_spacing
    # ntheta = linspace(-0.5,0.5,NSpatBins)

    # create our normalized Doppler frequencies
    ndop_spacing = 1.0 / NDopplerBins
    ndop = arange(NDopplerBins) * ndop_spacing
    # ndop = linspace(0.0, 1.0, NDopplerBins)

    # create an array in which we will store all of the space-time steering vectors
    stsv = zeros((NDopplerBins * NSpatBins, Ncpi, Nchan), order='C', dtype='complex64')

    # t_a = chebwin(Nchan,-45).reshape((1,Nchan))
    # t_a = kaiser(Nchan, 3.0).reshape((1,Nchan))
    # t_a = ones((1,Nchan))

    # loop through all of the Doppler and spatial frequencies and create the steering vectors
    for dopI in range(NDopplerBins):
        b_i = exp(1j * 2 * pi * b_w * ndop[dopI])

        for spatI in range(NSpatBins):
            a_i = exp(1j * 2 * pi * a_w * ntheta[spatI])  # * t_a
            # the dot product here provides an outer product and results
            # in a Ncpi x Nchan matrix
            stsv[dopI * NSpatBins + spatI, :, :] = b_i.dot(a_i)

    return stsv


def marcumq(alpha, T, end, numSamples):
    """My implimentation of the Marcum-Q function"""
    t = linspace(T, end, numSamples)
    dt = t[1] - t[0]
    Qm = 0.0
    alpha_2 = alpha ** 2
    for i in arange(t.size):
        Qm += t[i] * exp(-0.5 * (t[i] ** 2 + alpha_2)) * i0(alpha * t[i]) * dt
    return Qm


def DFT_matrix(N):
    X, Y = meshgrid(arange(N), arange(N))
    omega = exp(- 2 * pi * 1j / N)
    W = power(omega, X * Y) / sqrt(N)
    return W


def getPSPDDopplerFilterBank(M, K, SLL):
    # compute Mprime
    Mprime = M - K + 1
    dft = DFT_matrix(M)
    # Modify the DFT matrix to be Mprime x M instead of M x M
    dft = dft[:Mprime, :]
    # generate the Doppler window
    t_dop_win = chebwin(Mprime, SLL)
    # generate the Doppler filter bank
    Fmat0 = diag(t_dop_win).dot(dft.conj())

    # we need to return the hermitian of this
    Fmat = Fmat0.T.conj().copy()
    # make sure it is complex64 data type
    Fmat = Fmat.astype('complex64')

    return Fmat


def abpdSTAP(radar_data, Rcn, Nsam, Ncpi, Nchan, ind, K, sll, ntheta, estimate=0):
    """PRI-staggered post-Doppler"""
    print("Performing adjacent-bin post-Doppler with K=%d and sidelobe-level %0.1f dB" % (K, sll))
    P = float(K - 1) / 2.0
    Mp = Ncpi - K + 1

    # define the DFT matrix
    U = DFT_matrix(Ncpi)

    if sll == 0:
        t_f = ones(Ncpi)
    else:
        t_f = chebwin(Ncpi, sll)
    # Doppler filter bank
    Fmat = diag(t_f).dot(U.conj())

    b_wi = arange(0, Ncpi).reshape((Ncpi, 1))
    # t_b = chebwin(Ncpi,-40).reshape((Ncpi,1))
    t_b = ones((Ncpi, 1))
    a_wi = arange(-(float(Nchan) - 1) / 2.0, float(Nchan) / 2.0, 1).reshape((Nchan, 1))
    # t_a = chebwin(Nchan,-30).reshape((Nchan,1))
    t_a = ones((Nchan, 1))
    a_i = exp(1j * 2 * pi * a_wi * ntheta) * t_a

    freq_step = 1.0 / float(Ncpi)

    # precompute  the Nchan size identity matrix
    eyeNchan = eye(Nchan)

    # I want to record my computed weights for each subCPI at broadside
    W = zeros((Ncpi * Nchan, Mp), dtype='complex128')

    # loop through each Doppler bin to do the adaptive beamforming
    for ip in range(int(P), Mp):
        # extract the mth Doppler bin filter vector

        if K % 2:
            Fm = Fmat[:, ip - int(P)].reshape((Ncpi, 1))
            for k in range(int(1 - P), int(P + 1)):
                fm = Fmat[:, ip + k].reshape((Ncpi, 1))
                Fm = hstack((Fm, fm))
        else:
            Fm = Fmat[:, ip - int(P)].reshape((Ncpi, 1))
            for k in range(1 - int(P), int(P) + 2):
                fm = Fmat[:, ip + k].reshape((Ncpi, 1))
                Fm = hstack((Fm, fm))
        # define the preprocessor for the mth Doppler bin
        T = kron(Fm, eyeNchan)

        if estimate:
            Rabpd = zeros((K * Nchan, K * Nchan), dtype='complex128')
            gaurd_bins = 2
            start_ind = ind - int(estimate / 2) - gaurd_bins
            stop_ind = ind + estimate + 2 * gaurd_bins + 1

            num_bin_avg1 = 0
            for i in range(start_ind, stop_ind + 1):
                if ((i < (ind - gaurd_bins)) or (i > (ind + gaurd_bins))):
                    num_bin_avg1 += 1
                    x = radar_data[i, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C')
                    x_re = T.conj().T.dot(x)
                    Rabpd += x_re.dot(x_re.T.conj())
            Rabpd /= num_bin_avg1
        else:
            Rabpd = T.conj().T.dot(Rcn.dot(T))
        # calculate the inverse of the covariance matrix
        Rabpd_inv = pinv(Rabpd)

        # now that we have an estimate of the interference covariance matrix,
        # we can compute the spatial weights
        # need to calculate the normalized Doppler for this Doppler bin
        if K % 2:
            nf_rm = freq_step * float(ip)
        else:
            nf_rm = freq_step * float(ip + P)
        b_i = exp(1j * 2 * pi * b_wi * nf_rm) * t_b
        g_t = kron(b_i, a_i)
        g_re = T.conj().T.dot(g_t)
        w_re = Rabpd_inv.dot(g_re)
        W[:, ip] = T.dot(w_re).flatten()

    return W


def pspdSTAP(radar_data, Rcn, Nsam, Ncpi, Nchan, ind, K, sll, ntheta, diag_load, estimate=0, guard_bins=2):
    """PRI-staggered post-Doppler"""
    # print "Performing PRI-staggered post-Doppler with K=%d and sidelobe-level %0.1f dB" % (K, sll)
    Mp = Ncpi - K + 1
    # define the DFT matrix
    U = DFT_matrix(Ncpi)
    # need to modify DFT matrix to be M'x M instead of M x M
    U = U[:Mp, :]
    if sll == 0:
        t_f = ones(Mp)
    else:
        t_f = chebwin(Mp, sll)
    # Doppler filter bank
    Fmat = diag(t_f).dot(U.conj())

    b_wi = arange(0, Ncpi).reshape((Ncpi, 1))
    # t_b = chebwin(Ncpi,-40).reshape((Ncpi,1))
    t_b = ones((Ncpi, 1))
    a_wi = arange(-(float(Nchan) - 1) / 2.0, float(Nchan) / 2.0, 1).reshape((Nchan, 1))
    t_a = chebwin(Nchan, -45).reshape((Nchan, 1))
    t_a = ones((Nchan, 1))
    a_i = exp(1j * 2 * pi * a_wi * ntheta) * t_a

    freq_step = 1.0 / float(Ncpi)

    # precompute  the Nchan size identity matrix
    eyeNchan = eye(Nchan)
    # precompute some zeros vectors used in toeplitz
    toep_col = zeros((Ncpi, 1), dtype='complex128')
    toep_row = zeros((1, K), dtype='complex128')

    # create diagonal loading matrix
    Rdiag = eye(K * Nchan) * diag_load

    # I want to record my computed weights for each subCPI at broadside
    W = zeros((Ncpi * Nchan, Ncpi), dtype='complex128')

    # loop through each Doppler bin to do the adaptive beamforming
    for ip in range(Ncpi):
        # extract the mth Doppler bin filter vector
        fm = Fmat[:, ip].reshape((Mp, 1))
        # construct Doppler Toeplitz matrix
        toep_col[:Mp, 0] = fm.flatten()
        toep_row[0, 0] = fm.item(0)
        Fm = toeplitz(toep_col, toep_row)
        # define the preprocessor for the mth Doppler bin
        T = kron(Fm, eyeNchan)

        if estimate:
            Rpspd = zeros((K * Nchan, K * Nchan), dtype='complex128')
            start_ind = ind - int(estimate / 2) - guard_bins
            if (start_ind < 0):
                start_ind = 0
            stop_ind = ind + int(estimate / 2) + guard_bins + 1
            if (stop_ind > Nsam):
                stop_ind = Nsam - 1

            num_bin_avg1 = 0
            for i in range(start_ind, stop_ind + 1):
                if ((i < (ind - guard_bins)) or (i > (ind + guard_bins))):
                    num_bin_avg1 += 1
                    x = radar_data[i, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C')
                    x_re = T.conj().T.dot(x)
                    Rpspd += x_re.dot(x_re.T.conj())
            Rpspd /= num_bin_avg1
            Rpspd += Rdiag
        else:
            Rpspd = T.conj().T.dot(Rcn.dot(T))
        # calculate the inverse of the covariance matrix
        Rpspd_inv = pinv(Rpspd)

        # now that we have an estimate of the interference covariance matrix,
        # we can compute the spatial weights
        # need to calculate the normalized Doppler for this Doppler bin
        nf_rm = freq_step * ip
        b_i = exp(1j * 2 * pi * b_wi * nf_rm) * t_b
        g_t = kron(b_i, a_i)
        g_re = T.conj().T.dot(g_t)
        w_re = Rpspd_inv.dot(g_re)
        W[:, ip] = T.dot(w_re).flatten()

    return W


def pspdSTAPSolve(radar_data, Rcn, Nsam, Ncpi, Nchan, ind, K, sll, ntheta, diag_load, estimate=0, guard_bins=2):
    """PRI-staggered post-Doppler"""
    # print "Performing PRI-staggered post-Doppler with K=%d and sidelobe-level %0.1f dB" % (K, sll)
    Mp = Ncpi - K + 1
    # define the DFT matrix
    U = DFT_matrix(Ncpi)
    # need to modify DFT matrix to be M'x M instead of M x M
    U = U[:Mp, :]
    if sll == 0:
        t_f = ones(Mp)
    else:
        t_f = chebwin(Mp, sll)
    # Doppler filter bank
    Fmat = diag(t_f).dot(U.conj())

    b_wi = arange(0, Ncpi).reshape((Ncpi, 1))
    # t_b = chebwin(Ncpi,-40).reshape((Ncpi,1))
    t_b = ones((Ncpi, 1))
    a_wi = arange(-(float(Nchan) - 1) / 2.0, float(Nchan) / 2.0, 1).reshape((Nchan, 1))
    # t_a = chebwin(Nchan,-60).reshape((Nchan,1))
    t_a = ones((Nchan, 1))
    a_i = exp(1j * 2 * pi * a_wi * ntheta) * t_a

    freq_step = 1.0 / float(Ncpi)

    # precompute  the Nchan size identity matrix
    eyeNchan = eye(Nchan)
    # precompute some zeros vectors used in toeplitz
    toep_col = zeros((Ncpi, 1), dtype='complex128')
    toep_row = zeros((1, K), dtype='complex128')

    # create diagonal loading matrix
    Rdiag = eye(K * Nchan) * diag_load

    # I want to record my computed weights for each subCPI at broadside
    # W = zeros((Ncpi*Nchan,Ncpi),dtype='complex128')
    Z = zeros(Ncpi, dtype='complex128')

    # loop through each Doppler bin to do the adaptive beamforming
    for ip in range(Ncpi):
        # extract the mth Doppler bin filter vector
        fm = Fmat[:, ip].reshape((Mp, 1))
        # construct Doppler Toeplitz matrix
        toep_col[:Mp, 0] = fm.flatten()
        toep_row[0, 0] = fm.item(0)
        Fm = toeplitz(toep_col, toep_row)
        # define the preprocessor for the mth Doppler bin
        T = kron(Fm, eyeNchan)

        # get myself the reduced dimension data snapshot for this Doppler bin
        # and range bin
        x_d_re = T.conj().T.dot(radar_data[ind, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C'))

        if estimate:
            Rpspd = zeros((K * Nchan, K * Nchan), dtype='complex128')
            start_ind = ind - int(estimate / 2) - guard_bins
            if (start_ind < 0):
                start_ind = 0
            stop_ind = ind + int(estimate / 2) + guard_bins + 1
            if (stop_ind > Nsam):
                stop_ind = Nsam - 1

            num_bin_avg1 = 0
            for i in range(start_ind, stop_ind + 1):
                if ((i < (ind - guard_bins)) or (i > (ind + guard_bins))):
                    num_bin_avg1 += 1
                    x = radar_data[i, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C')
                    x_re = T.conj().T.dot(x)
                    Rpspd += x_re.dot(x_re.T.conj())
            Rpspd /= num_bin_avg1
            Rpspd += Rdiag
        else:
            Rpspd = T.conj().T.dot(Rcn.dot(T))
        # calculate the inverse of the covariance matrix
        Rpspd_inv = pinv(Rpspd)

        # now that we have an estimate of the interference covariance matrix,
        # we can compute the spatial weights
        # need to calculate the normalized Doppler for this Doppler bin
        nf_rm = freq_step * ip
        b_i = exp(1j * 2 * pi * b_wi * nf_rm) * t_b
        g_t = kron(b_i, a_i)
        g_re = T.conj().T.dot(g_t)
        w_re = Rpspd_inv.dot(g_re)
        Z[ip] = w_re.conj().T.dot(x_d_re).item()
        # W[:,ip] = T.dot(w_re).flatten()

    return Z


def pdabSTAP(radar_data, Rcn, Nsam, Ncpi, Nchan, ind, sll, ntheta, estimate=0):
    """post-Doppler adaptive-beamforming (factored STAP)"""
    print("Performing factored STAP with sidelobe-level: %0.1f dB" % (sll))
    # define the DFT matrix
    U = DFT_matrix(Ncpi)
    if sll == 0:
        t_d = ones(Ncpi)
    else:
        t_d = chebwin(Ncpi, sll)
    Fmat = diag(t_d).dot(U.conj())

    a_wi = arange(-(float(Nchan) - 1) / 2.0, float(Nchan) / 2.0, 1).reshape((Nchan, 1))
    # t_a = chebwin(Nchan,-30).reshape((Nchan,1))
    t_a = ones((Nchan, 1))
    a_i = exp(1j * 2 * pi * a_wi * ntheta) * t_a

    # precompute the Nchan size identity matrix
    eyeNchan = eye(Nchan)

    # I want to record my computed weights for each subCPI at broadside
    W = zeros((Ncpi * Nchan, Ncpi), dtype='complex128')

    # loop through each Doppler bin to do the adaptive spatial beamforming
    for ip in range(Ncpi):
        # extract the ith (mth) Doppler bin filter vector
        fm = Fmat[:, ip].reshape((Ncpi, 1))
        # define the preprocessor for the mth Doppler bin
        T = kron(fm, eyeNchan)

        if estimate:
            Rpdab = zeros((Nchan, Nchan), dtype='complex128')
            gaurd_bins = 2
            start_ind = ind - int(estimate / 2) - gaurd_bins
            stop_ind = ind + estimate + 2 * gaurd_bins + 1

            num_bin_avg1 = 0
            for i in range(start_ind, stop_ind + 1):
                if ((i < (ind - gaurd_bins)) or (i > (ind + gaurd_bins))):
                    num_bin_avg1 += 1
                    x = radar_data[i, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C')
                    x_re = T.conj().T.dot(x)
                    Rpdab += x_re.dot(x_re.T.conj())
            Rpdab /= num_bin_avg1
        else:
            Rpdab = T.conj().T.dot(Rcn.dot(T))
        # calculate the inverse of the covariance matrix
        Rpdab_inv = pinv(Rpdab)

        # now that we have an estimate of the interference covariance matrix,
        # we can compute the spatial weights
        # (I'm not going to taper the desired spatial response at all)
        g_re = a_i.copy() + 0.0
        w_re = Rpdab_inv.dot(g_re)
        W[:, ip] = T.dot(w_re).flatten()

    return W


def espdSTAP(radar_data, Rcn, Nsam, Ncpi, Nchan, ind, K, NDFT, ntheta, estimate=0):
    """Element-space pre-Doppler (Adaptive DPCA)"""
    print("Performing adaptive-DPCA (or Element-space Pre-Doppler STAP) with K=%d" % (K))
    Mp = Ncpi - K + 1
    print("There will be M' = %d subCPIs" % (Mp))

    # it may speed things up to predefine an identity matrix size K and size Nchan
    eyeK = eye(K)
    eyeNchan = eye(Nchan)

    # define the binomial window we will use
    if K == 2:
        bin_win = array([1.0, 1.0]).reshape((2, 1))
    if K == 3:
        bin_win = array([1.0, 2.0, 1.0]).reshape((3, 1))

    # define our Doppler space center (omega_b = omega_c + 0.5)
    omega_c = 0.0
    omega_b = omega_c + 0.5

    # and precompute the reduced dimension temporal steering vector
    b_wi = arange(0, K).reshape((K, 1))
    a_wi = arange(-(float(Nchan) - 1) / 2.0, float(Nchan) / 2.0, 1).reshape((Nchan, 1))
    b_re = bin_win * exp(1j * 2 * pi * b_wi * omega_b)

    # I want to record my computed weights for each subCPI at broadside
    W = zeros((Ncpi * Nchan, NDFT), dtype='complex128')
    # record the output signal from each subCPI
    # y = zeros((Mp,1),dtype='complex128')

    # let's start looping through the subCPIs
    for ip in range(Mp):
        # each subCPI is comprised of pulses p,...,p+K-1
        pulses = range(ip, ip + K)
        # create the selection matrix
        J = zeros((Ncpi, K))
        J[pulses, :] = eyeK
        # define our preprocessor
        T = kron(J, eyeNchan)

        # now, use the preprocessor to reduce the dimension of the data
        # and obtain an estimate of the Interference covariance matrix
        if estimate:
            Respd = zeros((K * Nchan, K * Nchan), dtype='complex128')
            gaurd_bins = 2
            start_ind = ind - int(estimate / 2) - gaurd_bins
            stop_ind = ind + estimate + 2 * gaurd_bins + 1
            num_bin_avg1 = 0
            for i in range(start_ind, stop_ind + 1):
                if ((i < (ind - gaurd_bins)) or (i > (ind + gaurd_bins))):
                    num_bin_avg1 += 1
                    x = radar_data[i, :Ncpi, :].reshape((Ncpi * Nchan, 1), order='C')
                    x_re = T.T.dot(x)
                    Respd += x_re.dot(x_re.T.conj())
            Respd /= num_bin_avg1
        else:
            Respd = T.T.dot(Rcn.dot(T))

        Respd_inv = pinv(Respd)

        # now that we have an estimate of the interference covariance matrix,
        # we can compute the weights over a range of angles
        # x = radar_data[ind,:Ncpi,:].reshape((Ncpi*Nchan,1),order='C')
        # x_re = T.T.dot(x)
        a_i = exp(1j * 2 * pi * a_wi * ntheta)
        g_re = kron(b_re, a_i)
        w_re = Respd_inv.dot(g_re)
        W[ip * Nchan:ip * Nchan + K * Nchan, ip] = w_re.flatten()
        # y[ip,0] = w_re.conj().T.dot(x_re).item(0)

    # define the DFT matrix
    U = DFT_matrix(NDFT)
    t_d = ones(NDFT)
    t_d = chebwin(NDFT, -40.0)
    Fmat = diag(t_d).dot(U.conj())

    # note that the W has an implicit zero-padding at the end
    Wespd3 = W.dot(Fmat)

    return Wespd3


def mdv_udsf_calc(Lsinr, doppler_bins, lamda, PRF, thresh):
    stats = {'MDV': {}, 'UDSF': {}, 'UDSFdir': {}}
    num_bins = Lsinr.size

    for i in range(thresh.size):
        ind = nonzero(Lsinr > 10.0 ** (thresh[i] / 10.0))
        num_above_threshold = ind[0].size
        if num_above_threshold > 0:
            # the lowest doppler value will be the largest still less than 0
            low_dop_ind = 0
            up_dop_ind = num_above_threshold - 1
            for ii in range(num_above_threshold):
                if (doppler_bins[ind[0][ii]] < 0.0):
                    low_dop_ind = ind[0][ii]
                    ii1 = ii + 1
                    if ii1 == num_above_threshold: ii1 = num_above_threshold - 1
                    up_dop_ind = ind[0][ii1]
            print("For threshold %0.1f, the low doppler is doppler_bin[%d] = %0.2f" % (
                i, low_dop_ind, doppler_bins[low_dop_ind]))
            print("And high doppler is doppler_bin[%d] = %0.2f" % (up_dop_ind, doppler_bins[up_dop_ind]))
            if low_dop_ind < 0: low_dop_ind = 0
            if up_dop_ind > num_bins - 1: up_dop_ind = num_bins - 1
        else:
            print("There weren't any LSINR values above %0.1f" % (thresh[i]))
            continue

        # since there were LSINR values above the threshold, we will
        # make the calculations and write them to stats
        low_dop = doppler_bins[low_dop_ind]
        up_dop = doppler_bins[up_dop_ind]
        f_min = 0.5 * (up_dop - low_dop)
        stats['MDV'][thresh[i]] = lamda * f_min / 2
        # most of the time the simplified way of UDSF calculation
        # is sufficient (when there is only a null on the mainlobe
        # clutter), but if there are nulls in other places, or the
        # Lsinr is very low, then the more involved computation
        # should be performed
        stats['UDSF'][thresh[i]] = 1 - 2 * f_min / PRF
        Prob = float(num_above_threshold) / float(num_bins)
        stats['UDSFdir'][thresh[i]] = Prob
    return stats


def getInterpolatedAntennaResponse(filename):
    # load in the matlab data with the antenna responses
    matDict = loadmat(filename)
    thetaDeg = matDict['Thetadeg']
    # 'uniform_spacing_array' or 'increased_spacing_array'
    arrayResponse = matDict['increased_spacing_array']
    arrayResponse = 10 ** (arrayResponse / 20.0)
    arrayResponse /= arrayResponse.max()

    # get the indices for -90 to 90 degrees
    ind = nonzero(logical_and(thetaDeg > -90.0, thetaDeg < 90.0))
    thetaDeg = thetaDeg[ind]
    arrayResponse = arrayResponse[ind]

    return thetaDeg, arrayResponse


def rxResponse(dxt, dyt, dzt, rt, dxr, dyr, dzr, rr, Ria, Nsub, wNsub, Nel, wNel, a, b, d, lamda):
    """
    Inputs are x, y, and z components of the intertial Range vector, which
    points from the aircraft to the point target (full magnitude, not normalized),
    as well as the rotation matrices from inertial-to-body,
    body-to-perpindicular body, and perpindicular body-to-antenna frames.
    As well as the weights for the Tx array, Rx array, and elevation array and
    the number of elements in the Tx, Rx, and Elevation arrays, and
    width (a) and height (b) of a single element, and the wavenumber.
    """
    # need to get rid of this maxAntResp crap.  This may very easily
    # lead to bugs later on
    maxAntResp = 0.0040339533864340834
    AFsubmax = wNsub.sum() / Nsub
    AFelmax = wNel.sum() / Nel
    maxAntResp = AFsubmax * AFelmax

    k = 2 * pi / lamda
    # create the vector pointing from the Tx to the target
    dpt = array([[dxt], [dyt], [dzt]])
    # create the vector pointing from the Rx to the target
    dpr = array([[dxr], [dyr], [dzr]])
    # rotate both of these vectors to the antenna frame
    r_at = Ria.dot(dpt)
    r_ar = Ria.dot(dpr)
    phit = arctan2(-r_at.item(2), r_at.item(0))
    phir = arctan2(-r_ar.item(2), r_ar.item(0))
    thetat = arccos(r_at.item(1) / rt)
    thetar = arccos(r_ar.item(1) / rr)

    ant_el = arcsin(r_at.item(2) / rr)
    ant_az = arctan2(r_at.item(0), r_at.item(1))

    # cttx = cos(thetat)
    sttx = sin(thetat)
    # ctrx = cos(thetar)
    strx = sin(thetar)
    cptx = cos(phit)
    sptx = sin(phit)
    cprx = cos(phir)
    # sprx = sin(phir)
    E_i = sinc(a * k * sttx * cptx / (2 * pi)) * sinc(
        b * k * sttx * sptx / (2 * pi))  # * sqrt(cttx**2 * cptx**2 + sptx**2)

    # compute the Rx array factor for the sub-array
    AFsub = 0.0
    for i in range(int(Nsub / 2)):
        AFsub += wNsub[i] * cos((2 * (i + 1) - 1) * k * d * strx * cprx / 2) / Nsub
    # compute the Elevation array factor
    AFel = 0.0
    for i in range(int(Nel / 2)):
        AFel += wNel[i] * cos((2 * (i + 1) - 1) * k * d * sttx * sptx / 2) / Nel

    # combine the element electric field with all of the array factors to obtain
    # the two way normalized radiation pattern
    S_2way = E_i * (AFsub * AFel)
    return S_2way / maxAntResp, ant_el, ant_az


def antennaResponse(dxt, dyt, dzt, rt, dxr, dyr, dzr, rr, Ria, N, wN, Nsub, wNsub, Nel, wNel, a, b, d, lamda):
    """
    Inputs are x, y, and z components of the intertial Range vector, which
    points from the aircraft to the point target (full magnitude, not normalized),
    as well as the rotation matrices from inertial-to-body,
    body-to-perpindicular body, and perpindicular body-to-antenna frames.
    As well as the weights for the Tx array, Rx array, and elevation array and
    the number of elements in the Tx, Rx, and Elevation arrays, and
    width (a) and height (b) of a single element, and the wavenumber.
    """
    # need to get rid of this maxAntResp crap.  This may very easily
    # lead to bugs later on
    maxAntResp = 0.0040339533864340834
    AFtxmax = wN.sum() / N
    AFsubmax = wNsub.sum() / Nsub
    AFelmax = wNel.sum() / Nel
    maxAntResp = AFtxmax * AFelmax * AFsubmax * AFelmax

    k = 2 * pi / lamda
    # create the vector pointing from the Tx to the target
    dpt = array([[dxt], [dyt], [dzt]])
    # create the vector pointing from the Rx to the target
    dpr = array([[dxr], [dyr], [dzr]])
    # rotate both of these vectors to the antenna frame
    r_at = Ria.dot(dpt)
    r_ar = Ria.dot(dpr)
    phit = arctan2(-r_at.item(2), r_at.item(0))
    phir = arctan2(-r_ar.item(2), r_ar.item(0))
    thetat = arccos(r_at.item(1) / rt)
    thetar = arccos(r_ar.item(1) / rr)

    ant_el = arcsin(r_at.item(2) / rr)
    ant_az = arctan2(r_at.item(0), r_at.item(1))

    # cttx = cos(thetat)
    sttx = sin(thetat)
    # ctrx = cos(thetar)
    strx = sin(thetar)
    cptx = cos(phit)
    sptx = sin(phit)
    cprx = cos(phir)
    # sprx = sin(phir)
    E_i = sinc(a * k * sttx * cptx / (2 * pi)) * sinc(
        b * k * sttx * sptx / (2 * pi))  # * sqrt(cttx**2 * cptx**2 + sptx**2)
    # compute the Tx array factor
    AFtx = 0.0
    for i in range(int(N / 2)):
        AFtx += wN[i] * cos((2 * (i + 1) - 1) * k * d * sttx * cptx / 2) / N
    # compute the Rx array factor for the sub-array
    AFsub = 0.0
    for i in range(int(Nsub / 2)):
        AFsub += wNsub[i] * cos((2 * (i + 1) - 1) * k * d * strx * cprx / 2) / Nsub
    # compute the Elevation array factor
    AFel = 0.0
    for i in range(int(Nel / 2)):
        AFel += wNel[i] * cos((2 * (i + 1) - 1) * k * d * sttx * sptx / 2) / Nel

    # combine the element electric field with all of the array factors to obtain
    # the two way normalized radiation pattern
    S_2way = E_i * (AFtx * AFel) * E_i * (AFsub * AFel)
    return S_2way / maxAntResp, ant_el, ant_az


def hornResponse(dxt, dyt, dzt, rt, dxr, dyr, dzr, rr, Ria, a, b, d, lamda):
    """
    Inputs are x, y, and z components of the intertial Range vector, which
    points from the aircraft to the point target (full magnitude, not normalized),
    as well as the rotation matrices from inertial-to-body,
    body-to-perpindicular body, and perpindicular body-to-antenna frames.
    As well as the weights for the Tx array, Rx array, and elevation array and
    the number of elements in the Tx, Rx, and Elevation arrays, and
    width (a) and height (b) of a single element, and the wavenumber.
    """
    maxAntResp = 0.0040339533864340834
    maxAntResp = 1.0
    k = 2 * pi / lamda
    # create the vector pointing from the Tx to the target
    dpt = array([[dxt], [dyt], [dzt]])
    # create the vector pointing from the Rx to the target
    dpr = array([[dxr], [dyr], [dzr]])
    # rotate both of these vectors to the antenna frame
    r_at = Ria.dot(dpt)
    r_ar = Ria.dot(dpr)
    phit = arctan2(-r_at.item(2), r_at.item(0))
    phir = arctan2(-r_ar.item(2), r_ar.item(0))
    thetat = arccos(r_at.item(1) / rt)
    thetar = arccos(r_ar.item(1) / rr)

    ant_el = arcsin(r_at.item(2) / rr)
    ant_az = arctan2(r_at.item(0), r_at.item(1))

    # cttx = cos(thetat)
    sttx = sin(thetat)
    # ctrx = cos(thetar)
    strx = sin(thetar)
    cptx = cos(phit)
    sptx = sin(phit)
    cprx = cos(phir)
    # sprx = sin(phir)
    E_i = sinc(a * k * sttx * cptx / (2 * pi)) * sinc(
        b * k * sttx * sptx / (2 * pi))  # * sqrt(cttx**2 * cptx**2 + sptx**2)
    # compute the Tx array factor
    #    AFtx = 0.0
    #    for i in range(int(N/2)):
    #        AFtx += wN[i]*cos((2*(i+1) - 1)*k*d*sttx*cptx/2) / N
    #    # compute the Rx array factor for the sub-array
    #    AFsub = 0.0
    #    for i in range(int(Nsub/2)):
    #        AFsub += wNsub[i]*cos((2*(i+1) - 1)*k*d*strx*cprx/2) / Nsub
    #    # compute the Elevation array factor
    #    AFel = 0.0
    #    for i in range(int(Nel/2)):
    #        AFel += wNel[i]*cos((2*(i+1) - 1)*k*d*sttx*sptx/2) / Nel

    # combine the element electric field with all of the array factors to obtain
    # the two way normalized radiation pattern
    S_2way = E_i * E_i  # * (AFsub*AFel) * (AFtx*AFel)
    return S_2way / maxAntResp, ant_el, ant_az


def antennaGain(N, wN, Nsub, wNsub, Nel, wNel, a, b, d, lamda):
    """
    Inputs are x, y, and z components of the intertial Range vector, which
    points from the aircraft to the point target (full magnitude, not normalized),
    as well as the rotation matrices from inertial-to-body,
    body-to-perpindicular body, and perpindicular body-to-antenna frames.
    As well as the weights for the Tx array, Rx array, and elevation array and
    the number of elements in the Tx, Rx, and Elevation arrays, and
    width (a) and height (b) of a single element, and the wavenumber.
    """
    k = 2 * pi / lamda
    thetat = 0.0
    thetar = 0.0
    phit = 0.0
    phir = 0.0
    cttx = cos(thetat)
    sttx = sin(thetat)
    ctrx = cos(thetar)
    strx = sin(thetar)
    cptx = cos(phit)
    sptx = sin(phit)
    cprx = cos(phir)
    sprx = sin(phir)
    E_i = sinc(a * k * sttx * cptx / (2 * pi)) * sinc(
        b * k * sttx * sptx / (2 * pi))  # * sqrt(cttx**2 * cptx**2 + sptx**2)
    # compute the Tx array factor
    AFtx = 0.0
    AFtx2 = 0.0
    for i in range(int(N / 2)):
        AFtx += 2 * wN[i]
        AFtx2 += 2 * (wN[i] ** 2)
    # compute the Rx array factor for the sub-array
    AFsub = 0.0
    AFsub2 = 0.0
    for i in range(int(Nsub / 2)):
        AFsub += 2 * wNsub[i]
        AFsub2 += 2 * (wNsub[i] ** 2)
    # compute the Elevation array factor
    AFel = 0.0
    AFel2 = 0.0
    for i in range(int(Nel / 2)):
        AFel += 2 * wNel[i]
        AFel2 += 2 * (wNel[i] ** 2)

    # let's try the solution from the other PDF from online
    DFtx = 0.0
    DFel = 0.0
    DFde = 0.0
    for i in range(int(N / 2)):
        for j in range(int(Nel / 2)):
            DFtx += 2 * (wN[i] * wNel[j]) * (((i * 2 + 1) / 2.0) ** 2)
            DFel += 2 * (wN[i] * wNel[j]) * (((j * 2 + 1) / 2.0) ** 2)
            DFde += 2 * wN[i] * wNel[j]
    DFtx = sqrt(DFtx)
    DFel = sqrt(DFel)

    MTI_DF = 8 * pi ** 2 * d * d * DFtx * DFel / (DFde * lamda ** 2)

    # let's try the solution from the other PDF from online
    DFsub = 0.0
    DFel = 0.0
    DFde = 0.0
    for i in range(int(Nsub / 2)):
        for j in range(int(Nel / 2)):
            DFsub += 2 * (wNsub[i] * wNel[j]) * ((i * 2 + 1) / 2.0) ** 2
            DFel += 2 * (wNsub[i] * wNel[j]) * ((j * 2 + 1) / 2.0) ** 2
            DFde += 2 * wNsub[i] * wNel[j]

    DFsub = sqrt(DFsub)
    DFel = sqrt(DFel)

    SAR_DF = 8 * pi ** 2 * d * d * DFsub * DFel / (DFde * lamda ** 2)

    # combine the element electric field with all of the array factors to obtain
    # the two way normalized radiation pattern
    MTI_D = AFtx ** 2 * AFel ** 2 / (AFtx2 * AFel2)
    SAR_D = AFsub ** 2 * AFel ** 2 / (AFsub2 * AFel2)
    return MTI_D, SAR_D, MTI_DF, SAR_DF

#    # compute the number of pulses until the aft antenna lines up with where the
#    # first was
#    dpcaPulseDelay = radar.header.RxLen / (2*sqrt(antVel.T.dot(antVel)).item(0)) / radar.header.T_r
#    
#    
#    rcdata = rangeCompressData(rawdata, matchedFilter, FastTimeFFTlength, FastTimeIFFTlength,
#                               Nrv, Ncpi, Nchan)
#    cpiSubL = Ncpi/2
#    DopplerWin = kaiser(cpiSubL, 5.0)
#    DopWin = DopplerWin.reshape((cpiSubL,1)).dot(ones((1,Nrv)))
#    # snag fore channel
#    chan3Dop = fftshift(fft(rcdata[:cpiSubL,3,:Nrv]*DopWin,axis=0),axes=0)
#    dInd = int(round(dpcaPulseDelay))
#    chan2Dop = fftshift(fft(rcdata[dInd:dInd+cpiSubL,2,:Nrv]*DopWin,axis=0),axes=0)
##    dInd = int(round(dpcaPulseDelay*2))
##    chan1Dop = fftshift(fft(rcdata[dInd:dInd+cpiSubL,1,:Nrv]*DopWin,axis=0),axes=0)
##    dInd = int(round(dpcaPulseDelay*3))
##    chan0Dop = fftshift(fft(rcdata[dInd:dInd+cpiSubL,0,:Nrv]*DopWin,axis=0),axes=0)
#    
#    # compute phase difference
#    gamma32 = chan3Dop * chan2Dop.conj()
##    gamma21 = chan2Dop * chan1Dop.conj()
##    gamma10 = chan1Dop * chan0Dop.conj()
##    gamma31 = chan3Dop * chan1Dop.conj()
##    gamma30 = chan3Dop * chan0Dop.conj()
#    
#    # calculate the mean phase difference over the clutter
#    # get indices where greater than -120
#    clutterInd = nonzero(20*log10(abs(gamma32)) > -120)
#    cRInd = (clutterInd[1])
#    A = array([ones(cRInd.shape), cRInd, cRInd**2, cRInd**3]).T#, cRInd**4]).T
#    b = angle(gamma32[clutterInd].reshape((cRInd.shape[0], 1))) * 180.0/pi
#    b[b<0] = b[b<0] + 360.0
#    x = pinv(A).dot(b)
#    rArray = arange(Nrv)
#    phaseOverRange = x.item(0) + x.item(1)*rArray + x.item(2)*rArray**2 + x.item(3)*rArray**3# + x.item(4)*rArray**4
#    
#    figure();plot(cRInd, b, '.', rArray, phaseOverRange, 'r')
#    
#    myR = arange(Nrv)
#    myA = array([ones(myR.shape), myR, myR**2, myR**3]).T
#    clutterPhaseBias = myA.dot(x)
#    figure();plot(myR, clutterPhaseBias);title('Clutter Phase Bias vs Range')
#    
#    chan2Dop = chan2Dop * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
##    chan1Dop = chan1Dop * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
##    chan0Dop = chan0Dop * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
#    
#    figure();imshow(20*log10(abs(gamma32.T)),interpolation='nearest',origin='lower')
#    title('Magnitude 3 - 2 (dB)');colorbar();axis('tight')
#    figure();imshow(180/pi*angle(gamma32.T),interpolation='nearest',origin='lower')
#    title('Phase Difference 3 - 2');colorbar();axis('tight')
##    figure();imshow(180/pi*angle(gamma21.T),interpolation='nearest',origin='lower')
##    title('Phase Difference 2 - 1');colorbar();axis('tight')
##    figure();imshow(180/pi*angle(gamma10.T),interpolation='nearest',origin='lower')
##    title('Phase Difference 1 - 0');colorbar();axis('tight')
#    
#    #chan3Data = rcdata[:cpiSubL,3,:Nrv]
#    chan3Data = ifft(fftshift(chan3Dop,axes=0),axis=0)
#    dInd2 = int(round(dpcaPulseDelay))
#    #chan2Data = rcdata[dInd2:dInd2+cpiSubL,2,:Nrv] * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
#    chan2Data = ifft(fftshift(chan2Dop,axes=0),axis=0)
##    dInd1 = int(round(dpcaPulseDelay*2))
##    chan1Data = ifft(fftshift(chan1Dop,axes=0),axis=0)
##    #chan1Data = rcdata[dInd1:dInd1+cpiSubL,1,:Nrv] * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
##    dInd0 = int(round(dpcaPulseDelay*3))
##    chan0Data = ifft(fftshift(chan0Dop,axes=0),axis=0)
##    #chan0Data = rcdata[dInd0:dInd0+cpiSubL,0,:Nrv] * exp(1j*repeat(clutterPhaseBias, cpiSubL, axis=1).T *pi/180)
#    
#    rangeWindow = 3
#    slowWindow = 5
#    kernel = zeros((slowWindow, rangeWindow))
#    # compute the impulse response of the gaussian filter in the 7x7 kernel by first
#    # making the kernel a 2-dimensional delta function (assign the center pixel 1)
#    kernel[slowWindow/2, rangeWindow/2] = 1
#    # now make it a circular kernel by setting the pixels with values less
#    # then the 3rd pixel in from the edge, on the edge
#    kernel = gaussian_filter(kernel,2)
#    #kernel = ones((slowWindow, rangeWindow))
#    # convolve the interferogram with the filter kernel separately
#    #kernel[kernel<kernel[2,0]] = 0
#    
#    gamma = computePhaseDifference(chan3Data, chan2Data, kernel)
#    maxHoldPhase = abs(angle(gamma)*180/pi).max(axis=0)
#    minHoldCoherence = abs(gamma).min(axis=0)
#    figure();plot(maxHoldPhase);title('Max-hold of phase difference between channels during a CPI')
#    figure();imshow(180/pi*angle(gamma.T),interpolation='nearest',origin='lower');colorbar();axis('tight')
#    figure();imshow(abs(gamma.T),interpolation='nearest',origin='lower',cmap='gray');colorbar();axis('tight')
#    #gamma = computePhaseDifference(rcdata[int(round(dpcaPulseDelay*1)), 0, :Nrv]*exp(1j*phaseDiff*pi/180), rcdata[0, 1, :Nrv], 3)
#    #figure();plot(angle(gamma)*180/pi)
#    #figure();plot(abs(gamma))
