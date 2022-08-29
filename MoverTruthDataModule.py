# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:53:14 2019

@author: Josh

@purpose: The class for loading in mover truth data.
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from scipy.io import savemat
from scipy.interpolate import CubicSpline
from STAP_helper import getEffectiveInertialAzimuthAndGraze, computeLatLonConv
from stanaggenerator import StanagGenerator
from DTEDManagerModule import DTEDManager
from ExoConfigParserModule import ExoConfiguration
import datetime as dt


class MoverTruthTrack(object):

    def __init__(self, gpsWeek, Nsam, northing, easting, ele, sec):
        self.gpsWeek = gpsWeek
        self.Nsam = Nsam
        self.northing = northing
        self.easting = easting
        self.ele = ele
        # self.speed = speed
        self.sec = sec

        # Generate a spline of the GPS data
        self.northSpline = CubicSpline(self.sec, self.northing)
        self.eastSpline = CubicSpline(self.sec, self.easting)
        self.eleSpline = CubicSpline(self.sec, self.ele)

    def __str__(self):
        printString = "      MoverTruthTrack Info:\n"
        printString += "      -----------------------------\n"
        printString += f"        gpsWeek: {self.gpsWeek:.2f}\n"
        printString += f"        Nsam: %d\n\n" % self.Nsam

        return printString

    def getPositionVelocity(self, time):
        iPos = zeros((3, 1))
        iVel = zeros((3, 1))

        if self.sec[-1] > time > self.sec[0]:
            iPos = array([
                [self.eastSpline(time).item(0)],
                [self.northSpline(time).item(0)],
                [self.eleSpline(time).item(0)]])
            # Take a position a very small amount earlier to calculate a
            #   velocity
            deltaT = 0.01
            pTime = time - deltaT
            priorPos = array([
                [self.eastSpline(pTime).item(0)],
                [self.northSpline(pTime).item(0)],
                [self.eleSpline(pTime).item(0)]])
            iVel = (iPos - priorPos) / deltaT

        # find the upper and lower indices
        #        uInd = nonzero( self.sec - time > 0 )[ 0 ][ 0 ]
        #        lInd = uInd - 1
        #
        #        # compute the linear interpolation of the northing, easting, elevation
        #        # and speed at the time of the radar
        #        timeRatio = ( time - self.sec[ lInd ] ) \
        #            / ( self.sec[ uInd ] - self.sec[ lInd ] )
        #        iPos[ 1, 0 ] = self.northing[ lInd ] +\
        #            ( self.northing[ uInd ] - self.northing[ lInd ] ) * timeRatio
        #        iPos[ 0, 0 ] = self.easting[lInd] +\
        #            ( self.easting[ uInd ] - self.easting[ lInd ] ) * timeRatio
        #        iPos[ 2, 0 ] = self.ele[ lInd ] +\
        #            ( self.ele[ uInd ] - self.ele[ lInd ] ) * timeRatio
        #        #iSpeed = self.speed[lInd] +\
        #        #    (self.speed[uInd] - self.speed[lInd]) * timeRatio

        #        # now compute the velocity
        #        iVel =\
        #            ( iPos - array(
        #                [ [ self.easting[ lInd ] ],
        #                  [ self.northing[ lInd ] ],
        #                  [ self.ele[ lInd ] ] ] ) ) / ( time - self.sec[ lInd ] )

        return iPos, iVel  # , iSpeed

    def getPositionVelocityTime(self, index):
        """
        Get the position, velocity and time at the sample index
        """
        iPos = zeros((3, 1))
        iSec = 0
        if self.Nsam > index > 0:
            iPos = array([
                [self.easting[index]],
                [self.northing[index]],
                [self.ele[index]]])
            iSec = self.sec[index]
            prevPos = array([
                [self.easting[index - 1]],
                [self.northing[index - 1]],
                [self.ele[index - 1]]])
            tDiff = iSec - self.sec[index - 1]
            iVel = (iPos - prevPos) / tDiff

        return iPos, iVel, iSec


class MoverTruthData(object):
    startWord = uint32(0x12345678)
    stopWord = uint32(0x12346789)

    def __init__(self, filename, latConv=0, lonConv=0, lamda=1, PRF=1):
        # store the longitude and latitude conversion factors
        self.latConv = latConv
        self.lonConv = lonConv
        self.lamda = lamda
        self.PRFhalf = PRF / 2
        # parse the file
        self.filename = filename
        fid = open(filename)
        self.parseFile(fid)
        fid.close()

    def __str__(self):
        retString = "MoverTruthData Info:\n"
        retString += "-----------------------------\n"
        retString += "  numTracks: %d\n\n" % self.numTracks
        for i in range(self.numTracks):
            retString += "    Track %d:\n" % i
            retString += "    --------------------\n"
            retString += self.tracks[i].__str__()

        return retString

    def parseFile(self, fid):
        self.numTracks = int(fromfile(fid, 'uint8', 1, ''))

        self.tracks = [0] * self.numTracks

        for i in range(self.numTracks):
            if (MoverTruthData.startWord
                    != fromfile(fid, 'uint32', 1, '')[0]):
                print("Error! Did not get an expected StartWord while " + \
                      "parsing truth data for track %d" % i)
                break

            gpsWeek = int(fromfile(fid, 'uint16', 1, ''))
            numPoints = int(fromfile(fid, 'uint32', 1, ''))
            lat = fromfile(fid, 'float64', numPoints, '')
            lon = fromfile(fid, 'float64', numPoints, '')
            # If either the lat or lon conversion value is zero, then they need
            #   to be initialized with the middle latitude value between minimum
            #   and maximum
            if self.latConv == 0 or self.lonConv == 0:
                self.latConv, self.lonConv = \
                    computeLatLonConv((lat.min() + lat.max()) / 2)
            ele = fromfile(fid, 'float64', numPoints, '')
            # speed = fromfile(fid, 'float64', numPoints, '')
            sec = fromfile(fid, 'float64', numPoints, '')
            self.tracks[i] = \
                MoverTruthTrack(
                    gpsWeek, numPoints, lat * self.latConv, lon * self.lonConv,
                    ele, sec)

            if (MoverTruthData.stopWord
                    != fromfile(fid, 'uint32', 1, '')[0]):
                print("Error! Did not get an expected StopWord while " + \
                      "parsing truth data for track %d" % i)
                break

        # close the file before returning
        fid.close()

        return

    def getPositionVelocityAtTime(self, radarTime):
        positions = []
        velocities = []

        # loop through the tracks and get their positions and velocities
        for i in range(self.numTracks):
            pos, vel = self.tracks[i].getPositionVelocity(radarTime)
            positions.append(pos)
            velocities.append(vel)

        return positions, velocities

    def getPositionVelocityTime(self, index):
        positions = []
        velocities = []
        times = []

        # loop through the tracks and get their positions and velocities
        for i in range(self.numTracks):
            pos, vel, time = self.tracks[i].getPositionVelocityTime(index)
            positions.append(pos)
            velocities.append(vel)
            times.append(time)

        return positions, velocities, times

    def getRangeDopplerAntennaAnglesForTime(
            self, radarTime, antPos, antVel, effAzI, effGrazeI, boreSightVec):
        # call the routine to get the position and velocity at the radar time
        #   for each track
        positions, velocities = self.getPositionVelocityAtTime(radarTime)
        ranges = zeros(self.numTracks)
        Dopplers = zeros(self.numTracks)
        antEles = zeros(self.numTracks)
        antAzs = zeros(self.numTracks)
        radVels = zeros(self.numTracks)
        tarRadVels = zeros(self.numTracks)

        # rotate the target pointing vector into the gimbal pointing frame
        # generate the rotation matrix
        sA = sin(effAzI)
        cA = cos(effAzI)
        sG = sin(effGrazeI)
        cG = cos(effGrazeI)
        rotItoA = array(
            [[cA, -sA, 0],
             [cG * sA, cG * cA, -sG],
             [sG * sA, sG * cA, cG]])

        # now we can compute the range and Doppler for each track
        for i in range(self.numTracks):
            # target Doppler
            r_i = positions[i] - antPos
            tRange = sqrt(r_i.T.dot(r_i))
            r_i = r_i / tRange
            tarRadVels[i] = r_i.T.dot(velocities[i]).item(0)
            f_t = (2.0 / self.lamda) * tarRadVels[i]
            # Get the effective ineratial azimuth and graze for this target
            tarGrazeI, tarAzI = getEffectiveInertialAzimuthAndGraze(r_i)
            # Need to account for the excess Doppler by simply compensating the
            #   entire range-Doppler map by the phases for the boresight, when
            #   in-fact individual targets are not located at the center
            # Construct a new vector that is a combination of the tarGrazeI and
            #   the antenna effAzI
            cen_r_i = array([
                [cos(tarGrazeI) * sin(effAzI)],
                [cos(tarGrazeI) * cos(effAzI)],
                [-sin(tarGrazeI)]])
            # use the radial velocity to compute the expected Doppler
            # Compute the difference between the vehicle Doppler at the center
            #   of beam versus where the target is
            f_s_tar = -(2.0 / self.lamda) * (r_i.T.dot(antVel)).item(0)
            f_s_cen = -(2.0 / self.lamda) \
                      * (cen_r_i.T.dot(antVel)).item(0)
            tDoppler = f_t + (f_s_tar - f_s_cen)
            #            if( tDoppler > self.PRFhalf * 2 ):
            #                tDoppler -= self.PRFhalf * 2
            #            if( tDoppler < 0 ):
            #                tDoppler += self.PRFhalf * 2
            if tDoppler < -self.PRFhalf * 2:
                tDoppler += self.PRFhalf * 2
            if tDoppler > 0:
                tDoppler -= self.PRFhalf * 2
            radVels[i] = tDoppler * self.lamda / 2.0

            # calculate the antenna azimuth and elevation angles

            r_a = rotItoA.dot(r_i)
            antEle = arcsin(r_a[2, 0])
            antAz = arctan2(r_a[0, 0], r_a[1, 0])

            # save the range, Doppler, and antenna angles
            ranges[i] = tRange
            Dopplers[i] = tDoppler
            antEles[i] = antEle
            antAzs[i] = antAz

        return ranges, Dopplers, radVels, tarRadVels, antEles, antAzs

    def generateSTANAG4607Output(self, year, month, day):
        """
        Generate a STANAG 4607 output containing the truth positions of the
          targets within the scene without the radar information.
        """
        print("Generating STANAG 4607 report for truth.")
        # define the epoch
        epoch = dt.datetime(1980, 1, 6)
        # Get the date string
        dtString = "%02d%02d%04d" % (month, day, year)
        # compute the number of GPS weeks
        gpxDT = dt.datetime.strptime(dtString, '%m%d%Y')
        self.numWeeks = int((gpxDT - epoch).days / 7)
        # now calculate the datetime for the beginning of the week
        begWeek = epoch + dt.timedelta(self.numWeeks * 7)
        print(begWeek)
        gpsWeekDelta = gpxDT - begWeek
        self.gpsWeekSecDiff = gpsWeekDelta.total_seconds()

        # Create the 4607 filename
        stanagFilename = self.filename.replace(".dat", "")

        # Read in the parameters from the Exo-configuration so we can get the
        #   DTED directory name
        config = ExoConfiguration()

        # Instantiate the DTEDManager object
        dtedManager = DTEDManager(config.dtedDir)

        # Create the StanagGenerator object
        sg = StanagGenerator()
        sg.openFileForWriting(stanagFilename)
        flightPlan = "%02d%02d%02d555555" % (month, day, year - 2000)
        missionData = \
            ['SlimSDRMonoP', flightPlan, 255, 'WBBellyPod', year, month, day]
        # Write the dumb mission segment
        sg.writeMissionSegment(missionData)

        # Initialize the CPICounter
        self.NumberOfCPIs = self.tracks[0].Nsam
        for i in range(1, self.NumberOfCPIs):
            dwellSegData = self._getStanagDwellSegmentData(dtedManager, i)
            sg.writeDwellSegment(dwellSegData)
            truthTargetReports = \
                self._getSimpleStanagTargetTruthReportData(i)
            for k in range(self.numTracks):
                sg.writeTargetReport(truthTargetReports[k])
        sg.closeFileForWriting()

        return stanagFilename

    def _getAveragePosVelTime(self, index):
        """
        Returns the average position and velocity of all of the truth target
          tracks available.
        """
        meanPos = zeros((3, 1))
        meanVel = zeros((3, 1))
        meanSec = 0
        for i in range(self.numTracks):
            # Get the position, velocity and time from each track and sum them
            pos, vel, sec = self.tracks[i].getPositionVelocityTime(index)
            meanPos += pos
            meanVel += vel
            meanSec += sec
        # Finalize the computation of the mean
        meanPos /= self.numTracks
        meanVel /= self.numTracks
        meanSec = meanSec / self.numTracks - self.gpsWeekSecDiff - 18

        return meanPos, meanVel, meanSec

    def _getStanagDwellSegmentData(self, dtedManager, index):
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
        existenceMask = uint64(0xFF071FC39E780000)

        dwellData = [0.0] * 31
        dwellData[0] = existenceMask
        # Not sure yet how I'm going to do the revisit index (for now just make
        #   it 0)
        dwellData[1] = 0
        dwellData[2] = index
        dwellData[3] = 0
        if index == self.NumberOfCPIs:
            dwellData[3] = 1
        dwellData[4] = self.numTracks
        # let's grab the geocoordinates and time
        pos, vel, time = self._getAveragePosVelTime(index)
        # Shift the position to the up and right by 5 km
        shiftPos = pos + array([[5], [5], [0]]) * 1e3
        dwellData[5] = time * 1e3

        dwellData[6] = shiftPos.item(1) / self.latConv
        dwellData[7] = shiftPos.item(0) / self.lonConv
        dwellData[8] = shiftPos.item(2) * 1e2
        # use the velocity information to get the track heading and speed and
        # vertical velocity
        # first the heading
        sensorHeading = arctan2(vel.item(0), vel.item(1)) * 180 / pi
        if sensorHeading < 0.0:
            sensorHeading += 360.0
        dwellData[14] = sensorHeading
        # then the ground speed
        groundSpeed = sqrt(vel.item(0) ** 2 + vel.item(1) ** 2) * 1e3
        dwellData[15] = groundSpeed
        # then the vertical velocity
        dwellData[16] = vel.item(2) * 1e1
        # Get the attitude information of the platform
        dwellData[20] = sensorHeading
        dwellData[21] = 0
        dwellData[22] = 0
        # Compute the dwell area. To do this, we need to get the boresight
        #   vector and then do some computations to determine where it is
        #   pointing on the ground. Technically, we should use the DTED to get
        #   that information most accurately. But, maybe an approximation would
        #   be fine so as to avoid computationally expensive DTED searches.
        sceneCen = pos
        dwellData[23] = sceneCen.item(1) / self.latConv
        dwellData[24] = sceneCen.item(0) / self.lonConv
        # Set the range and half beam extent
        rangeExtent = 1e3
        halfBeamExtent = 2.5
        dwellData[25] = rangeExtent / 1e3
        dwellData[26] = halfBeamExtent

        return dwellData

    def _getSimpleStanagTargetTruthReportData(self, index):
        """
        This function returns an array (list) with data in the expected format
          to pass into the StanagGenerator class's generateTargetReport method.
        """
        # This array needs to be 18 long, but only certain ones need to be
        #   filled in with information, and the rest are zeros or don't matter
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
        # call the routine to get the position and velocity at the radar time
        #   for each track
        targetReports = []
        positions, velocities, times = self.getPositionVelocityTime(index)

        # now we can compute the range and Doppler for each track
        for i in range(self.numTracks):
            # Target speed
            speed = sqrt(velocities[i].T.dot(velocities[i]))

            stanagTargetData = [
                i, 39.1, 259.1, 0, 0, 1129, 302.5, 405.7, 12, 0, 0, 0, 0, 0, 0,
                0, 0, 0]
            # get the latitude and longitude from the position
            stanagTargetData[1] = positions[i].item(1) / self.latConv
            stanagTargetData[2] = positions[i].item(0) / self.lonConv
            stanagTargetData[5] = positions[i].item(2)
            # convert the range rate to cm/s and save it to the stanag target 
            #   report
            stanagTargetData[6] = speed.item(0) * 1e2
            # compute and write out the target wrap velocity 
            stanagTargetData[7] = 100 * 1e2
            # Assign a fixed SNR
            stanagTargetData[8] = 100
            # We know that our truth movers are wheeled moving vehicles (code 2)
            stanagTargetData[9] = 9
            # save out all of the uncertainty values (1 std dev)
            stanagTargetData[11] = 0
            # calculate the cross-range std dev using the far range small angle
            # approximation for the azimuth uncertainty and target range
            stanagTargetData[12] = 0
            stanagTargetData[13] = 0
            stanagTargetData[14] = 0
            targetReports.append(stanagTargetData)

        return targetReports

    def getStanagTargetTruthReportData(
            self, targetReportIndex, radarTime, antPos, radar):
        """
        This function returns an array (list) with data in the expected format
          to pass into the StanagGenerator class's generateTargetReport method.
        """
        # This array needs to be 18 long, but only certain ones need to be
        #   filled in with information, and the rest are zeros or don't matter
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
        # call the routine to get the position and velocity at the radar time
        #   for each track
        targetReports = []
        positions, velocities = self.getPositionVelocityAtTime(radarTime)

        # now we can compute the range and Doppler for each track
        for i in range(self.numTracks):
            # target Doppler
            r_i = positions[i] - antPos
            tRange = sqrt(r_i.T.dot(r_i))
            r_i = r_i / tRange
            radVel = r_i.T.dot(velocities[i]).item(0)

            stanagTargetData = [
                targetReportIndex + i, 39.1, 259.1, 0, 0, 1129, 302.5, 405.7,
                12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # get the latitude and longitude from the position
            stanagTargetData[1] = positions[i].item(1) / self.latConv
            stanagTargetData[2] = positions[i].item(0) / self.lonConv
            stanagTargetData[5] = positions[i].item(2)
            # convert the range rate to cm/s and save it to the stanag target 
            #   report
            stanagTargetData[6] = radVel * 1e2
            # compute and write out the target wrap velocity 
            stanagTargetData[7] = \
                (self.lamda * radar.xmlData.PRF / 4.0) * 1e2
            # Assign a fixed SNR
            stanagTargetData[8] = 100
            # We know that our truth movers are wheeled moving vehicles (code 2)
            stanagTargetData[9] = 1
            # save out all of the uncertainty values (1 std dev)
            stanagTargetData[11] = 0
            # calculate the cross-range std dev using the far range small angle
            # approximation for the azimuth uncertainty and target range
            stanagTargetData[12] = 0
            stanagTargetData[13] = 0
            stanagTargetData[14] = 0
            targetReports.append(stanagTargetData)

        return targetReports

    def writeTracksToMATFile(self):
        # Create a dictionary containing all of the track data for each track
        dataDict = {'latitudeConversion': self.latConv,
                    'longitudeConversion': self.lonConv}
        for i in range(self.numTracks):
            dataDict['Mover%d' % (i + 1)] = \
                {'gpsWeek': self.tracks[i].gpsWeek,
                 'numSamples': self.tracks[i].Nsam,
                 'latitude': self.tracks[i].northing / self.latConv,
                 'longitude': self.tracks[i].easting / self.lonConv,
                 'elevation': self.tracks[i].ele,
                 'seconds': self.tracks[i].sec}

        # Save out the dictionary as a .MAT file for ingestion into Matlab
        matFilename = self.filename[:-4] + '.mat'
        savemat(matFilename, dataDict)
