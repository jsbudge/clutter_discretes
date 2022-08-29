# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:50:57 2013

@author: Josh Bradley

@purpose: To make some generator classes that will generate and format the
    STANAG 4607 reports
    
    I'm not really sure the best way to go about this, because I have never
    done it before.  But I would like the code to be as simple,
    understandable, and reusable as possible.  And also very easy to use.
    
    Really, it makes the most sense to implement this as a part of the 
    GMTICommand class because that is where all of this data is already
    stored, and no transfer of the information would be necessary.  But I 
    don't want to do that, because I want the code to be reusable in
    different code and circumstances, and on different data sets.
    As an alternative to this, I could pass in an instance of the GMTICommand
    class to the generator and then access the data that way, but again, that
    would make it slightly less useable because it would require a GMTICommand
    object as input, which is very particular.
    
    It would also make sense to implement as a part of the node classes,
    because then, a particular dwell could hand over control of writing its
    dwell segments to each of its "children targets", and they could write 
    their own target report as a part of the dwell segment. 
    ****(This actually seems like the best idea so far.)****
    
    I was thinking though, that we wanted to write out a separate dwell
    segment for each subaperture, because then it would give the data
    exploitation or interpreting team a detailed record of the geolocation of 
    detections and estimations.  However, I don't currently have a
    subaperture node class, but only a collection node, and gmti target node.
    
    ####################### Method of choice ################################
    I will create another node class for subapertures, which will keep track
    of its own targets within the subaperture dwell.
    #########################################################################
"""
from numpy import uint32, uint8, uint16, uint64, int32, int8, int16, isnan, \
    sum
from stanagdatatypes import SA16, SA32, B16, B32, BA16, BA32, iSA16, iSA32, \
    iB16, iB32, iBA16, iBA32
from struct import unpack
import os
import io


class StanagGenerator(object):
    globalCount = 0
    """Class for generating STANAG 4607 formatted GMTI reports"""
    # define field type arrays with all of the types for the field data
    packetHeaderTypes = [str, uint32, str, uint8, str, uint16, uint8, str,
                         uint32, uint32]
    packetHeaderTypStr = ['c', '>I', 'c', '>B', 'c', '>H', '>B', 'c', '>I',
                          '>I']
    packetHeaderNames = ['VersionID', 'PacketSize', 'Nationality',
                         'Classification', 'ClassificationSystem',
                         'PacketSecurityCode', 'ExerciseIndicator',
                         'PlatformID', 'MissionID', 'JobID']
    segmentHeaderTypes = [uint8, uint32]
    segmentHeaderTypStr = ['>B', '>I']
    segmentHeaderNames = ['SegmentType', 'SegmentSize']
    missionSegmentTypes = [str, str, uint8, str, uint16, uint8, uint8]
    missionSegmentTypStr = ['c', 'c', '>B', 'c', '>H', '>B', '>B']
    missionSegmentNames = ['MissionPlan', 'FlightPlan', 'PlatformType',
                           'PlatformConfiguration', 'Year', 'Month', 'Day']
    freeTextSegmentTypes = [str, str, str]
    freeTextSegmentTypStr = ['c', 'c', 'c']
    freeTextSegmentNames = ['OriginatorID', 'RecipientID', 'FreeText']
    jobDefSegmentTypes = [uint32, uint8, str, uint8, uint8, SA32, BA32, SA32,
                          BA32, SA32, BA32, SA32, BA32, uint8, uint16, uint16,
                          uint16, uint16, uint8, uint16, uint16, BA16, uint16,
                          uint8, uint8, uint8, uint8, uint8]
    jobDefSegmentTypesI = [uint32, uint8, str, uint8, uint8, iSA32, iBA32,
                           iSA32, iBA32, iSA32, iBA32, iSA32, iBA32, uint8,
                           uint16, uint16, uint16, uint16, uint8, uint16,
                           uint16, iBA16, uint16, uint8, uint8, uint8, uint8,
                           uint8]
    jobDefSegmentTypStr = ['>I', '>B', 'c', '>B', '>B', '>I', '>I', '>I',
                           '>I', '>I', '>I', '>I', '>I', '>B', '>H', '>H',
                           '>H', '>H', '>B', '>H', '>H', '>H', '>H',
                           '>B', '>B', '>B', '>B', '>B']
    jobDefSegmentNames = ['JobID', 'SensorIDType', 'SensorIDModel',
                          'TargetFilteringFlag', 'RadarPriority', 'PtALat',
                          'PtALon', 'PtBLat', 'PtBLon', 'PtCLat', 'PtCLon',
                          'PtDLat', 'PtDLon', 'RadarMode',
                          'RevisitInterval', 'AlongTrackUncertainty',
                          'CrossTrackUncertainty', 'TrackAltitudeUncertainty',
                          'TrackHeadingUncertainty', 'SensorSpeedUncertainty',
                          'SlantRangeStdDev', 'CrossRangeStdDev',
                          'LOSVelocityStdDev', 'MDV', 'DetectionProbability',
                          'FalseAlarmDensity', 'ElevationModelUsed',
                          'GeoidModelUsed']
    dwellSegmentTypes = [uint64, uint16, uint16, uint8, uint16, uint32, SA32,
                         BA32, int32, SA32, BA32, uint32, uint32, uint16, BA16,
                         uint32, int8, uint8, uint16, uint16, BA16, SA16, SA16,
                         SA32, BA32, B16, BA16, BA16, SA16, SA16, uint8]
    dwellSegmentTypesI = [uint64, uint16, uint16, uint8, uint16, uint32, iSA32,
                          iBA32, int32, iSA32, iBA32, uint32, uint32, uint16,
                          iBA16, uint32, int8, uint8, uint16, uint16, iBA16,
                          iSA16, iSA16, iSA32, iBA32, iB16, iBA16, iBA16, iSA16,
                          iSA16, uint8]
    dwellSegmentTypStr = ['>Q', '>H', '>H', '>B', '>H', '>I', '>I', '>I', '>I',
                          '>I', '>I', '>I', '>I', '>H', '>H', '>I', '>b', '>B',
                          '>H', '>H', '>H', '>H', '>H', '>I', '>I', '>H', '>H',
                          '>H', '>H', '>H', '>B']
    dwellSegmentNames = ['ExistenceMask', 'RevisitIndex', 'DwellIndex',
                         'LastDwell', 'TargetReportCount', 'DwellTime',
                         'SensorLat', 'SensorLon', 'SensorAlt', 'LatScale',
                         'LonScale', 'AlongTrackUncertainty',
                         'CrossTrackUncertainty', 'AltitudeUncertainty',
                         'SensorTrack', 'SensorSpeed', 'SensorVerticalVel',
                         'SensorTrackUncertainty', 'SensorSpeedUncertainty',
                         'SensorVerticalSpeedUncertainty', 'PlatformHeading',
                         'PlatformPitch', 'PlatformRoll', 'CenterLat',
                         'CenterLon', 'RangeHalfExtent', 'DwellAngleHalfExtent',
                         'SensorHeading', 'SensorPitch', 'SensorRoll', 'MDV']
    targetTypes = [uint16, SA32, BA32, int16, int16, int16, int16, uint16, int8,
                   uint8, uint8, uint16, uint16, uint8, uint16, uint8, uint32,
                   int8]
    targetTypesI = [uint16, iSA32, iBA32, int16, int16, int16, int16, uint16,
                    int8, uint8, uint8, uint16, uint16, uint8, uint16, uint8,
                    uint32, int8]
    targetTypStr = ['>H', '>I', '>I', '>h', '>h', '>h', '>h', '>H', '>b', '>B',
                    '>B', '>H', '>H', '>B', '>H', '>B', '>I', '>b']
    targetNames = ['MTIReportIndex', 'HiResLat', 'HiResLon', 'DeltaLat',
                   'DeltaLon', 'GeodeticHeight', 'LOSVelocity', 'WrapVelocity',
                   'SNR', 'Classification', 'ClassificationProbability',
                   'SlantRangeUncertainty', 'CrossRangeUncertainty',
                   'HeightUncertainty', 'LOSVelocityUncertainty',
                   'TruthTagApplication', 'TruthTagEntity', 'RCS']
    packetHeaderBytes = [2, 4, 2, 1, 2, 2, 1, 10, 4, 4]
    segmentHeaderBytes = [1, 4]
    missionBytes = [12, 12, 1, 10, 2, 1, 1]
    jobDefBytes = [4, 1, 6, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 2, 2, 1,
                   2, 2, 2, 2, 1, 1, 1, 1, 1]
    dwellBytes = [8, 2, 2, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 1, 1,
                  2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 1]
    targetBytes = [2, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 4, 1]
    freeTextBytes = [10, 10]

    def __init__(self):
        super(StanagGenerator, self).__init__()

        # specify file name
        self.packetCounter = 0
        self.filename = 'default_stanag_report_filename.det'
        # open up a binary file buffer for writing
        self.fid = None

        # initialize the size of the packet (in bytes) to the number of bytes 
        # in the packet header and the mission segment 
        #   (including segment header)
        # calculate the byte size of a packet header
        self.packetHeaderByteSize = sum(self.packetHeaderBytes)
        # calculate the byte size of a semengt header
        self.segmentHeaderByteSize = sum(self.segmentHeaderBytes)
        # calculate the byte size of a mission segment
        self.missionSegmentByteSize = sum(self.missionBytes)
        # calculate the byte size of a Job Definition Segment
        self.jobDefSegmentByteSize = sum(self.jobDefBytes)
        # calculate the minimum size of a free text segment
        self.freeTextSegmentMinByteSize = sum(self.freeTextBytes)
        self.fileByteSize = 0

        # get the index for the TargetReportCount
        self.targetReportCountIndex = \
            self.dwellSegmentNames.index('TargetReportCount')

        ########################################
        ## Layout the packet header stuff
        ########################################
        self.version = '31'
        # this is the number of bytes in the entire packet, including the header
        # it seems like it could be a little tricky to figure this part out, but
        # I suspect that I will have the dwell segments write themselves out to 
        # file first, and do this one last, and then just add up the bytes.
        self.packetSize = uint32(self.fileByteSize)
        # the nationality is a digraph (not sure what this means)
        self.nationality = 'US'
        # classification: 1 - top secret, 2 - secret, 3 - confidential, 
        #   4 - restricted, 5 - unclassified
        self.classification = uint8(5)  # E8 (which I think can just be an uint8)
        # this indicates the security system to which the data conforms
        #   (0x20 indicates no security which is '  ')
        self.classSystem = '  '
        # 2 byte flag, that should just be set to 0x0000 to signify no
        #   additional control
        self.code = uint16(0)  # flag
        # this is to indicate if the data is with operation, or exercise,
        #   (real, simulated, synthesized, etc.)
        # 128 stands for exercise, real data
        self.exerciseIndicator = uint8(128)  # E8
        # the tail number of the aircraft (10 bytes, and the unused bytes should
        #   contain spaces, ie. 0x20)
        self.platformID = '     10855'
        # an integer field, assigned by the platform, that uniquely identifies
        #   the mission for the platform
        self.missionID = uint32(1)
        # a platform-assigned number identifying the specific request or task to
        #   which the packet pertains.  The job id is unique within a mission.  
        #   A job id of 0 indicates there is no reference to any specific 
        #   request or task.  But, don't ever put 0, unless you don't want to 
        #   put any dwell segments in the packet.  If job id is 0, then the 
        #   packet cannot contain dwell, hrr, or range-doppler segments
        self.jobID = uint32(1)

        self.maxNumTargetPerDwell = -1
        self.totalNumberOfTargets = 0
        self.numCPIsWithTargets = 0

    def openFileForWriting(self, filename):
        """
        Inputs: str
        Opens a file named "filename".det for writing the STANAG report
        """
        # create the filename
        self.filename = f'{filename}.4607'
        # create the sub-directory if it doesn't already exist
        subDirPath = filename[:filename.rfind('/')]
        if not os.path.exists(subDirPath):
            os.makedirs(subDirPath)
        # open the file for writing binary data
        self.fid = io.open(self.filename, 'wb')

    def openFileForReading(self, filename):
        """
        Inputs: str
        Opens a file named "filename".det for reading in GMTI data from the
        STANAG report
        """
        # store the filename
        self.filename = filename
        # open the file for reading binary data
        self.fid = io.open(self.filename, 'rb')

    def closeFileForWriting(self):
        """Close the file that was being written to"""
        # before closing the file, we need to go back to the beginning of the
        # file and update the packet size field
        #        self.packetSize = uint32(self.fileByteSize)
        #        self.fid.seek(2,0)
        #        self.fid.write(self.packetSize.newbyteorder('>'))
        self.fid.close()

    def closeFileForReading(self, filename):
        """Close the file that was being read from"""
        self.fid.close()

    def _readPacket(self):
        """Takes care of reading in a packet"""
        # we will want to store everything into an empty dictionary
        packet = {'PacketHeader': self._readPacketHeader()}
        # determine the number of bytes left to be read
        bytesToRead = packet['PacketHeader'][1] - self.packetHeaderByteSize

        # as long as there are more bytes to be read, we will keep going through 
        #   this loop
        while bytesToRead > 0:
            # the next 5 bytes belong to a segment header (so read them in)
            if bytesToRead < self.segmentHeaderByteSize:
                print("Error! Bytes to read is less than segment header size.")
                break
            segmentHeader = self._readSegmentHeader()
            # only attempt to read if the segment size does not exceed remaining 
            #   bytes to read
            if bytesToRead < segmentHeader[1]:
                print("Error! Bytes to read is less than segment size.")
                break
            # we need to determine what the next segment is
            #   (1: mission, 2: dwell, 5: job definition)
            if segmentHeader[0] == 1:
                # this is a mission segment
                packet['MissionSegment'] = self._readMissionSegment()
            elif segmentHeader[0] == 2:
                # this is a dwell segment
                packet['DwellSegment'], success = \
                    self._readDwellSegment(bytesToRead)
                if not success:
                    return packet, success
            elif segmentHeader[0] == 5:
                # This is a job definition segment
                packet['JobDefSegment'] = self._readJobDefinitionSegment()
            elif segmentHeader[0] == 6:
                # this is a free text segment
                packet['FreeTextSegment'] = self._readFreeTextSegment(
                    segmentHeader[1])

            # decrement the number of bytes to be read in by the size of the
            #   segment
            bytesToRead -= segmentHeader[1]

        return packet, 1

    def readContents(self):
        """Takes care of reading the GMTI data contents from the STANAG file"""
        # store everything in an empty dictionary
        gmtiData = []

        # get the number of bytes in the file
        bytesInFile = self.fid.seek(0, 2)

        # seek back to the beginning
        self.fid.seek(0)

        # initialize the packet Counter
        firstVersionID = 0
        packetFirst2FieldSize = \
            self.packetHeaderBytes[0] + self.packetHeaderBytes[1]
        while bytesInFile > 0:
            # after a packet has been read in fully, the next set of bytes
            #   should always be another packet, which will contain the version 
            #   ID within first 2 bytes. If these don't match the version then
            #   something is wrong, and if it is blank, then we have probably 
            #   reached the end of file
            if bytesInFile > packetFirst2FieldSize:
                versionID = self.fid.read(self.packetHeaderBytes[0])
                if not firstVersionID:
                    firstVersionID = versionID
                packetSize = unpack(
                    self.packetHeaderTypStr[1],
                    self.fid.read(self.packetHeaderBytes[1]))[0]
                #                print("PacketSize: %d" % packetSize)
                if packetSize > bytesInFile or versionID != firstVersionID:
                    print('We have reached end of usable file.')
                    break
            # seek back by 6 bytes (or the number of bytes used for the first
            #   two fields of the packet header)
            self.fid.seek(-packetFirst2FieldSize, 1)

            # from this point on, we assume there will be a valid packet to read
            #   in and we try to read it
            packet, success = self._readPacket()

            gmtiData.append(packet)
            self.packetCounter += 1
            if not success:
                print("Error! Packet read was a failure. Exiting early.")
                break
            bytesInFile -= packet['PacketHeader'][1]
        #            print('PacketCounter: %d, remainingBytes: %d' %\
        #                 (packetCounter, bytesInFile))

        # Finalize the statistics for average number of targets per CPI
        self.aveNumTargetsPerCPI = \
            float(self.totalNumberOfTargets) / self.numCPIsWithTargets
        print("%d packets were read from file %s" % \
              (self.packetCounter, self.filename))
        print("The maximum number of targets for a dwell: %d" % \
              self.maxNumTargetPerDwell)
        print("Average number of targets per dwell: %d" % \
              self.aveNumTargetsPerCPI)
        return gmtiData

    def _readPacketHeader(self):
        """Takes care of reading in the info for the packet header"""
        return {i: self.fid.read(self.packetHeaderBytes[i]) if self.packetHeaderTypes[i] is str else
        unpack(self.packetHeaderTypStr[i], self.fid.read(self.packetHeaderBytes[i]))[0] for
                i in range(len(self.packetHeaderBytes))}

    def _readSegmentHeader(self):
        """Takes care of reading in the info for a segment header"""
        return [unpack(self.segmentHeaderTypStr[i], self.fid.read(self.segmentHeaderBytes[i]))[0] for
                i in range(len(self.segmentHeaderBytes))]

    def _readMissionSegment(self):
        """Takes care of reading in the data for the mission segment"""
        return {i: self.fid.read(self.missionBytes[i]) if self.missionSegmentTypes[i] is str else
        unpack(self.missionSegmentTypStr[i], self.fid.read(self.missionBytes[i]))[0] for
                i in range(len(self.missionBytes))}

    def _readFreeTextSegment(self, segmentSize):
        """Takes care of reading in the data for a free text segment"""
        data = {}
        # compute the number of bytes in the segment header
        headerNumBytes = sum(self.segmentHeaderBytes[i] for i in range(len(self.segmentHeaderBytes)))

        # initialize the remaining bytes in the free text segment
        remainingBytes = segmentSize - headerNumBytes
        for i in range(len(self.freeTextBytes)):
            data[i] = self.fid.read(self.freeTextBytes[i])
            remainingBytes -= self.freeTextBytes[i]
        # The rest of the bytes in this segment should be text
        data[len(self.freeTextSegmentNames) - 1] = self.fid.read(remainingBytes)
        return data

    def _readJobDefinitionSegment(self):
        """Takes care of reading in the data for the job definition segment"""
        data = {}
        for i in range(len(self.jobDefBytes)):
            if self.jobDefSegmentTypes[i] is str:
                data[i] = self.fid.read(self.jobDefBytes[i])
            else:
                val = self.jobDefSegmentTypesI[i](
                    unpack(
                        self.jobDefSegmentTypStr[i],
                        self.fid.read(self.jobDefBytes[i]))[0])
                data[i] = val
        return data

    def _readDwellSegment(self, bytesToRead):
        """Takes care of reading in the data for the dwell segment"""
        # we will add the segmentHeader to the data array
        # this means the offset into the data array for dwell info is 2
        dwell = {}

        # first read in the existence mask and add it to data, so we can use it
        #   to read in the bytes correctly
        if bytesToRead < self.dwellBytes[0]:
            print("Error! Bytes to read is less than the existence mask.")
            return dwell, 0
        existenceMask = self.dwellSegmentTypesI[0](
            unpack(
                self.dwellSegmentTypStr[0],
                self.fid.read(self.dwellBytes[0]))[0])
        dwell[0] = existenceMask
        leftOverBytes = bytesToRead - self.dwellBytes[0]

        bitMask = 2 ** 63

        # compute the expected size of the base dwell segment dat based on the
        #   existence mask, and the expected size of a single target report
        dwellBaseByteSize = 0
        for i in range(1, len(self.dwellBytes)):
            if existenceMask & uint64(bitMask):
                dwellBaseByteSize += self.dwellBytes[i]
            bitMask >>= 1

        bitMask = 2 ** 33
        targetReportByteSize = 0
        for i in range(len(self.targetBytes)):
            if existenceMask & uint64(bitMask):
                targetReportByteSize += self.targetBytes[i]
            bitMask >>= 1

        # check to see that the base dwell size is not larger than the leftover
        #   bytes
        if (leftOverBytes < dwellBaseByteSize):
            print("Error! Bytes to read is less than the base dwell byte size.")
            return dwell, 0

        bitMask = 2 ** 63
        # read in the dwell segment data
        for i in range(1, len(self.dwellBytes)):
            if existenceMask & uint64(bitMask):
                val = self.dwellSegmentTypesI[i](
                    unpack(
                        self.dwellSegmentTypStr[i],
                        self.fid.read(self.dwellBytes[i]))[0])
                dwell[i] = val
            bitMask >>= 1

        # Update the left over number of bytes
        leftOverBytes -= dwellBaseByteSize

        # grab the number of targets in the dwell
        numTargets = dwell[self.targetReportCountIndex]
        if numTargets > 0:
            self.numCPIsWithTargets += 1
            self.totalNumberOfTargets += numTargets
        if numTargets > self.maxNumTargetPerDwell:
            self.maxNumTargetPerDwell = numTargets

        if numTargets > 50:
            print("packetCounter: %d, numTargets: %d" % (self.packetCounter, numTargets))

        # check to see that the bytes required for the number of targets does
        #   not exceed the number of leftover bytes
        if leftOverBytes < numTargets * targetReportByteSize:
            print("Error! Bytes to read is less than the number of bytes for" + \
                  " the number of targets that need to be read.")
            print("leftOverBytes: %d, numTargets: %d, bytes for targets: %d" % \
                  (leftOverBytes, numTargets, numTargets * targetReportByteSize))
            currentPlace = self.fid.tell()
            print("Bytes left in file: %d" % (
                    self.fid.seek(0, 2) - currentPlace))
            return dwell, 0

        tarData = []
        targetCounter = 0
        while targetCounter < numTargets:
            # read in the data for a target
            targetReport = self._readTargetReport(existenceMask)
            tarData.append(targetReport)
            # decrement the target count
            targetCounter += 1

        # add the tarData to the dwell
        if numTargets > 0:
            dwell['TargetReports'] = tarData

        return dwell, 1

    def _readTargetReport(self, existenceMask):
        """
        Takes care of reading in the data for the target report portion of
        the dwell segment
        """
        data = {}
        bitMask = 2 ** 33
        for i in range(len(self.targetBytes)):
            if existenceMask & uint64(bitMask):
                #                print "index = {}".format(i)
                #                print self.targetTypesI[i]
                #                print self.targetTypStr[i]
                #                print self.targetBytes[i]
                val = self.targetTypesI[i](
                    unpack(
                        self.targetTypStr[i],
                        self.fid.read(self.targetBytes[i]))[0])
                data[i] = val
            bitMask >>= 1
        return data

    def _determineDwellSegmentSize(self, targetCount):
        """Return the size in bytes of the current dwell"""
        # initialize the running sum of dwell size to the size of the segment
        # header (5 bytes) plus the size of the existence mask
        totalDwellSize = 5 + 8

        # now loop through the dwell only fields and add up the bytes
        bitMask = 2 ** 63
        for i in range(1, len(self.dwellBytes)):
            if self.existenceMask & uint64(bitMask):
                totalDwellSize += self.dwellBytes[i]
            bitMask >>= 1

        # loop through the target only fields of the dwell to get bytes per
        #   target
        targetSize = 0
        for i in range(len(self.targetBytes)):
            if self.existenceMask & uint64(bitMask):
                targetSize += self.targetBytes[i]
            bitMask >>= 1
            # calculate the total dwell size
        totalDwellSize += targetSize * targetCount

        return totalDwellSize

    def writePacketHeader(self, packetSize):
        """This functions write out the packet header at the beginning of a new
        segment.  While you can have multiple segments within a packet, that
        just makes life more complicated."""
        # write out the packet header to file
        self.fid.write(self.version.encode('ASCII'))
        self.fid.write(packetSize.newbyteorder('>'))
        self.fid.write(self.nationality.encode('ASCII'))
        self.fid.write(self.classification.newbyteorder('>'))
        self.fid.write(self.classSystem.encode('ASCII'))
        self.fid.write(self.code.newbyteorder('>'))
        self.fid.write(self.exerciseIndicator.newbyteorder('>'))
        self.fid.write(self.platformID.encode('ASCII'))
        self.fid.write(self.missionID.newbyteorder('>'))
        self.fid.write(self.jobID.newbyteorder('>'))

    def writeTargetReport(self, targetData):
        """
        Inputs: python array of target report data (size=18)
        the existence mask for the dwell is used to extract only the data
        fields that are designated bye the user to be written in the STANAG
        report
        """
        # save the target report data
        self.targetData = targetData

        # create a bit mask for extracting fields using the existence mask
        #   NOTE: the target report portion of the dwell segment starts on 
        #   bit 33 (the 34th bit)
        bitMask = 2 ** 33
        # write out the target report data as a part of the dwell segment
        for i in range(len(self.targetTypes)):
            if self.existenceMask & uint64(bitMask):
                val = self.targetTypes[i](self.targetData[i])
                self.fid.write(val.newbyteorder('>'))
            bitMask >>= 1

    def writeDwellSegment(self, dwellData):
        """
        Inputs: python array of dwell data (size=31)
        The existence mask is the first element, and will be saved and used
        to extract only the data fields that are designated by the user to be
        written in the STANAG report.  
        NOTE: this only writes out the initial dwell segment stuff, the 
        target reports will not be written out, but should be written out
        with a call to "writeTargetReport" for each target.
        """
        # save the existence mask
        self.existenceMask = self.dwellSegmentTypes[0](dwellData[0])

        # save the targetCount
        targetCount = dwellData[4]

        # determine the size in bytes of this dwell
        dwellByteSize = self._determineDwellSegmentSize(targetCount)

        # This will be sent out as a new packet, so write a packet header first
        packetSize = self.packetHeaderByteSize + dwellByteSize
        self.writePacketHeader(uint32(packetSize))

        # add that to the running sum for the total byte size of the packet
        self.fileByteSize += packetSize

        # define the dwell segment header
        self.dwellSegmentHeader = [uint8(2), uint32(dwellByteSize)]

        # save the dwell segment field data
        self.dwellData = dwellData

        # write out the mission segment header
        for i in range(2):
            val = self.segmentHeaderTypes[i](self.dwellSegmentHeader[i])
            self.fid.write(val.newbyteorder('>'))

        # first write out the existence mask since its own value doesn't show up 
        #   in the mask
        self.fid.write(self.existenceMask.newbyteorder('>'))
        bitMask = 2 ** 63
        # write out the dwell segment data
        for i in range(1, len(self.dwellSegmentTypes)):
            if self.existenceMask & uint64(bitMask):
                val = self.dwellSegmentTypes[i](self.dwellData[i])
                self.fid.write(val.newbyteorder('>'))
            bitMask >>= 1

        # self.existenceMask =
        # uint64(0b111111110000011100011111111
        # 1111110001100000010000000000000000000)
        # self.existenceMask = uint64(0xFF071FFF8C080000)

    def writeMissionSegment(self, missionData):
        """
        Input: python array of mission fields (size=7)
        missionData = [missionPlan, flightPlan, platformType, platformConfig,
                       year, month, day]
        Set all of the fields for the mission segment
        """
        # declare the mission size (in bytes)
        missionByteSize = 44

        # This will be sent out as a new packet, so write a packet header first
        packetSize = self.packetHeaderByteSize + missionByteSize
        self.writePacketHeader(uint32(packetSize))

        # add to packet size running sum
        self.fileByteSize += packetSize

        # define the mission segment header
        self.missionSegmentHeader = [uint8(1), uint32(missionByteSize)]

        # record the mission field data
        self.missionData = missionData

        # write out the mission segment header
        for i in range(2):
            val = self.segmentHeaderTypes[i](self.missionSegmentHeader[i])
            self.fid.write(val.newbyteorder('>'))

        # write out the mission segment data
        for i in range(len(self.missionSegmentTypes)):
            val = self.missionSegmentTypes[i](self.missionData[i])
            if type(val) is str:
                self.fid.write(val.encode('ASCII'))
            else:
                self.fid.write(val.newbyteorder('>'))


if __name__ == "__main__":
    # generator
    sg = StanagGenerator()
    # sg.openFileForReading(
    #    '/home/josh/Data/STANAG/04012019/SAR_04012019_125612.4607')
    # sg.openFileForReading(
    #    'D:/Data/STANAG/09232019/DetectionUpdated/SAR_09232019_094024.4607')
    # sg.openFileForReading('D:/Data/ARCHIVE/09232019/SAR_09232019_094024.4607')
    # sg.openFileForReading(
    #    '/home/josh/Data/ARCHIVE/09232019/SAR_09232019_093358.4607')
    # sg.openFileForReading('C:/Users/Josh/Downloads/SAR_09232019_094024.4607')
    # filename = 'C:/Users/Josh/Downloads/SAR_09232019_093358.4607'
    # filename = 'D:/Data/ARCHIVE/11122019/SAR_11122019_123345.4607'
    # filename = 'D:/Data/STANAG/09232019/SAR_09232019_094024_truth.4607'
    # filename = 'C:/Users/Josh/Downloads/191212205513.4607'
    filename = '/home/josh/Data/ARCHIVE/04202022/SAR_04202022_140046.4607'
    filename = '/home/josh/Downloads/SAR_06292022_220956.4607'
    sg.openFileForReading(filename)
    gmtiData = sg.readContents()
    sg.closeFileForReading('')
