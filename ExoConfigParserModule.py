#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:49:49 2019

@author: josh

@purpose: Class for reading the configuration parameters and storing them.
"""
import xml.etree.ElementTree as ET
from numpy import loadtxt

class ExoConfiguration(object):
    
    def __init__(self):
        self.filename = "exoConfig.xml"
        
        e = ET.parse(self.filename).getroot()
        
        # get the input file
        inputFileNode = e.findall('InputFile')
        self.inputFile = 0
        if (inputFileNode):
            self.inputFile = inputFileNode[0].text
        
        # for brevity get the Directories tree
        dirTree = e.findall('Directories')[0]
        
        # get the flight path start and stop points
        self.rawDir = dirTree.findall('RawDirectory')[0].text
        self.debugDir = dirTree.findall('DebugDirectory')[0].text
        self.stanagDir = dirTree.findall('StanagDirectory')[0].text
        self.videoDir = dirTree.findall('VideoDirectory')[0].text
        self.truthDir = dirTree.findall('TruthDirectory')[0].text
        self.dtedDir = dirTree.findall('DTEDDirectory')[0].text
        
        # get the collection information
        colInfoTree = e.findall('CollectionInfo')[0]
        self.year = int(colInfoTree.findall('Year')[0].text)
        self.month = int(colInfoTree.findall('Month')[0].text)
        self.day = int(colInfoTree.findall('Day')[0].text)
        self.colTime = int(colInfoTree.findall('Time')[0].text)
        # formulate the date string
        self.dateString = '%02d%02d%04d' % (self.month, self.day, self.year)
        # formulate the SAR collection name
        self.sarFilename = 'SAR_%s_%06d' % (self.dateString, self.colTime)
        
        # get the processing parameters
        procParamTree = e.findall('ProcParameters')[0]
        self.Ncpi = int(procParamTree.findall('CPILength')[0].text)
        self.nearRangePartialPulsePercent =\
            float(procParamTree.findall('NearRangePartialPulsePercent')[0].text)
        self.farRangePartialPulsePercent =\
            float(procParamTree.findall('FarRangePartialPulsePercent')[0].text)
        self.rangeInterpFactor =\
            int(procParamTree.findall('RangeInterpFactor')[0].text)
        self.dopInterpFactor =\
            int(procParamTree.findall('DopplerInterpFactor')[0].text)
        self.fDelay = float(procParamTree.findall('FDelay')[0].text)
        self.exoBroadeningFactor =\
            float(procParamTree.findall('ExoBroadeningFactor')[0].text)
        self.FAFactor = float(procParamTree.findall('FalseAlarmFactor')[0].text)
        self.truthExists = False
        if ( procParamTree.findall('TruthExistence')[0].text == "True" ):
            self.truthExists = True
        
        # Now parse the Input File
        if (self.inputFile):
            self.collectTimes = loadtxt( self.inputFile, dtype='int32' )
    
    def __str__(self):
        printString =  "||||||||||||||||||||||||||||||||||||||||||||\n"
        printString += "||||||                                ||||||\n"
        printString += "||||||    Processing Configuration    ||||||\n"
        printString += "||||||                                ||||||\n"
        printString += "||||||||||||||||||||||||||||||||||||||||||||\n\n"
        printString += "  RAW Directory: %s\n" % self.rawDir
        printString += "  DEBUG Directory: %s\n" % self.debugDir
        printString += "  STANAG Directory: %s\n" % self.stanagDir
        printString += "  VIDEO Directory: %s\n" % self.videoDir
        printString += "  TRUTH Directory: %s\n" % self.truthDir
        printString += "  DTED Directory: %s\n" % self.dtedDir
        printString += "  Collection Name: %s\n" % self.sarFilename
        printString += "  CPI Length: %d\n" % self.Ncpi
        printString += "  Near Range Partial Pulse Percent: %0.2f %%\n" % \
                       self.nearRangePartialPulsePercent
        printString += "  Far Range Partial Pulse Percent: %0.2f %%\n" %\
            (self.farRangePartialPulsePercent)
        printString += "  Range Interpolation Factor: %d\n" %\
            (self.rangeInterpFactor)
        printString += "  Doppler Interpolation Factor: %d\n" %\
            (self.dopInterpFactor)
        printString += "  FDelay: %d\n" % (self.fDelay)
        printString += "  Exo-Boundary Broadening Factor: %0.1f\n" %\
            (self.exoBroadeningFactor)
        printString += "  False Alarm Factor: %0.1f\n" % (self.FAFactor)
        printString += "  Truth Existence: %d\n\n" % (self.truthExists)
        
        return printString  
        

