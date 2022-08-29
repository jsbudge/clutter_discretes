#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:10:40 2019

@author: josh

@purpose: Manage the getting of DTED data
"""
from numpy import *
from numpy.fft import *
from numpy.linalg import *
from osgeo import gdal

class DTEDManager(object):
    
    def __init__(self, dtedDirectory):
        self.dtedDir = dtedDirectory
        
    def getDTEDName(self, lat, lon):
        """Return the path and name of the dted to load for the given lat/lon"""
        tmplat = int( floor(lat) )
        tmplon = int( floor(lon) )
        direw = 'e'
        if (tmplon < 0):
            direw = 'w'
        dirns = 'n'
        if (tmplat < 0):
            dirns = 's'
        
        # record the filename and path of the DTED
        self.dtedName = '%s/%s%03d/%s%02d.dt2' %\
            (self.dtedDir, direw, abs(tmplon), dirns, abs(tmplat))
        
        return self.dtedName
        
    def getDTEDCorrection(self, lat, lon):
        """Returns the ellipsoid correction for the DTED values so that it
        corresponds to the geocoordinates in the GPS."""
        # the filename for the correction data
        filename = "%s/EGM96.DAT" % (self.dtedDir)
        fid = open(filename, 'rb')
        egm96Data = fromfile(fid, 'float64', -1, '')
        egm96Data = egm96Data.reshape((721, 1441), order='C')
        fid.close()
        
        # we want to be able to perform a bilinear interpolation for the data we
        #   get out, so we will get all of the points surrounding our lat/lon 
        #   point
        egN = ceil(lat/0.25) * 0.25
        egS = floor(lat/0.25) * 0.25
        egE = ceil(lon/0.25) * 0.25
        egW = floor(lon/0.25) * 0.25
        egNI = int( (egN + 90.0 + 0.25) / 0.25 ) - 1
        egSI = int( (egS + 90.0 + 0.25) / 0.25 ) - 1
        egEI = int( (egE + 180.0 + 0.25) / 0.25 ) - 1
        egWI = int( (egW + 180.0 + 0.25) / 0.25 ) - 1
        sepInv = 1.0 / ( (egE - egW) * (egN - egS) )
        
        # grab the four data
        eg01 = egm96Data[ egNI, egWI ]
        eg02 = egm96Data[ egSI, egWI ]
        eg03 = egm96Data[ egNI, egEI ]
        eg04 = egm96Data[ egSI, egEI ]
        
        egc = sepInv * ( eg02 * (egE - lon) * (egN - lat) \
            + eg04 * (lon - egW) * (egN - lat) \
            + eg01 * (egE - lon) * (lat - egS) \
            + eg03 * (lon - egW) * (lat - egS) )
        
        return egc
    
    def getInterpolatedDTEDGrid(self, eastings, northings, dtedName, correction, 
                                lonConv, latConv):
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
        
        # only if these x and y indices are within the bounds of the DTED, get
        #   the raster band and try to read in the DTED values
        dtedInterp = ones_like(eastings) * 1e-20
        if ( (minX >= 0 and maxX < ds.RasterXSize) and 
             (minY >= 0 and maxY < ds.RasterYSize) ):
            rasterBand = ds.GetRasterBand(1)
            xSize = int(maxX - minX)
            ySize = int(maxY - minY)
            dtedData = rasterBand.ReadAsArray(
                int(minX), int(minY), xSize, ySize)
            
            # use nearest-neighbor interpolation initially to get the elevation
            #   for the pixels
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
                ( dtedData[floor(py).astype('int'), floor(px).astype('int')] \
                    * rightLonDiff * lowLatDiff \
                + dtedData[floor(py).astype('int'), ceil(px).astype('int')] \
                    * leftLonDiff * lowLatDiff \
                + dtedData[ceil(py).astype('int'), floor(px).astype('int')] \
                    * rightLonDiff * upLatDiff \
                + dtedData[ceil(py).astype('int'), ceil(px).astype('int')] \
                    * leftLonDiff * upLatDiff )
        
        return dtedInterp + correction
    
    def getDTEDPoint(self, lat, lon):
        """Returns the digirtal elevation value closest to a latitude and 
        longitude"""
        dtedName = self.getDTEDName(lat, lon)
        correction = self.getDTEDCorrection(lat, lon)
        correction = 0
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
        
        # only if these x and y indices are within the bounds of the DTED, get 
        #   the raster band and try to read in the DTED values
        elevation = -1e20
        if ( (px >= 0 and px < ds.RasterXSize) and 
             (py >= 0 and py < ds.RasterYSize) ):
            rasterBand = ds.GetRasterBand(1)
            elevation = rasterBand.ReadAsArray(int(px), int(py), 1, 1)
            
        return elevation.item(0) + correction
        
    def getInterpolatedDTED(self, lat, lon):
        """Returns the digital elevation for a latitude and longitude"""
        dtedName = self.getDTEDName(lat, lon)
        correction = self.getDTEDCorrection(lat, lon)
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
        
        # only if these x and y indices are within the bounds of the DTED, get 
        #   the raster band and try to read in the DTED values
        elevation = -1e20
        if ( (px >= 0 and px < ds.RasterXSize) and 
             (py >= 0 and py < ds.RasterYSize) ):
            rasterBand = ds.GetRasterBand(1)
            dtedData = rasterBand.ReadAsArray(int(px), int(py), 2, 2)
            
            # use bilinear interpolation to get the elevation for the lat/lon
            leftLon = int(px) * xres + ulx
            upLat = int(py) * yres + uly
            
            # pre compute the differences for the bilinear interpolation
            rightLonDiff = (leftLon + xres) - lon
            upLatDiff = upLat - lat
            #lowLatDiff = lat - lowLat
            leftLonDiff = lon - leftLon
            lowLatDiff = lat - (upLat + yres)
            #upLatDiff = (lowLat + yres) - lat
            
            elevation = elevSpacInv * ( 
                dtedData[0, 0] * rightLonDiff * lowLatDiff +\
                dtedData[0, 1] * leftLonDiff * lowLatDiff +\
                dtedData[1, 0] * rightLonDiff * upLatDiff +\
                dtedData[1, 1] * leftLonDiff * upLatDiff )
            
        return elevation + correction
    

