# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:27:35 2013

@author: Josh Bradley

@purpose: implement some of the datatypes associated with the STANAG 4607
    GMTI format.  In particular, the B16, B32, H32, BA16, BA32, SA16,
    and SA32.
"""
from numpy import uint8, uint16, uint32, int16, int32, float64

class abstractSignedBinaryDecimal(object):
    """
    Note that this implementation of the numerical types works as long as
    it is given proper input (or input that doesn't exceed the value it is
    capable of representing)
    """
    _fracbits = 0
    _intbits = 0
    _type = 'abstractSignedBinaryDecimal'
    intInt = int
    fracInt = int
    repInt = int
    
    def __init__(self, value=0, inverse=False):
        """Input: value can be anyform of int or float"""
        super(abstractSignedBinaryDecimal,self).__init__()
        denominator = float64(2**self._fracbits)
        if not inverse:
            self._val = float64(value)
            if self._val < 0:
                self._sign = -1
            else:
                self._sign = 1
            self._int = self.intInt(abs(self._val))
            fraction = float64(abs(self._val) - self._int)
            self._frac = self.fracInt(round(fraction * denominator))
            self._newval = self._sign * (self._int + self._frac/denominator)
            self._formBitRepresentation()
        else:
            self.bits = self.repInt(value)
            self._parseBits()
            self._newval = self._sign * (self._int + self._frac/denominator)
            self._val = self._newval
        
    def _formBitRepresentation(self):
        """
        Returns an unsigned integer of the proper size with the correct
        bit representation for the numerical type
        """
        binrep = 0
        # if it is negative, then assign the MSB 1
        if self._sign < 0:
            binrep += 1
        # shift the bits left by the number of integer bits (_intbits)
        binrep = binrep << self._intbits
        # now bitwise OR this result with the integer portion
        binrep = binrep | self._int
        # shift the bits left by the number of fractional bits (_fracbits)
        binrep = binrep << self._fracbits
        # now bitwise OR this result with the fraction portion
        binrep = binrep | self._frac
        self.bits = self.repInt(binrep)
        
    def _parseBits(self):
        """Parses the bits into the proper intBits, fracBits, and signBits"""
        # set an extent mask that goes up to the highest bit representable for any of the types (ie. 32 bits)
        # this will be used for creating the value masks for extracting the values from the shifted bits
        extentMask = 0xFFFFFFFF
        bits = self.bits
        # first, we need to create the value mask for the fractional part
        valueMask = ((extentMask << self._fracbits) ^ extentMask) & extentMask
        # now take a bitwise AND between this value mask and the bits to get the fractional part
        self._frac = self.fracInt(bits & valueMask)
        # next, we need to create the value mask for the integer part (note that the bits will be
        # shifted right by the number of frac bits)
        valueMask = ((extentMask << self._intbits) ^ extentMask) & extentMask
        # shift the bits right by the number of frac bits
        bits = bits >> self._fracbits
        # now take a bitwise AND between the value mask and the bits to get the integer part
        self._int = self.intInt(bits & valueMask)
        # now shift the bits right once more by the number of integer bits, and this
        # should leave us with the sign bit
        bits = bits >> self._intbits
        if bits:
            self._sign = -1
        else:
            self._sign = 1
    
    def __str__(self):
        return 'Set value: %f\nNew value: %f\nBinary: 0x%08x' % \
        (self._val,self._newval,self.bits)
        
    def __repr__(self):
        return '%s(%f)' % (self._type,self._newval)
        #return self.bits
    
    def newbyteorder(self, order):
        """
        Return the bits in a new byte order (a wrapper for the newbyteorder
        method of the underlying numerical type)
        """
        return self.bits.newbyteorder(order)
        
class B16(abstractSignedBinaryDecimal):
    _fracbits = 7
    _intbits = 8
    _type = 'B16'
    intInt = uint8
    fracInt = uint8
    repInt = uint16
    
class iB16(B16):
    """
    I need to make a wrapper class around B16 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iB16, self).__init__(value, True)
        
class B32(abstractSignedBinaryDecimal):
    _fracbits = 23
    _intbits = 8
    _type = 'B32'
    intInt = uint8
    fracInt = uint32
    repInt = uint32
    
class iB32(B32):
    """
    I need to make a wrapper class around B32 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iB32, self).__init__(value, True)
               
class H32(abstractSignedBinaryDecimal):
    _fracbits = 16
    _intbits = 15
    _type = 'H32'
    intInt = uint16
    fracInt = uint16
    repInt = uint32
    
class iH32(H32):
    """
    I need to make a wrapper class around H32 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iH32, self).__init__(value, True)
    
class abstractBinaryAngle(object):
    """
    Abstract class for representing the binary angle numerical format (as
    well as the signed cases)
    
    Note that this implementation of the numerical types works as long as
    it is given proper input (or input that doesn't exceed the value it is
    capable of representing)
    """
    _numBits = 0
    _type = 'abstractBinaryAngle'
    specInt = int
    _maxAngle = 360.0
    
    def __init__(self, value=0, inverse=False):
        """Input: value can be anyform of int or float"""
        super(abstractBinaryAngle,self).__init__()
        if not inverse:
            # store the original value
            self._setVal(value)
            # define the scale value
            scale = 2.0**self._numBits / self._maxAngle
            # compute the new bit representation
            self.bits = self.specInt(round(self._val * scale))
            # compute the new value as a float
            self._newval = self.bits / scale
        else:
            # record the bits
            self.bits = self.specInt(value)
            # define the scale value
            scale = self._maxAngle / (2.0**self._numBits)
            # compute the value as a float
            self._newval = self.bits * scale
            self._val = self._newval
            
    def _setVal(self, value):
        """Sets the _val parameter correctly"""
        # if the value is less than 0, fix it to be in range 0-360
        value = value % self._maxAngle
        self._val = float64(value)
            
    def __str__(self):
        return 'Set value: %f\nNew value: %f\nBinary: 0x%08x' % \
        (self._val,self._newval,self.bits)
        
    def __repr__(self):
        return '%s(%f)' % (self._type,self._newval)
        #return self.bits
        
    def newbyteorder(self, order):
        """
        Return the bits in a new byte order (a wrapper for the newbyteorder
        method of the underlying numerical type)
        """
        return self.bits.newbyteorder(order)
                  
class BA16(abstractBinaryAngle):
    """Input values will be represented in the range 0-360"""
    _numBits = 16
    _type = 'BA16'
    specInt = uint16
    
class iBA16(BA16):
    """
    I need to make a wrapper class around BA16 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iBA16, self).__init__(value, True)
        
class BA32(abstractBinaryAngle):
    """Input values will be represented in the range 0-360"""
    _numBits = 32
    _type = 'BA32'
    specInt = uint32
    
class iBA32(BA32):
    """
    I need to make a wrapper class around BA32 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iBA32, self).__init__(value, True)
        
class SA16(abstractBinaryAngle):
    """Values should be in the range -90:90"""
    _numBits = 16
    _type = 'SA16'
    specInt = int16
    _maxAngle = 180.0
    
    def _setVal(self, value):
        """Sets the _val parameter correctly"""
        # for the signed binary angle, we don't need to correct for the sign
        self._val = float64(value)
        
class iSA16(SA16):
    """
    I need to make a wrapper class around SA16 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """    
    def __init__(self, value=0):
        super(iSA16, self).__init__(value, True)
        
class SA32(abstractBinaryAngle):
    """Values should be in the range -90:90"""
    _numBits = 32
    _type = 'SA32'
    specInt = int32
    _maxAngle = 180.0
    
    def _setVal(self, value):
        """Sets the _val parameter correctly"""
        # for the signed binary angle, we don't need to correct for the sign
        self._val = float64(value)
        
class iSA32(SA32):
    """
    I need to make a wrapper class around SA32 to implement the inverse
    construction of the BA16 numerical type by just reimplementing __init__
    """
    def __init__(self, value=0):
        super(iSA32, self).__init__(value, True)
