import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows


def getNextPowerOfTwo(x):
    """ Returns the next power of 2 of x."""
    return int(2 ** (np.ceil(np.log2(abs(x)))))


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
    if sll > 0:
        sll *= -1
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


def GetSimpleMatchedFilter(chan):
    # Get the index at which the chirp actually starts in the reference pulse. The
    #   more accurate this is, the more accurate the ranges will be, otherwise there
    #   will be an offset related to the difference and the sampling rate.
    # NOTE: 120 is pretty close to what it normally is for a 2 GHz sampling rate,
    #   but this is just an approximation.
    chirpStartInd = 120 // int(2e9 / chan.fs)
    # Compute the convolution length and the FFT length to use
    convolution_length_N = chan.cal_chirp.shape[0] + chan.pulse_length_N - 1
    fft_len = getNextPowerOfTwo(convolution_length_N)

    # Calculate the spectrum start and stop frequency
    basebandedCenterFreqHz = chan.baseband_fc
    basebandedChirpRateHzPerS = chan.chirp_rate
    if (chan.NCO_freq_Hz > 0):
        basebandedChirpRateHzPerS *= -1
    halfBandwidthHz = chan.bw / 2.0
    basebandedStartHz = basebandedCenterFreqHz - halfBandwidthHz
    basebandedStopHz = basebandedCenterFreqHz + halfBandwidthHz
    if (basebandedChirpRateHzPerS < 0):
        basebandedStartHz = basebandedCenterFreqHz + halfBandwidthHz
        basebandedStopHz = basebandedCenterFreqHz - halfBandwidthHz
    # Compute the FFT
    fftRefPulse = np.fft.fft(
        chan.cal_chirp[chirpStartInd: chirpStartInd + chan.pulse_length_N],
        fft_len)
    # Calculate the length of the frequency window
    lenFreqWin = int(np.floor(chan.bw * fft_len / chan.fs))
    # Generate the Taylor window
    window = window_taylor(lenFreqWin, 7, -35)
    # If the sign of the start and stop frequency don't match then this crosses
    #   DC, otherwise, it is offset video
    if (np.sign(basebandedStartHz) != np.sign(basebandedStopHz)):
        maxFreqHz = max(basebandedStartHz, basebandedStopHz)
        aboveZeroLength = int(np.ceil(maxFreqHz * fft_len / chan.fs))
        belowZeroLength = lenFreqWin - aboveZeroLength
        # Window the upper spectrum
        fftRefPulse[:aboveZeroLength] *= window[-aboveZeroLength:]
        # Window the lower spectrum
        fftRefPulse[-belowZeroLength:] *= window[:belowZeroLength]
        # Zero out everything else
        fftRefPulse[aboveZeroLength:-belowZeroLength] = 0
    else:
        # Offset video
        minFreqHz = min(basebandedStartHz, basebandedStopHz)
        spectrumMinIndex = \
            int(np.floor(minFreqHz * fft_len / chan.fs))
        fftRefPulse[spectrumMinIndex: spectrumMinIndex + lenFreqWin] *= \
            window
        # Zero out the spectrum before and after
        fftRefPulse[:spectrumMinIndex] = 0
        fftRefPulse[spectrumMinIndex + lenFreqWin:] = 0

    # Conjugate before returning
    return fftRefPulse.conj()


def GetTooSimpleMatchedFilter(chan):
    convolution_length_N = chan.cal_chirp.shape[0] + chan.pulse_length_N - 1
    fft_len = getNextPowerOfTwo(convolution_length_N)

    # Get the index at which the chirp actually starts in the reference pulse. The
    #   more accurate this is, the more accurate the ranges will be, otherwise there
    #   will be an offset related to the difference and the sampling rate.
    # NOTE: 120 is pretty close to what it normally is for a 2 GHz sampling rate,
    #   but this is just an approximation.
    chirp_start_ind = 120 // int(2e9 / chan.fs)
    # Calculate the number of FFT bins wide the spectrum is (divided by two)
    band_sz = int(chan.bw * fft_len / chan.fs) // 2
    # Get the index at which the center of the spectrum should be
    fc_baseband_bin = int(chan.baseband_fc * fft_len / chan.fs) // 2
    simple_match_filter = np.fft.fft(
        chan.cal_chirp[
        chirp_start_ind:chirp_start_ind + chan.pulse_length_N],
        fft_len).conj().T
    # Generate a window for sidelobe control
    upper_len = fc_baseband_bin + band_sz
    lower_len = band_sz * 2 - upper_len
    range_window = window_taylor(band_sz * 2, 5, -35.0)
    # Place the window in the correct place with relation to the spectrum and let
    #   the rest be zeros
    rc_window = np.zeros(fft_len)
    rc_window[:upper_len] = range_window[-upper_len:]
    rc_window[-lower_len:] = range_window[:lower_len]
    # Multiply the simple matched filter by the window
    simple_match_filter *= rc_window
    # Alternatively, you could load in the matched filter created by the parser
    return simple_match_filter


def GetBaseBandWaveformMatchedFilter(chan, nbar=5, SLL=-35):
    # !/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Apr 18 11:10:00 2022
    @brief Parses the calibration data and waveform from binary files and
    generates the matched filter for the pure waveform
    @author: Josh Bradley
    """

    #    plt.close( "all" )

    # Things the PS will need to know from the configuration
    numSamples = chan.nsam
    samplingFreqHz = chan.fs
    basebandedChirpRateHzPerS = chan.chirp_rate
    # If the NCO was positive it means we will have sampled the reverse spectrum
    #   and the chirp will be flipped
    if (chan.NCO_freq_Hz > 0):
        basebandedChirpRateHzPerS *= -1
    halfBandwidthHz = chan.bw / 2.0
    # Get the basebanded center, start and stop frequency of the chirp
    basebandedCenterFreqHz = chan.baseband_fc
    basebandedStartFreqHz = chan.baseband_fc - halfBandwidthHz
    basebandedStopFreqHz = chan.baseband_fc + halfBandwidthHz
    if (basebandedChirpRateHzPerS < 0):
        basebandedStartFreqHz = chan.baseband_fc + halfBandwidthHz
        basebandedStopFreqHz = chan.baseband_fc - halfBandwidthHz

    # Get the reference waveform and mix it down by the NCO frequency and
    #   downsample to the sampling rate of the receive data if necessary
    # The waveform input into the DAC has already had the Hilbert transform
    #   and downsample operation performed on it by SDRParsing, so it is
    #   complex sampled data at this point at the SlimSDR base complex sampling
    #   rate.
    # Compute the decimation rate if the data has been low-pass filtered and
    #   downsampled
    decimationRate = 1
    if chan.is_lpf:
        decimationRate = \
            int(np.floor(chan.BASE_COMPLEX_SRATE_HZ / samplingFreqHz))

    # Grab the waveform
    waveformData = chan.ref_chirp

    #    plt.figure()
    #    plt.title( "Waveform time domain" )
    #    plt.plot( waveformData.real, color = "red", label = "Real" )
    #    plt.plot( waveformData.imag, color = "blue", label = "Imaginary" )
    #    plt.legend( loc = "upper right" )

    # Create the plot for the FFT of the waveform
    waveformLen = len(waveformData)
    waveformFFTLen = getNextPowerOfTwo(waveformLen)
    FFTWaveformData = np.fft.fft(waveformData, waveformFFTLen)
    # Display onlt the real spectrum
    FFTWaveformAxis = \
        np.linspace(0, chan.BASE_COMPLEX_SRATE_HZ, waveformFFTLen)
    #    plt.figure()
    #    plt.title( "Waveform frequency domain" )
    #    plt.plot( FFTWaveformAxis / 1e6, 20 * np.log10( abs( FFTWaveformData ) ) )
    #    plt.xlabel( 'Frequency (MHz)' )
    #    plt.ylabel( 'Power (dB)' )

    # Compute the mixdown signal
    mixDown = \
        np.exp(1j * (2 * np.pi * chan.NCO_freq_Hz * np.arange(waveformLen) \
                     / chan.BASE_COMPLEX_SRATE_HZ))
    basebandWaveform = mixDown * waveformData

    # Decimate the waveform if applicable
    if decimationRate > 1:
        basebandWaveform = basebandWaveform[:: decimationRate]
    # Calculate the updated baseband waveform length
    basebandWaveformLen = len(basebandWaveform)

    # Examine the basebanded and potentially downsampled waveform
    #    plt.figure()
    #    plt.title( "Baseband waveform time domain" )
    #    plt.plot( basebandWaveform.real, color = "red", label = "Real" )
    #    plt.plot( basebandWaveform.imag, color = "blue", label = "Imaginary" )
    #    # Add the legend to the plot
    #    plt.legend( loc = "upper right" )

    # Create the plot for the FFT of the waveform
    #    plt.figure()
    #    plt.title( "Baseband waveform frequency domain" )
    FFTBasebandWaveformData = np.fft.fftshift(np.fft.fft(basebandWaveform))
    FFTBasebandWaveformAxis = np.linspace(
        -1 * samplingFreqHz / 2, samplingFreqHz / 2,
        len(FFTBasebandWaveformData))
    #    plt.plot(
    #        FFTBasebandWaveformAxis / 1e6,
    #        20 * np.log10( abs( FFTBasebandWaveformData ) ) )
    #    plt.xlabel( 'Frequency (MHz)' )
    #    plt.ylabel( 'Power (dB)' )

    # Grab the calibration data
    calData = chan.cal_chirp

    # Calculate the convolution length
    convolutionLength = basebandWaveformLen + basebandWaveformLen - 1
    FFTLength = getNextPowerOfTwo(convolutionLength)

    # Generate the Taylor window
    TAYLOR_NBAR = nbar
    TAYLOR_SLL_DB = SLL
    windowSize = \
        int(np.floor(halfBandwidthHz * 2.0 / samplingFreqHz * FFTLength))
    print('Window Size:%d' % (windowSize))
    taylorWindow = np.ones(windowSize)
    if (SLL != 0):
        taylorWindow = window_taylor(
            windowSize, nbar=TAYLOR_NBAR, sll=TAYLOR_SLL_DB)

    # Create the matched filter and polish up the inverse transfer function
    matchedFilter = np.fft.fft(basebandWaveform, FFTLength)
    # IQ baseband vs offset video
    if np.sign(basebandedStartFreqHz) != np.sign(basebandedStopFreqHz):
        # Apply the inverse transfer function
        aboveZeroLength = \
            int(np.ceil((basebandedCenterFreqHz + halfBandwidthHz) \
                        / samplingFreqHz * FFTLength))
        belowZeroLength = int(windowSize - aboveZeroLength)
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[int(FFTLength / 2) - aboveZeroLength \
                             : int(FFTLength / 2) - aboveZeroLength + windowSize] = taylorWindow
        # Zero out the invalid part of the inverse transfer function
        taylorWindowExtended = np.fft.fftshift(taylorWindowExtended)
        matchedFilter = \
            matchedFilter.conj() * taylorWindowExtended
        #  tempReal = matchedFilter.real * taylorWindowExtended
        #  tempImag = matchedFilter.imag * taylorWindowExtended
        #  matchedFilter = tempReal + tempImag * 1j
    else:
        # Apply the inverse transfer function
        bandStartInd = \
            int(np.floor((basebandedCenterFreqHz - halfBandwidthHz) \
                         / samplingFreqHz * FFTLength))
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[bandStartInd: bandStartInd + windowSize] = \
            taylorWindow
        matchedFilter = \
            matchedFilter.conj() * taylorWindowExtended

    # Plot the matched filter
    FFTBasebandWaveformAxis = \
        np.linspace( \
            -1 * samplingFreqHz / 2, samplingFreqHz / 2, FFTLength)
    #    plt.figure()
    #    plt.title( "Matched filter Magnitude" )
    # Change inf to 0 to show zero line
    matchedFilterMagnitude = \
        20 * np.log10(np.fft.fftshift(abs(matchedFilter)))
    matchedFilterMagnitude[matchedFilterMagnitude == -np.inf] = 0
    #    plt.plot( FFTBasebandWaveformAxis / 1e6, matchedFilterMagnitude )

    return matchedFilter


def GetAdvMatchedFilter(chan, nbar=5, SLL=-35, sar=None, pulseNum=20):
    # Things the PS will need to know from the configuration
    numSamples = chan.nsam
    samplingFreqHz = chan.fs
    basebandedChirpRateHzPerS = chan.chirp_rate
    # If the NCO was positive it means we will have sampled the reverse spectrum
    #   and the chirp will be flipped
    if chan.NCO_freq_Hz > 0:
        basebandedChirpRateHzPerS *= -1
    halfBandwidthHz = chan.bw / 2.0
    # Get the basebanded center, start and stop frequency of the chirp
    basebandedCenterFreqHz = chan.baseband_fc
    basebandedStartFreqHz = chan.baseband_fc - halfBandwidthHz
    basebandedStopFreqHz = chan.baseband_fc + halfBandwidthHz
    if basebandedChirpRateHzPerS < 0:
        basebandedStartFreqHz = chan.baseband_fc + halfBandwidthHz
        basebandedStopFreqHz = chan.baseband_fc - halfBandwidthHz

    # Get the reference waveform and mix it down by the NCO frequency and
    #   downsample to the sampling rate of the receive data if necessary
    # The waveform input into the DAC has already had the Hilbert transform
    #   and downsample operation performed on it by SDRParsing, so it is
    #   complex sampled data at this point at the SlimSDR base complex sampling
    #   rate.
    # Compute the decimation rate if the data has been low-pass filtered and
    #   downsampled
    decimationRate = 1
    if chan.is_lpf:
        decimationRate = int(np.floor(chan.BASE_COMPLEX_SRATE_HZ / samplingFreqHz))

    # Grab the waveform
    waveformData = chan.ref_chirp

    # Create the plot for the FFT of the waveform
    waveformLen = len(waveformData)

    # Compute the mixdown signal
    mixDown = np.exp(1j * (2 * np.pi * chan.NCO_freq_Hz * np.arange(waveformLen) / chan.BASE_COMPLEX_SRATE_HZ))
    basebandWaveform = mixDown * waveformData

    # Decimate the waveform if applicable
    if decimationRate > 1:
        basebandWaveform = basebandWaveform[:: decimationRate]
    # Calculate the updated baseband waveform length
    basebandWaveformLen = len(basebandWaveform)
    # Grab the calibration data
    calData = chan.cal_chirp + 0.0
    # Grab the pulses
    if sar:
        calData = sar.getPulse(pulseNum, channel=0).T + 0.0

    # Calculate the convolution length
    convolutionLength = numSamples + basebandWaveformLen - 1
    FFTLength = getNextPowerOfTwo(convolutionLength)

    # Calculate the inverse transfer function
    FFTCalData = np.fft.fft(calData, FFTLength)
    FFTBasebandWaveformData = np.fft.fft(basebandWaveform, FFTLength)
    inverseTransferFunction = FFTBasebandWaveformData / FFTCalData
    # NOTE! Outside of the bandwidth of the signal, the inverse transfer function
    #   is invalid and should not be viewed. Values will be enormous.

    # Generate the Taylor window
    TAYLOR_NBAR = 5
    TAYLOR_NBAR = nbar
    TAYLOR_SLL_DB = -35
    TAYLOR_SLL_DB = SLL
    windowSize = \
        int(np.floor(halfBandwidthHz * 2.0 / samplingFreqHz * FFTLength))
    taylorWindow = window_taylor(windowSize, nbar=TAYLOR_NBAR, sll=TAYLOR_SLL_DB) if SLL != 0 else np.ones(windowSize)

    # Create the matched filter and polish up the inverse transfer function
    matchedFilter = np.fft.fft(basebandWaveform, FFTLength)
    # IQ baseband vs offset video
    if np.sign(basebandedStartFreqHz) != np.sign(basebandedStopFreqHz):
        # Apply the inverse transfer function
        aboveZeroLength = int(np.ceil((basebandedCenterFreqHz + halfBandwidthHz) / samplingFreqHz * FFTLength))
        belowZeroLength = int(windowSize - aboveZeroLength)
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[int(FFTLength / 2) - aboveZeroLength:int(FFTLength / 2) - aboveZeroLength + windowSize] = \
            taylorWindow
        # Zero out the invalid part of the inverse transfer function
        inverseTransferFunction[aboveZeroLength: -belowZeroLength] = 0
        taylorWindowExtended = np.fft.fftshift(taylorWindowExtended)
    else:
        # Apply the inverse transfer function
        bandStartInd = \
            int(np.floor((basebandedCenterFreqHz - halfBandwidthHz) / samplingFreqHz * FFTLength))
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[bandStartInd: bandStartInd + windowSize] = taylorWindow
        inverseTransferFunction[: bandStartInd] = 0
        inverseTransferFunction[bandStartInd + windowSize:] = 0
    matchedFilter = matchedFilter.conj() * inverseTransferFunction * taylorWindowExtended
    return matchedFilter


def performInterpolatedRangeCompression(
        sar, channelNum, pulseNum, matchedFilter, interpolationFactor):
    # Do not try and grab more pulses than there is data
    numFrames = sar[channelNum].nframes
    if (pulseNum > numFrames):
        return 0

    # Grab the pulses
    data = sar.getPulse(pulseNum, channel=channelNum).T
    # Take the FFT of the data in the fast-time dimension
    FFTLength = len(matchedFilter)
    dataFFT = np.fft.fft(data, FFTLength)
    # Multiply by the advanced matched filter and take the zero-padded IFFT to
    #   finish the range-compression
    if (sar[channelNum].isOffsetVideo):
        rcData = np.fft.ifft(
            matchedFilter * dataFFT, FFTLength * interpolationFactor)
    else:
        # We can't just put zeros at the end in this case because it would be
        #   right in the middle of our signal spectrum
        IFFTInput = np.zeros(
            FFTLength * interpolationFactor, dtype='complex128')
        IFFTInput[:FFTLength] = matchedFilter * dataFFT
        # Move the 2nd half of the FFT data to the last FFTLength / 2 samples of
        #   the zero-padded array and set the values to zero from which we moved
        #   the data
        IFFTInput[-FFTLength // 2:] = \
            IFFTInput[FFTLength // 2: FFTLength]
        IFFTInput[FFTLength // 2: FFTLength] = 0
        # Now we can compute the IFFT
        rcData = np.fft.ifft(IFFTInput)

    return rcData


def performInterpolatedRangeCompressionWaveform(
        sar, channelNum, matchedFilter, interpolationFactor):
    # Things the PS will need to know from the configuration
    chan = sar[channelNum]
    samplingFreqHz = chan.fs
    basebandedChirpRateHzPerS = chan.chirp_rate
    # If the NCO was positive it means we will have sampled the reverse spectrum
    #   and the chirp will be flipped
    if (chan.NCO_freq_Hz > 0):
        basebandedChirpRateHzPerS *= -1

    # Get the reference waveform and mix it down by the NCO frequency and
    #   downsample to the sampling rate of the receive data if necessary
    # The waveform input into the DAC has already had the Hilbert transform
    #   and downsample operation performed on it by SDRParsing, so it is
    #   complex sampled data at this point at the SlimSDR base complex sampling
    #   rate.
    # Compute the decimation rate if the data has been low-pass filtered and
    #   downsampled
    decimationRate = 1
    if chan.is_lpf:
        decimationRate = \
            int(np.floor(chan.BASE_COMPLEX_SRATE_HZ / samplingFreqHz))

    # Grab the waveform
    waveformData = chan.ref_chirp

    # Create the plot for the FFT of the waveform
    waveformLen = len(waveformData)

    # Compute the mixdown signal
    mixDown = \
        np.exp(1j * (2 * np.pi * chan.NCO_freq_Hz * np.arange(waveformLen) \
                     / chan.BASE_COMPLEX_SRATE_HZ))
    basebandWaveform = mixDown * waveformData

    # Decimate the waveform if applicable
    if decimationRate > 1:
        basebandWaveform = basebandWaveform[:: decimationRate]

    # Grab the pulses
    data = basebandWaveform + 0.0
    dataMaxdB = 20 * np.log10(abs(data).max())
    sigmaNoise = 10 ** ((dataMaxdB - 50) / 20) / np.sqrt(2)
    # Let's add noise to the waveform with SNR = -35
    signalNoise = np.random.randn(2, len(data)) * sigmaNoise
    data += signalNoise[0, :] + 1j * signalNoise[1, :]

    # Take the FFT of the data in the fast-time dimension
    FFTLength = len(matchedFilter)
    dataFFT = np.fft.fft(data, FFTLength)
    # Multiply by the advanced matched filter and take the zero-padded IFFT to
    #   finish the range-compression
    if (sar[channelNum].isOffsetVideo):
        rcData = np.fft.ifft(
            matchedFilter * dataFFT, FFTLength * interpolationFactor)
    else:
        # We can't just put zeros at the end in this case because it would be
        #   right in the middle of our signal spectrum
        IFFTInput = np.zeros(
            FFTLength * interpolationFactor, dtype='complex128')
        IFFTInput[:FFTLength] = matchedFilter * dataFFT
        # Move the 2nd half of the FFT data to the last FFTLength / 2 samples of
        #   the zero-padded array and set the values to zero from which we moved
        #   the data
        IFFTInput[-FFTLength // 2:] = \
            IFFTInput[FFTLength // 2: FFTLength]
        IFFTInput[FFTLength // 2: FFTLength] = 0
        # Now we can compute the IFFT
        rcData = np.fft.ifft(IFFTInput)

    return rcData


def performInterpolatedRangeCompressionCal(
        sar, channelNum, matchedFilter, interpolationFactor):
    # Things the PS will need to know from the configuration
    chan = sar[channelNum]
    # Grab the data
    data = chan.cal_chirp

    # Take the FFT of the data in the fast-time dimension
    FFTLength = len(matchedFilter)
    dataFFT = np.fft.fft(data, FFTLength)
    # Multiply by the advanced matched filter and take the zero-padded IFFT to
    #   finish the range-compression
    if (sar[channelNum].isOffsetVideo):
        rcData = np.fft.ifft(
            matchedFilter * dataFFT, FFTLength * interpolationFactor)
    else:
        # We can't just put zeros at the end in this case because it would be
        #   right in the middle of our signal spectrum
        IFFTInput = np.zeros(
            FFTLength * interpolationFactor, dtype='complex128')
        IFFTInput[:FFTLength] = matchedFilter * dataFFT
        # Move the 2nd half of the FFT data to the last FFTLength / 2 samples of
        #   the zero-padded array and set the values to zero from which we moved
        #   the data
        IFFTInput[-FFTLength // 2:] = \
            IFFTInput[FFTLength // 2: FFTLength]
        IFFTInput[FFTLength // 2: FFTLength] = 0
        # Now we can compute the IFFT
        rcData = np.fft.ifft(IFFTInput)

    return rcData
