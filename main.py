import numpy as np
from platform_helper import RadarPlatform, SDRPlatform
from data_helper import AnimationManager
from SDRParsing import SDRParse
from scipy.signal.windows import taylor
from movlib import getDopplerLine, computeExoMDV, detectExoClutterMoversRVMap, getExoClutterDetectedMoversRVBlob, \
    getStanagDwellSegmentData
from simulation_functions import findPowerOf2, db, enu2llh, getElevation, loadMatchedFilter, loadReferenceChirp, \
    getRawData, getElevationMap, llh2enu
from ExoConfigParserModule import ExoConfiguration
from matched_filters import GetAdvMatchedFilter, window_taylor
import cupy as cupy
from tqdm import tqdm
from stanaggenerator import StanagGenerator
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from targetlib import TrackManager
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

# Constants
c0 = 299792458.0
kb = 1.3806503e-23
T0 = 290.0

# Conversions
DTR = np.pi / 180

# Settings
# sar_dir = '/home/jeff/repo/SAR_DATA/06152022/'
sar_dir = '/data5/SAR_DATA/2022/06152022/'
sar_fnme = sar_dir + 'SAR_06152022_145909.sar'
debug_dir = '/data5/ClutterDiscrete_GMTI_Data/DEBUG/06152022/'
# debug_dir = '/home/jeff/repo/Debug/'
gps_fnme = \
    debug_dir + 'SAR_06152022_145909_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat'
mf_fnme = debug_dir + 'SAR_06152022_145909_Channel_1_MatchedFilter.dat'
ref_fnme = debug_dir + 'SAR_06152022_145909_Channel_1_X-Band_9_GHz_VV_Waveform.dat'
data_fnme = [
    debug_dir + 'SAR_06152022_145909_Channel_1_X-Band_9_GHz_VV_RawData.dat',
    debug_dir + 'SAR_06152022_145909_Channel_2_X-Band_9_GHz_VV_RawData.dat']
gimbal_fnme = [
    debug_dir + 'SAR_06152022_145909_Channel_1_X-Band_9_GHz_VV_GimbalData.dat',
    debug_dir + 'SAR_06152022_145909_Channel_2_X-Band_9_GHz_VV_GimbalData.dat']
csv_fnme = './test.csv'
do_stanag = False
do_video = True
max_cpi_count = 100

# Other settings are contained in the XML file in this same directory, grab them
config = ExoConfiguration()
cpi_len = config.Ncpi
nearRange_partial_pulse_percent = config.nearRangePartialPulsePercent
partial_pulse_percent = config.farRangePartialPulsePercent
upsample = config.rangeInterpFactor
dop_upsample = config.dopInterpFactor
Kexo = config.exoBroadeningFactor
Pfa = config.FAFactor
fDelay = config.fDelay
TRUTH_EXISTS = config.truthExists

print('Loading SAR file...')
sdr_f = SDRParse(sar_fnme)

print('Getting platform data...')
rp = SDRPlatform(sdr_f, debug_fnme=gps_fnme, gimbal_debug=gimbal_fnme[0])
figplotter = AnimationManager()

# Grab the STANAG generator for STANAG generation
sg = StanagGenerator()
nsam = rp.calcNumSamples(0.)
with open(data_fnme[0], 'rb') as fid:
    nframes = np.fromfile(fid, 'uint32', 1, '')[0]
fft_len = findPowerOf2(nsam + rp.calcPulseLength(0, use_tac=True))
myDopWindow = window_taylor(cpi_len * dop_upsample, 11, -70)
slowTimeWindow = myDopWindow.reshape((cpi_len * dop_upsample, 1)).dot(np.ones((1, nsam * upsample)))
myRanges = rp.calcRangeBins(0., upsample=upsample, partial_pulse_percent=nearRange_partial_pulse_percent)
midRange = myRanges.mean()
offset_shift = int(5e6 / (1 / fft_len * sdr_f[0].fs))
taywin = int(sdr_f[0].bw / sdr_f[0].fs * fft_len)
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taytay = taylor(taywin, 11, 70)
wavelength = c0 / sdr_f[0].fc
lamda = wavelength
total_cpi_count = nframes // cpi_len
rad_vel_res = rp.calcRadVelRes(cpi_len)

# Compute the separation between antenna phase centers
antSep = abs(sdr_f.port[sdr_f[0].rec_num].x - sdr_f.port[sdr_f[1].rec_num].x)
cpi_time = cpi_len / sdr_f[0].prf

# Initialize a low-pass filter version of the noisePowerLPF
LPFCutoffFreqHz = 0.25
LPFTimeConstant = 1 / (2 * np.pi * LPFCutoffFreqHz)
LPFDecayFactor = 1 - np.exp(-cpi_time / LPFTimeConstant)
noisePowerLPF = 0
noisePowerVarLPF = 0.02
timeSinceValidCPI = 0
RFICounter = 0
noisePowers = []
noisePowerLPFs = []
noisePowerVarLPFs = []

# Initialize the tracker
# Measurement Noise Covariance
meas_cov = np.diag(np.array([2., 3.5, .2, .2, .2, .2, .2, .2, .5, .5])**2)
tracker = TrackManager(deadtrack_time=2., R=meas_cov)

# Chirp and matched filter calculations
# chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, rp.fs, fc,
#                             bwidth) * 1e4, up_fft_len)
mfilts = np.zeros((sdr_f.n_channels, fft_len, cpi_len), dtype=np.complex128)
for chan in range(sdr_f.n_channels):
    # chirp = np.fft.fft(np.mean(sdr_f.getPulses(np.arange(200), chan, is_cal=True), axis=1), fft_len)
    mfilt = GetAdvMatchedFilter(sdr_f[chan])
    # mfilt = chirp.conj()
    # mfilt[:taywin // 2 + offset_shift] *= taytay[taywin // 2 - offset_shift:]
    # mfilt[-taywin // 2 + offset_shift:] *= taytay[:taywin // 2 - offset_shift]
    # mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0
    mfilts[chan, :, :] = np.tile(mfilt, (cpi_len, 1)).T
mfilt_gpu = cupy.array(mfilts, dtype=np.complex128)
slowTimeWithChannels = np.zeros((sdr_f.n_channels, nsam * upsample, cpi_len * dop_upsample), dtype=np.complex128)
slowTimeWithChannels[0, ...] = slowTimeWindow.T
slowTimeWithChannels[1, ...] = slowTimeWindow.T
slow_time_gpu = cupy.array(slowTimeWithChannels, dtype=np.complex128)
fig = None
plotdata = []
cpi_times = []

with open(csv_fnme, 'w') as csv:
    idx_t = np.arange(nframes)
    cpi_count = 0
    detectionCount = 0
    for tidx in tqdm([idx_t[pos:pos + cpi_len] for pos in range(0, nframes, cpi_len)]):

        ts = sdr_f[0].pulse_time[tidx]
        tmp_len = len(ts)
        if tmp_len < cpi_len:
            break
        cpi_count += 1

        timeSinceValidCPI += cpi_time

        # Create an empty array to which we will append the data for a row
        cpiData = []

        # Get the position of the airplane and calculate the boresight Vec
        antPos = rp.txpos(ts[0])
        antVel = rp.vel(ts[0])
        boresightVec = rp.boresight(ts[0])
        effAzI = rp.pan(ts[0])
        effGrazeI = rp.tilt(ts[0])
        headingI = rp.heading(ts[0])  # np.arctan2(antVel[0], antVel[1])

        """ Detection ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"""
        # Detect the movers in the exo-clutter. In the exo-clutter, in the absence
        #   of a moving target, the radar returns are expected to be noise. As we
        #   have a pretty good idea of the noise figure of the radar, we can use
        #   this to set a threshold based on a false alarm rate.
        # Remember that we added in extra noise to hide the effects of the Hilbert
        #   Transform

        dopCenLine, dopUpLine, dopDownLine, grazeOverRanges = getDopplerLine(
            effAzI, myRanges, antVel, antPos, effGrazeI, rp.az_half_bw, sdr_f[0].prf,
            wavelength, rp.origin)
        radVelErrors = abs(dopUpLine - dopDownLine) * wavelength / 2.0
        # The high-fidelity Doppler line is based on the high-fidelity grazing
        #   angle computation. Which should probably be reserved probably for when
        #   doing either GMTI Backprojection or when better precision is needed for
        #   the slow-time phase compensation.
        #    (dopCenLine, dopUpLine, dopDownLine, grazeOverRanges,
        #         surfaceHeights, numIterations) = getDopplerLineHiFi( effAzI, myRanges,
        #            antVel, antPos, radar, dtedName, dtedCorrection, lamda,
        #            azNullBeamwidthHalf, PRF)

        """Range compress the data from each channel for the CPI"""
        slowtimes = \
            np.arange(cpi_len * dop_upsample).reshape((1, cpi_len * dop_upsample)) / sdr_f[0].prf
        slowtimePhases = np.zeros((sdr_f.n_channels, nsam * upsample, cpi_len * dop_upsample), dtype=np.complex128)
        for ch in range(sdr_f.n_channels):
            slowtimePhases[ch, ...] = np.exp(1j * 2 * np.pi * -dopCenLine.reshape(nsam * upsample, 1).dot(slowtimes))
        phases_gpu = cupy.array(slowtimePhases, dtype=np.complex128)
        #    slowtimePhases = 1.0

        # Python call for range compression of the CPI data
        data = np.zeros((sdr_f.n_channels, nsam, cpi_len), dtype=np.complex128)
        for ch in range(sdr_f.n_channels):
            # data[ch, ...], _, _, _ = getRawData(data_fnme[ch], cpi_len, start_pulse=tidx[0], isIQ=True)
            data[ch, ...] = sdr_f.getPulses(tidx, ch)
        data_gpu = cupy.array(data, dtype=np.complex128)
        rcdata_gpu = cupy.fft.fft(data_gpu, fft_len, axis=1) * mfilt_gpu
        upsample_data_gpu = cupy.zeros((sdr_f.n_channels, fft_len * upsample, cpi_len), dtype=np.complex128)
        upsample_data_gpu[:, :fft_len // 2, :] = rcdata_gpu[:, :fft_len // 2, :]
        upsample_data_gpu[:, -fft_len // 2:, :] = rcdata_gpu[:, -fft_len // 2:, :]
        rcdata_gpu = cupy.fft.ifft(upsample_data_gpu, axis=1)[:, :nsam * upsample, :]
        cupy.cuda.Device().synchronize()

        # Doppler FFT
        upsample_dopdata_gpu = cupy.zeros((sdr_f.n_channels, nsam * upsample, cpi_len * dop_upsample),
                                          dtype=np.complex128)
        upsample_dopdata_gpu[:, :, :cpi_len] = rcdata_gpu
        dopdata_gpu = cupy.fft.fft(upsample_dopdata_gpu * slow_time_gpu * phases_gpu, cpi_len * dop_upsample, axis=2)
        # dopdata_gpu = cupy.fft.fft(rcdata_gpu, cpi_len * dop_upsample, axis=2)

        cupy.cuda.Device().synchronize()

        # Grab data off of GPU
        rcdopdata = dopdata_gpu.get()

        # First, some memory management
        del rcdata_gpu
        del data_gpu
        del upsample_data_gpu
        del upsample_dopdata_gpu
        del dopdata_gpu
        del phases_gpu
        cupy.cuda.MemoryPool().free_all_blocks()

        # Now compute the sum of the channels and the phase
        chanSum = np.sum(rcdopdata, axis=0)
        magData = abs(chanSum)
        antAz = np.arcsin(np.angle(rcdopdata[0, ...].conj() * rcdopdata[1, ...]) * lamda / (2 * np.pi * antSep))
        #    antAz = zeros_like( antAz )
        #    monoPhase = angle( rcdopdata[ 0 ].conj() * rcdopdata[ 1 ] )

        # Compute the upper and lower MDV along with the wrap velocity
        wrapVel = sdr_f[0].prf * lamda / 4.0
        MDV, MDVapproach, MDVrecede = computeExoMDV(
            antVel, rp.az_half_bw, grazeOverRanges[-1], effAzI,
            headingI, Kexo)
        threshVel = max(MDV, rad_vel_res)
        myDopps = np.fft.fftshift(np.linspace(-wrapVel, wrapVel, dop_upsample * cpi_len))

        (detMap, noisePower, thresh) = detectExoClutterMoversRVMap(
            magData, threshVel, -threshVel, Pfa, rp.calcWrapVel())
        # Store the noise powers
        noisePowers.append(noisePower)
        if not noisePowerLPF:
            noisePowerLPF = 10 * np.log10(noisePower)
            noisePowerVarLPF = 0.02
            timeSinceValidCPI = 0
        else:
            noisePowerdB = 10 * np.log10(noisePower)
            # Only update the LPF if a valid CPI was encountered
            if cpi_count + 1 < 100000:
                if noisePowerdB < noisePowerLPF + 1.0:
                    LPFDecayFactor = 1 - np.exp(-timeSinceValidCPI / LPFTimeConstant)
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
                    # print("CPI with likely RFI encountered.")
                    RFICounter += 1
            else:
                if noisePowerdB < noisePowerLPF + 5.0 * np.sqrt(noisePowerVarLPF):
                    LPFDecayFactor = 1 - np.exp(-timeSinceValidCPI / LPFTimeConstant)
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
        # if cpi_count % 2 == 0:
        tracker.propogate(ts[0])
        targetList = getExoClutterDetectedMoversRVBlob(
            detMap, antPos, boresightVec.flatten(), myRanges, myDopps, antVel, sdr_f[0].fc,
            rp.origin, antAz)

        # Add the detections we got to the dwell track manager
        valid_cpi_time = timeSinceValidCPI if timeSinceValidCPI > 0 else cpi_time
        for t in targetList:
            tracker.add(t, ts[0], boresightVec.flatten(), rp.origin, sdr_f[0].fc, dt=valid_cpi_time, threshold=10)

        cpi_times.append(ts[0])

        tracker.cullTracks(ts[0])
        tracker.fuse()

        """ Parameter Estimation |||||||||||||||||||||||||||||||||||||||||||||||"""
        # First, compute the hAgl (to get closer to the actual hAgl, we should
        #   lookup the DTED value around the center of the swath)
        '''cenSwathPointHat = antPos[:, 0] + boresightVec[:, 0] * midRange
        cenlat, cenlon, cenalt = enu2llh(*cenSwathPointHat, rp.origin)
        hAglHat = antPos[2, 0] + rp.origin[2] - getElevation((cenlat, cenlon))

        detectionCount += len(targetList)'''

        # If wanted, do an animation of the CPI data to see what's going on
        if do_video:
            plotdata.append(db(magData))

        # We need to generate the STANAG info and write it out to the stream. We get
        #   the data from the targets themselves for the target reports, and from
        #   the MoverPositionData object for the dwell
        if do_stanag:
            dwellData = getStanagDwellSegmentData(
                len(targetList), cpi_count, total_cpi_count, antPos[:, 0], antVel[:, 0], rp.att(ts[0]), rp.origin,
                boresightVec[:, 0], midRange, rp.az_half_bw, scan_rate)
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
                        and myRanges[-1] > truthR[i] > myRanges[0]):
                    truthInBeam = True
            if truthInBeam:
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
                if len(targetList) > 0:
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

                    if not closeTargets:
                        continue

                    # Now, let's compute the euclidian distance. If any of them are
                    #   close, we will take the one with the largest magnitude
                    minEuclideanDist = 0
                    maxMagnitude = 0
                    closestCloseTarget = 0
                    brightestCloseTarget = 0
                    for i in range(len(closeTargets)):
                        if i == 0:
                            minEuclideanDist = closeTargets[i][3]
                            closestCloseTarget = i
                            maxMagnitude = closeTargets[i][4]
                            brightestCloseTarget = i
                            continue
                        if closeTargets[i][3] < minEuclideanDist:
                            minEuclideanDist = closeTargets[i][3]
                            closestCloseTarget = i
                        if closeTargets[i][4] > maxMagnitude:
                            maxMagnitude = closeTargets[i][4]
                            brightestCloseTarget = i
                    if closestCloseTarget != brightestCloseTarget:
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
                          % closestTarget)
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
        if cpi_count >= max_cpi_count:
            tracker.removeSingletons()
            break
del slow_time_gpu
del mfilt_gpu
cupy.cuda.MemoryPool().free_all_blocks()
velStep = myDopps[1] - myDopps[0]
fig = plt.figure()
ax = plt.axes(
    xlim=[0.5 * velStep, (wrapVel * 2 - velStep / 2.0)],
    ylim=[myRanges[0], myRanges[-1]])
imageDat = ax.imshow(
    magData, cmap='jet', origin='lower', interpolation='nearest',
    extent=[0.5 * velStep, (wrapVel * 2 - velStep / 2.0),
            myRanges[0], myRanges[-1]], aspect='auto',
    vmin=plotdata[0].max() - 60, vmax=plotdata[0].max())
targ_rvel_array = [[] for _ in range(cpi_count)]
for tidx, t in enumerate(tracker.tracks):
    targ_log = t.state_log[:, :8]
    for cidx, cpi in enumerate(cpi_times):
        if cpi in tracker.update_times[tidx]:
            nt = [idx for idx in range(len(tracker.update_times[tidx])) if cpi == tracker.update_times[tidx][idx]][0]
            targ_rvel_array[cidx].append([targ_log[nt, 0], targ_log[nt, 1]])
for tidx, t in enumerate(tracker.deadtracks):
    targ_log = t.state_log[:, :8]
    for cidx, cpi in enumerate(cpi_times):
        if cpi in tracker.dead_updates[tidx]:
            nt = [idx for idx in range(len(tracker.dead_updates[tidx])) if cpi == tracker.dead_updates[tidx][idx]][0]
            targ_rvel_array[cidx].append([targ_log[nt, 0], targ_log[nt, 1]])
for n in range(cpi_count):
    if len(targ_rvel_array[n]) != 0:
        targ_rvel_array[n] = np.array(targ_rvel_array[n])
    else:
        targ_rvel_array[n] = np.array([[0, 0.]])
trackplot, = ax.plot(targ_rvel_array[0][:, 0], targ_rvel_array[0][:, 1], 'ro', markersize=15, fillstyle='none')
ax.set_xlabel('Radial Velocity (m/s)')
ax.set_ylabel('Range (m)')
ax.set_title('Nuts')
# plot the clutter boundary lines
lowerVThreshDat = ax.axvline(x=1, color='red')
upperVThreshDat = ax.axvline(x=5, color='red')
truthPlotDat, = ax.plot(0, myRanges[0], 'ro', markersize=15, fillstyle='none')


def init():
    # Set the image data with random noise
    imageDat.set_array(np.random.random((nsam, cpi_len)))
    trackplot.set_xdata(targ_rvel_array[0][:, 0])
    trackplot.set_ydata(targ_rvel_array[0][:, 1])
    lowerVThreshDat.set_xdata(20)
    upperVThreshDat.set_xdata(60)
    truthPlotDat.set_xdata(0)
    truthPlotDat.set_ydata(myRanges[0])

    return imageDat, lowerVThreshDat, upperVThreshDat, truthPlotDat


def animate(i):
    # Set the plot data
    imageDat.set_array(plotdata[i])
    trackplot.set_xdata(targ_rvel_array[i][:, 0])
    trackplot.set_ydata(targ_rvel_array[i][:, 1])
    lowerVThreshDat.set_xdata(threshVel + velStep)
    upperVThreshDat.set_xdata(wrapVel * 2 - threshVel + velStep)
    ax.set_title('CPINum: %d' % i)
    return imageDat, lowerVThreshDat, upperVThreshDat, truthPlotDat


""" Run the animation """
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(plotdata) - 1,
    interval=cpi_time / 1e-3, blit=True)

# Save the animation as mp4 video file
anim.save('./Test_TruthRDVideo.mp4')

tr1 = tracker.tracks[0].state_log[:, :14]
mx, my = np.meshgrid(np.linspace(-100, 500, 30), np.linspace(1300, 2800, 30))
mlat, mlon, malt = enu2llh(mx.ravel(), my.ravel(), np.zeros((mx.shape[0] * mx.shape[1],)), rp.origin)
malt = getElevationMap(mlat, mlon)
me, mn, mu = llh2enu(mlat, mlon, malt, rp.origin)
fig = px.scatter_3d(x=tr1[:, 11], y=tr1[:, 12], z=tr1[:, 13])
fig.add_mesh3d(x=me, y=mn, z=mu)
for tr in tracker.tracks:
    tr1 = tr.state_log[:, :14]
    fig.add_scatter3d(x=tr1[:, 11], y=tr1[:, 12], z=tr1[:, 13])
fig.show()
