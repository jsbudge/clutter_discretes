import numpy as np
from numpy import *
from dataclasses import dataclass
from simulation_functions import getElevation, enu2llh
from scipy.spatial.distance import pdist
from itertools import combinations
from kalman import UKF


# Constants
c0 = 299792458.0
kb = 1.3806503e-23
T0 = 290.0


@dataclass
class Target(object):
    rad_vel: float
    range: float
    ant_pos: list
    ant_vel: list
    az: float
    el: float
    rng_err: float = 5.
    dopp_err: float = 2.

    def accept(self, di, ri):
        Vi = np.linalg.pinv(np.array([[self.rng_err ** 2, 0],
                                      [0, self.dopp_err ** 2]]))
        mu = np.array([self.range, self.rad_vel])
        x = np.array([ri, di])
        dist = np.sqrt((x - mu).dot(Vi).dot(x - mu))
        if dist < 5:
            self.rad_vel = (self.rad_vel + di) / 2
            self.range = (self.range + ri) / 2
            return True
        return False

    @property
    def meas(self):
        return np.array([self.rad_vel, self.range, *self.ant_pos, *self.ant_vel,
                         self.az, self.el, self.range])


class Track:

    def __init__(self, fd_o, r_o, ant_pos, ant_vel, boresight, origin, fc, init_time, dt=1.0, Q=None, R=None):
        targ_pos = calcGroundENU(r_o, ant_pos, boresight, origin)
        # targ_vel = calcDoppVel(boresight, fd_o, fc, ant_vel)
        init_state = np.array([fd_o, r_o, *ant_pos, *ant_vel, 0, 0, 0, *targ_pos,
                               np.arctan2(boresight[0], boresight[1]), 0, np.arcsin(boresight[2]), 0, r_o])
        self.origin = origin
        self._kf = UKF(init_state, self.process, self.measure, init_time, dt=dt, adaptive_constant=None, Q=Q, R=R)

    def update(self, z, ts):
        self._kf.update(z, ts)

    def propogate(self, ts=None):
        self._kf.predict(ts)

    def merge(self, ot):
        pass

    @property
    def state(self):
        return self._kf.x

    @property
    def meas(self):
        return np.array([*self._kf.x[:8], self._kf.x[14], self._kf.x[16], self._kf.x[1]])

    @property
    def state_log(self):
        return np.array(self._kf.x_log)

    def process(self, x, dt=1.0):
        new_state = np.zeros((len(x),))
        new_plane_pos = x[2:5] + x[5:8] * dt
        rng = x[1] + x[0] * dt
        new_az = x[14] + x[15] * dt
        new_el = x[16] + x[17] * dt
        # Inertial grazing angle
        boresight = np.array([np.cos(new_el) * np.sin(new_az),
                  np.cos(new_el) * np.cos(new_az),
                  np.sin(new_el)])
        # Radial velocity
        new_state[0] = x[0]
        # Range to target
        new_state[1] = rng
        # Plane pos
        new_state[2:5] = new_plane_pos
        # Plane vel
        new_state[5:8] = x[5:8] + x[8:11] * dt
        # Plane accel
        new_state[8:11] = x[8:11]
        # Target pos
        new_state[11:14] = boresight * rng + new_plane_pos
        # Inertial Azimuth
        new_state[14] = new_az
        # Inertial Azimuth Velocity
        new_state[15] = x[15]
        # Intertial Grazing Angle
        new_state[16] = new_el
        # Inertial grazing angle velocity
        new_state[17] = x[17]
        # Error state for range
        new_state[18] = np.linalg.norm(new_plane_pos - new_state[11:14])
        return new_state

    def measure(self, state, dt=1.0):
        new_meas = np.zeros((11,))
        # Radial velocity
        new_meas[0] = state[0]
        # Range to target
        new_meas[1] = state[1]
        # Plane pos
        new_meas[2:5] = state[2:5]
        # Plane vel
        new_meas[5:8] = state[5:8]
        # Inertial Azimuth
        new_meas[8] = state[14]
        # Inertial grazing
        new_meas[9] = state[16]
        # Error between state range and actual range
        new_meas[10] = state[1]
        return new_meas


class TrackManager(object):
    _tracks = None

    def __init__(self, deadtrack_time=5, Q=None, R=None):
        self._tracks = []
        self._deadtracks = []
        self._dt = deadtrack_time
        self.update_times = []
        self.dead_updates = []
        if R is not None:
            self._errs = R
        else:
            self._errs = np.diag(np.array([2.5, 10, 10, 10, 10, 10., 10, 10, 10])**2)
        self._R = R
        self._Q = Q

    def add(self, t, current_time, boresight, origin, fc, dt=1.0, threshold=10):
        if len(self._tracks) > 1:
            # Calculate Mahal distance between t and the tracks
            Vi = np.linalg.pinv(self._errs)
            pv = self.getTrackPosVel()
            mu = t.meas
            dists = np.array([np.sqrt((pv[n, :] - mu).dot(Vi.dot(pv[n, :] - mu)))
                              for n in range(len(self._tracks))])
            if np.any(dists < threshold):
                trs = np.argsort(dists)
                tr = 0
                while current_time in self._tracks[trs[tr]]._kf.update_time:
                    tr += 1
                    if dists[trs[tr]] > threshold:
                        nt = Track(t.rad_vel, t.range, t.ant_pos, t.ant_vel, boresight, origin, fc, current_time,
                                   dt=dt, Q=self._Q, R=self._R)
                        self._tracks.append(nt)
                        self.update_times.append([current_time])
                        return
                self._tracks[trs[tr]].update(mu, current_time)
                if current_time not in self.update_times[trs[tr]]:
                    self.update_times[trs[tr]].append(current_time)
            else:
                nt = Track(t.rad_vel, t.range, t.ant_pos, t.ant_vel, boresight, origin, fc, current_time,
                           dt=dt, Q=self._Q, R=self._R)
                self._tracks.append(nt)
                self.update_times.append([current_time])
        else:
            nt = Track(t.rad_vel, t.range, t.ant_pos, t.ant_vel, boresight, origin, fc, current_time,
                       dt=dt, Q=self._Q, R=self._R)
            self._tracks.append(nt)
            self.update_times.append([current_time])

    def propogate(self, ts):
        # For each of the tracks, update the position based on radial velocity
        for tr in self._tracks:
            tr.propogate(ts)

    def fuse(self, threshold=10):
        Vi = np.linalg.pinv(self._errs)
        dists = pdist(self.getTrackPosVel(), metric='mahalanobis', VI=Vi)
        fuzors = dists < threshold
        if np.any(fuzors):
            nuke = np.zeros((len(self._tracks,))).astype(bool)
            idxes = np.array(list(combinations(np.arange(len(self._tracks)), 2)))[dists < threshold, :]
            uniques = np.unique(idxes)
            # Iterate until all the idxes are taken care of
            while len(uniques) > 0:
                val = uniques[0]
                mergers = idxes[np.logical_or(idxes[:, 0] == val, idxes[:, 1] == val), :]
                for n in range(1, mergers.shape[0]):
                    deletor = mergers[n, 0] if mergers[n, 0] != val else mergers[n, 1]
                    self._tracks[val].merge(self._tracks[deletor])
                    nuke[deletor] = True
                    idxes = idxes[np.logical_not(np.logical_or(idxes[:, 0] == deletor, idxes[:, 1] == deletor)), :]
                idxes = idxes[np.logical_not(np.logical_or(idxes[:, 0] == val, idxes[:, 1] == val)), :]
                uniques = np.unique(idxes)
            # Remove all the merged tracks in reverse order so they don't mess each other up
            for idx in sorted(np.arange(len(self._tracks))[nuke], reverse=True):
                del self._tracks[idx]
                del self.update_times[idx]

    def cullTracks(self, ts):
        nuke = [idx for idx in range(len(self._tracks)) if self.update_times[idx][-1] < ts - self._dt]
        for idx in sorted(nuke, reverse=True):
            # If it's not truly a track, just delete it
            if len(self.update_times[idx]) < 3:
                del self._tracks[idx]
                del self.update_times[idx]
            else:
                self._deadtracks.append(self._tracks.pop(idx))
                self.dead_updates.append(self.update_times.pop(idx))

    def removeSingletons(self):
        nuke = [idx for idx in range(len(self._tracks)) if len(self.update_times[idx]) < 3]
        for idx in sorted(nuke, reverse=True):
            del self._tracks[idx]
            del self.update_times[idx]

    def getTrackPosVel(self):
        return np.array([t.meas for t in self._tracks])

    @property
    def tracks(self):
        return self._tracks

    @property
    def deadtracks(self):
        return self._deadtracks


def calcGroundENU(rng, platform, boresight, origin):
    az_inertial = np.arctan2(boresight[0], boresight[1])
    el_inertial = -np.arcsin(boresight[2])
    guess = platform + boresight * rng
    lat, lon, alt = enu2llh(*guess, origin)
    surf_height = getElevation((lat, lon))
    delta_height = surf_height - (origin[2] + alt)
    its = 0
    while abs(delta_height) > 1. and its < 15:
        h_agl = platform[2] + origin[2] - surf_height
        el_inertial = np.arcsin(h_agl / rng)
        Rvec = np.array([np.cos(el_inertial) * np.sin(az_inertial),
                         np.cos(el_inertial) * np.cos(az_inertial),
                         -np.sin(el_inertial)])
        guess = platform + Rvec * rng
        lat, lon, alt = enu2llh(*guess, origin)
        surf_height = getElevation((lat, lon))

        # check the error in the elevation compared to what was calculated
        delta_height = surf_height - alt
        its += 1

    # Calc out point with corrected grazing angle
    Rvec = np.array([np.cos(el_inertial) * np.sin(az_inertial),
                     np.cos(el_inertial) * np.cos(az_inertial),
                     -np.sin(el_inertial)])
    return platform + Rvec * rng


def calcDoppVel(boresight, rad_vel, fc, platform_vel):
    # Very first thing, let's resolve wrapping issues if this target
    # has been flagged as being wrapped. It is really kind of sixes to
    # know which way we need to unwrap without further information or
    # more of a data history on the target. So, we could either choose
    # up always, or we could try unwrapping toward the side with the
    # greater mass. I really don't know if that will buy us anything or
    # be of much benefit though. Or whether it will be worth the cost
    # of computation. Maybe I'll do that later, I'm not sure buys me
    # much of anything.
    eff_az = np.arctan2(boresight[0], boresight[1])
    radVelVal = rad_vel * np.sqrt(boresight[0]**2 + boresight[1]**2)
    vr = np.array([np.sin(eff_az) * radVelVal, np.cos(eff_az) * radVelVal, 0])
    vr[2] = 0
    return vr


def computeGrazingAngle(effAzIR, grazeIR, antPos, theRange, origin):
    # initialize the pointing vector to first range bin
    Rvec = array([cos(grazeIR) * sin(effAzIR),
                  cos(grazeIR) * cos(effAzIR),
                  -sin(grazeIR)])

    groundPoint = antPos + Rvec * theRange
    nlat, nlon, alt = enu2llh(*groundPoint, origin)
    # look up the height of the surface below the aircraft
    surfaceHeight = getElevation((nlat, nlon), False)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - alt

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while abs(elevDiff) > heightDiffThresh and numIterations < iterationThresh:
        hAgl = antPos[2] + origin[2] - surfaceHeight
        newGrazeR = arcsin(hAgl / theRange)
        if isnan(newGrazeR) or isinf(newGrazeR):
            print('NaN or inf found.')
        Rvec = array([cos(newGrazeR) * sin(effAzIR),
                      cos(newGrazeR) * cos(effAzIR),
                      -sin(newGrazeR)])
        groundPoint = antPos + Rvec * theRange
        nlat, nlon, alt = enu2llh(*groundPoint, origin)
        surfaceHeight = getElevation((nlat, nlon), False)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - alt
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations