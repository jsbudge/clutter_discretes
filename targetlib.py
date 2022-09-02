import numpy as np
from dataclasses import dataclass
from simulation_functions import getElevation, enu2llh
from scipy.spatial.distance import pdist
from itertools import combinations
from kalman import UKF


# Constants
c0 = 299792458.0
kb = 1.3806503e-23
T0 = 290.0


class Target:

    def __init__(self, fd_o, r_o, ant_pos, ant_vel):
        init_state = np.array([fd_o, r_o, *ant_pos, *ant_vel])
        self._kf = UKF(init_state, process, measure)
        self.poss_z = np.zeros((8,))
        self.n_z = 0

    def accept(self, fd, rng, ant_pos, ant_vel):
        Vi = np.linalg.pinv(self._kf.Q)
        mu = self._kf.getMeasurement()
        x = np.array([fd, rng, *ant_pos, *ant_vel])
        dist = np.sqrt((x - mu).dot(Vi).dot(x - mu))
        if dist < 5:
            self.poss_z += x
            self.n_z += 1
            return True
        else:
            return False

    def predict(self):
        self._kf.predict()

    def update(self, curr_cpi):
        if self.n_z != 0:
            z = self.poss_z / self.n_z
            self._kf.update(z, curr_cpi)
            self.poss_z = np.zeros((8,))
            self.n_z = 0
            return True
        return False

    def merge(self, to):
        self._kf.x = (self._kf.x + to.x) / 2

    @property
    def x(self):
        return self._kf.x


def process(x, dt=1.0):
    new_state = np.zeros((len(x),))
    new_plane_pos = x[2:5] + x[5:8]
    new_targ_pos = x[8:11] + x[11:14]
    boresight = new_plane_pos - new_targ_pos
    boresight /= np.linalg.norm(boresight)
    # Radial velocity
    new_state[0] = np.linalg.norm(x[5:8] - x[11:14]) * np.sign(np.cross(boresight, x[5:8]))
    # Range to target
    new_state[1] = np.linalg.norm(new_plane_pos - new_targ_pos)
    # Plane pos
    new_state[2:5] = new_plane_pos
    # Plane vel
    new_state[5:8] = x[5:8]
    # Target pos
    new_state[8:11] = new_targ_pos
    # Target vel
    new_state[11:14] = x[11:14]
    return new_state


def measure(state, dt=1.0):
    new_meas = np.zeros((8,))
    # Radial velocity
    new_meas[0] = state[0]
    # Range to target
    new_meas[1] = state[1]
    # Plane pos
    new_meas[2:5] = state[2:5]
    # Plane vel
    new_meas[5:8] = state[5:8]
    return new_meas


def calcGroundENU(rng_idx, platform, boresight, ranges, origin):
    az_inertial = np.arctan2(boresight[0], boresight[1])
    el_inertial = -np.arcsin(boresight[2])
    rng = np.interp(rng_idx, np.arange(len(ranges)), ranges)
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


def calcDoppVel(boresight, dopp, dopp_idx, fc, platform_vel):
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
    radVelVal = np.interp(dopp_idx, np.arange(len(dopp)), dopp) * np.sqrt(boresight[0]**2 + boresight[1]**2)
    vr = np.array([np.sin(eff_az) * radVelVal, np.cos(eff_az) * radVelVal, 0])
    vr[2] = 0
    return vr


class TrackManager(object):
    _tracks = None

    def __init__(self, deadtrack_time=5):
        self._tracks = []
        self._deadtracks = []
        self._dt = deadtrack_time
        self.update_times = []
        self.dead_updates = []
        self._errs = np.array([5, 10, 2, 2, 2, 2, 2, 2])

    def add(self, t, current_time, threshold=10):
        if len(self._tracks) > 1:
            # Calculate Mahal distance between t and the tracks
            Vi = np.linalg.pinv(np.diag(self._errs))
            pv = self.getTrackPosVel()
            mu = np.array([t.x[8:14]])
            dists = np.array([np.sqrt((pv[n, :] - mu).dot(Vi.dot(pv[n, :] - mu))) for n in range(len(self._tracks))])
            if np.any(dists < threshold):
                tr = np.where(dists == dists.min())[0][0]
                self._tracks[tr].merge(t)
                if current_time not in self.update_times[tr]:
                    self.update_times[tr].append(current_time)
            else:
                self._tracks.append(t)
                self.update_times.append([current_time])
        else:
            self._tracks.append(t)
            self.update_times.append([current_time])

    def propogate(self, ts):
        # For each of the tracks, update the position based on radial velocity
        for tr in self._tracks:
            tr.move(ts)

    def fuse(self, threshold=10):
        Vi = np.linalg.pinv(np.diag(self._errs))
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

    def update(self, ts):
        for t in self._tracks:
            t.update(ts)
        nuke = [idx for idx in range(len(self._tracks)) if self.update_times[idx][-1] < ts - self._dt]
        for idx in sorted(nuke, reverse=True):
            self._deadtracks.append(self._tracks.pop(idx))
            self.dead_updates.append(self.update_times.pop(idx))

    def predict_all(self):
        for t in self._tracks:
            t.predict()

    def getTrackPosVel(self):
        return np.array([t.x[8:14] for t in self._tracks])

    @property
    def tracks(self):
        return self._tracks

    @property
    def deadtracks(self):
        return self._deadtracks
