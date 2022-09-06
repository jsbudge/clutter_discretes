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
                         self.az, self.el])


class Track:

    def __init__(self, fd_o, r_o, ant_pos, ant_vel, boresight, origin, fc, Q=None, R=None):
        targ_pos = calcGroundENU(r_o, ant_pos, boresight, origin)
        targ_vel = calcDoppVel(boresight, fd_o, fc, ant_vel)
        init_state = np.array([fd_o, r_o, *ant_pos, *ant_vel, 0, 0, 0, *targ_pos, *targ_vel,
                               np.arctan2(boresight[0], boresight[1]), 0, np.arcsin(boresight[2]), 0])
        self._kf = UKF(init_state, process, measure, Q=Q, R=R)

    def accept(self, target):
        Vi = np.linalg.pinv(self._kf.R)
        mu = target.meas
        x = np.array([*self._kf.x[:8], self._kf.x[17], self._kf.x[19]])
        dist = np.sqrt((x - mu).dot(Vi).dot(x - mu))
        if dist < 5:
            self._kf.update(target.meas, 1)
            return True
        else:
            return False

    def update(self, z, ts):
        self._kf.update(z, ts)

    def move(self, ts=0):
        self._kf.predict()

    def merge(self, ot):
        pass

    @property
    def state(self):
        return self._kf.x

    @property
    def meas(self):
        return np.array([*self._kf.x[:8], self._kf.x[17], self._kf.x[19]])

    @property
    def state_log(self):
        return np.array(self._kf.x_log)


def process(x, dt=1.0):
    new_state = np.zeros((len(x),))
    new_plane_pos = x[2:5] + x[5:8] * dt
    new_targ_pos = x[8:11] + x[11:14] * dt
    boresight = np.array([np.cos(x[19]) * np.sin(x[17]),
                  np.cos(x[19]) * np.cos(x[17]),
                  -np.sin(x[19])])
    # Radial velocity
    new_state[0] = np.linalg.norm(x[5:8] - x[11:14]) * np.sign(np.cross(boresight, x[5:8]))[2]
    # Range to target
    new_state[1] = np.linalg.norm(new_plane_pos - new_targ_pos)
    # Plane pos
    new_state[2:5] = new_plane_pos
    # Plane vel
    new_state[5:8] = x[5:8] + dt * x[8:11]
    # Plane accel
    new_state[8:11] = x[8:11]
    # Target pos
    new_state[11:14] = new_targ_pos
    # Target vel
    new_state[14:17] = x[14:17]
    # Inertial Azimuth
    new_state[17] = x[17] + dt * x[18]
    # Inertial Azimuth Velocity
    new_state[18] = new_state[18]
    # Inertial Elevation
    new_state[19] = x[19] + dt * x[20]
    # Inertial Elevation Velocity
    new_state[20] = x[20]
    return new_state


def measure(state, dt=1.0):
    new_meas = np.zeros((10,))
    # Radial velocity
    new_meas[0] = state[0]
    # Range to target
    new_meas[1] = state[1]
    # Plane pos
    new_meas[2:5] = state[2:5]
    # Plane vel
    new_meas[5:8] = state[5:8]
    # Inertial Azimuth
    new_meas[8] = state[17]
    # Inertial Elevation
    new_meas[9] = state[19]
    return new_meas


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
            self._errs = np.diag(np.array([2.5, 10, 10, 10, 10, 10., 10, 10, 10, 10])**2)
        self._R = R
        self._Q = Q

    def add(self, t, current_time, boresight, origin, fc, threshold=10):
        if len(self._tracks) > 1:
            # Calculate Mahal distance between t and the tracks
            Vi = np.linalg.pinv(self._errs)
            pv = self.getTrackPosVel()
            mu = t.meas
            dists = np.array([np.sqrt((pv[n, :] - mu).dot(Vi.dot(pv[n, :] - mu)))
                              for n in range(len(self._tracks)) if current_time not in self.update_times[n]])
            if np.any(dists < threshold):
                tr = np.where(dists == dists.min())[0][0]
                self._tracks[tr].update(mu, current_time)
                if current_time not in self.update_times[tr]:
                    self.update_times[tr].append(current_time)
            else:
                nt = Track(t.rad_vel, t.range, t.ant_pos, t.ant_vel, boresight, origin, fc, Q=self._Q, R=self._R)
                self._tracks.append(nt)
                self.update_times.append([current_time])
        else:
            nt = Track(t.rad_vel, t.range, t.ant_pos, t.ant_vel, boresight, origin, fc, Q=self._Q, R=self._R)
            self._tracks.append(nt)
            self.update_times.append([current_time])

    def propogate(self, ts):
        # For each of the tracks, update the position based on radial velocity
        for tr in self._tracks:
            tr.move(ts)

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
            self._deadtracks.append(self._tracks.pop(idx))
            self.dead_updates.append(self.update_times.pop(idx))
            # del self._tracks[idx]
            # del self.update_times[idx]

    def getTrackPosVel(self):
        return np.array([t.meas for t in self._tracks])

    @property
    def tracks(self):
        return self._tracks

    @property
    def deadtracks(self):
        return self._deadtracks
