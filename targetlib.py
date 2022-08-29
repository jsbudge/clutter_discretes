import numpy as np
from dataclasses import dataclass
from simulation_functions import getElevation, enu2llh
from scipy.spatial.distance import pdist
from itertools import combinations


@dataclass
class Target:
    dopp_idx: float
    range_idx: float
    ant_az: float
    det_sz: float = 1.
    n_pts: int = 1
    e: float = 0
    n: float = 0
    u: float = 0
    ve: float = 0
    vn: float = 0
    vu: float = 0

    def calcENU(self, platform, boresight, ranges, origin):
        pt = calcGroundENU(self.range_idx, platform, boresight, ranges, origin)

        # Set the point in our object
        self.e = pt[0]
        self.n = pt[1]
        self.u = pt[2]
        self.loc = pt

    def calcVel(self, boresight, dopp, dopp_fft_len):
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
        radVelVal = -self.dopp_idx * dopp * 2 / dopp_fft_len
        self.ve = sin(eff_az) * radVelVal
        self.vn = cos(eff_az) * radVelVal
        self.vu = 0

    def calc(self, platform, boresight, ranges, origin, dopp, dopp_fft_len):
        self.calcENU(platform, boresight, ranges, origin)
        self.calcVel(boresight, dopp, dopp_fft_len)

    def accept(self, rng_idx, dopp_idx, rng_err, dopp_err):
        Vi = np.linalg.pinv(np.array([[rng_err ** 2, 0],
                                      [0, dopp_err ** 2]]))
        mu = np.array([self.range_idx, self.dopp_idx])
        x = np.array([rng_idx, dopp_idx])
        dist = np.sqrt((x - mu).dot(Vi).dot(x - mu))
        if dist < 5:
            self.range_idx = self.range_idx + (rng_idx - self.range_idx) / (self.n_pts + 1)
            self.dopp_idx = self.dopp_idx + (dopp_idx - self.dopp_idx) / (self.n_pts + 1)
            self.n_pts += 1
            return True
        else:
            return False

    def merge(self, ot):
        self.dopp_idx = (self.dopp_idx + ot.dopp_idx) / 2
        self.range_idx = (self.range_idx + ot.range_idx) / 2
        self.ant_az = (self.ant_az + ot.ant_az) / 2
        self.det_sz = max(self.det_sz, ot.det_sz)
        self.n_pts = max(self.n_pts, ot.n_pts)
        self.e = (self.e + ot.e) / 2
        self.n = (self.n + ot.n) / 2
        self.u = (self.u + ot.u) / 2
        self.ve = (self.ve + ot.ve) / 2
        self.vn = (self.vn + ot.vn) / 2
        self.vu = (self.vu + ot.vu) / 2

    def move(self, ts):
        self.e += self.ve * ts
        self.n += self.vn * ts
        self.u += self.vu * ts


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


class TrackManager(object):
    _tracks = None

    def __init__(self):
        self._tracks = []
        self._errs = np.array([10, 10, 10, 5, 5, 5.])

    def add(self, t):
        self._tracks.append(t)

    def propogate(self, ts):
        # For each of the tracks, update the position based on radial velocity
        for tr in self._tracks:
            tr.move(ts)

    def fuse(self, errs=None, threshold=10):
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

    def split(self):
        pass

    def getTrackPosVel(self):
        return np.array([[t.e, t.n, t.u, t.ve, t.vn, t.vu] for t in self._tracks])

    @property
    def tracks(self):
        return self._tracks
