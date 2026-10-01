"""
mior.matching
=============
Stage 2: feature correspondence, implemented from scratch.

Banned here: cv2.BFMatcher, cv2.FlannBasedMatcher, sklearn - none are imported
anywhere in this file.  We compute the Euclidean distance matrix ourselves and
apply Lowe's ratio test ourselves.

DIRECTION OF MATCHING  (this is the single most important design decision)
-------------------------------------------------------------------------
The naive thing to do is to match TEMPLATE -> SCENE: for each template
descriptor find its nearest neighbour in the scene.  That fails immediately on
a multi-instance problem.  If the packet appears four times, a given template
feature has four almost equally good neighbours in the scene, so the ratio
d1/d2 is close to 1.0 and Lowe's test *rejects every correct match*.  The test
was designed to reject ambiguity, and genuine repetition looks exactly like
ambiguity from that direction.

So we match SCENE -> TEMPLATE, as the assignment hints.  Each scene keypoint
asks "which single template feature do I look like?".  A scene feature lying on
instance A and a scene feature lying on instance B independently map to the
same template feature, and neither is ambiguous, so both survive.  The ratio
test now does what we actually want: it rejects scene keypoints that come from
background clutter and happen to look vaguely like several template features.

MATHEMATICS
-----------
Euclidean distance is expanded so it can be evaluated as one matrix product:

    ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2

With S (Ns x 128) scene descriptors and T (Nt x 128) template descriptors,

    D2 = ||S||^2 (column) - 2 S T^T + ||T||^2 (row)

is the full Ns x Nt matrix of squared distances, computed in a few BLAS calls.
It is evaluated in row blocks so that memory stays bounded for a scene with
tens of thousands of keypoints.

Lowe's ratio test: let d1 <= d2 be the two smallest distances from a scene
descriptor to the template set.  Keep the match to the nearest only if

    d1 / d2  <  rho          (rho = 0.80 in Lowe 2004)

The justification is that for a correct match d1 comes from the true
corresponding feature while d2 comes from an unrelated one, so d1 << d2; for a
false match both distances are drawn from the same "random descriptor"
distribution and the ratio sits near 1.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    """Scene->template correspondences that survived the ratio test.

    scene_idx / tmpl_idx : (M,) int   indices into the two FeatureSets
    d1, d2               : (M,) float nearest / second-nearest distances
    ratio                : (M,) float d1 / d2
    """
    scene_idx: np.ndarray
    tmpl_idx: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    ratio: np.ndarray

    def __len__(self):
        return len(self.scene_idx)

    def subset(self, mask_or_idx):
        i = np.asarray(mask_or_idx)
        return Matches(self.scene_idx[i], self.tmpl_idx[i],
                       self.d1[i], self.d2[i], self.ratio[i])


def squared_distance_block(S, T):
    """||s - t||^2 for every pair, via the expansion above.

    Returns an (len(S), len(T)) float32 array.  Small negative values caused by
    floating-point cancellation are clamped to zero before the caller takes a
    square root.
    """
    s2 = np.einsum("ij,ij->i", S, S)[:, None]      # ||s||^2 as a column
    t2 = np.einsum("ij,ij->i", T, T)[None, :]      # ||t||^2 as a row
    d2 = s2 - 2.0 * (S @ T.T) + t2
    np.maximum(d2, 0.0, out=d2)
    return d2


def two_nearest(S, T, block=2048):
    """Brute-force 1st and 2nd nearest neighbour of every row of S inside T.

    Implemented by hand rather than with a matcher object.  Row-blocked so the
    peak memory is block x len(T) floats regardless of how many scene keypoints
    the cluttered image produced.

    Returns
    -------
    idx1, d1, d2 : nearest index, nearest distance, second-nearest distance
    """
    n = len(S)
    idx1 = np.empty(n, np.int64)
    d1 = np.empty(n, np.float64)
    d2 = np.empty(n, np.float64)

    for start in range(0, n, block):
        stop = min(start + block, n)
        D2 = squared_distance_block(S[start:stop], T)

        # argpartition gives us the two smallest entries per row in O(Nt)
        # without sorting the whole row.
        part = np.argpartition(D2, 1, axis=1)[:, :2]
        rows = np.arange(stop - start)[:, None]
        two = D2[rows, part]
        order = np.argsort(two, axis=1)
        first = part[rows[:, 0], order[:, 0]]
        second = part[rows[:, 0], order[:, 1]]

        idx1[start:stop] = first
        d1[start:stop] = np.sqrt(D2[np.arange(stop - start), first])
        d2[start:stop] = np.sqrt(D2[np.arange(stop - start), second])

    return idx1, d1, d2


def match_scene_to_template(scene_desc, tmpl_desc, ratio=0.80, block=2048):
    """Full correspondence stage: nearest neighbours + Lowe's ratio test.

    Parameters
    ----------
    ratio : float
        Lowe's rho.  0.80 is the classic value.  Because we match in the
        scene->template direction, repetition of the object does NOT inflate
        the ratio, so this can stay strict even with four instances present.

    Returns
    -------
    Matches (only the surviving correspondences), and the unfiltered
    (idx1, d1, d2) triple so that the report can visualise the "naive" stage.
    """
    idx1, d1, d2 = two_nearest(scene_desc, tmpl_desc, block=block)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d2 > 1e-12, d1 / d2, 1.0)

    keep = np.flatnonzero(r < ratio)
    m = Matches(scene_idx=keep, tmpl_idx=idx1[keep],
                d1=d1[keep], d2=d2[keep], ratio=r[keep])
    return m, (idx1, d1, d2, r)


def all_nearest_matches(idx1, d1, d2, r):
    """Every nearest-neighbour match with NO ratio test - the 'naive' baseline.

    Used only to produce the chaotic comparison figure for the report.
    """
    n = len(idx1)
    return Matches(scene_idx=np.arange(n), tmpl_idx=idx1, d1=d1, d2=d2, ratio=r)
