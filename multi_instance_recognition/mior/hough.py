"""
mior.hough
==========
Stage 3: parameter-space clustering by a 4D Generalised Hough Transform,
implemented from scratch (no sklearn, no cv2 clustering).

WHY A HOUGH TRANSFORM AND NOT PLAIN RANSAC
------------------------------------------
RANSAC over the whole match set needs the inlier fraction to be reasonably
high.  Here it is catastrophically low.  Suppose the ratio test leaves 900
matches, of which 60 belong to instance A, 25 to B, 40 to C and 15 to D, and
760 are background clutter.  The inlier fraction for instance D is 15/900 =
1.7%.  The probability that a random 3-point sample is all-D is 0.017^3 = 5e-6,
so RANSAC would need on the order of a million iterations to see D once.

The Hough transform sidesteps this completely because SIFT gives us a
*similarity pose from a single match*.  A keypoint match is not just a point
pair - it is (location, scale, orientation) on both sides, i.e. four numbers of
pose.  So every individual match votes for a complete hypothesis, and we only
need 3 votes to agree, not 3 correct samples to be drawn at random.

THE VOTE
--------
For a match between template keypoint (x_t, y_t, sigma_t, theta_t) and scene
keypoint (x_s, y_s, sigma_s, theta_s):

    scale       s      = sigma_s / sigma_t
    rotation    dtheta = (theta_s - theta_t)  mod 360

    Let c_t = (W/2, H/2) be the template centre and v = c_t - (x_t, y_t) the
    vector from this keypoint to the centre, expressed in template pixels.
    Under a similarity of scale s and rotation dtheta that vector maps to
    s * R(dtheta) * v, so the predicted object centre in the scene is

        c_s = (x_s, y_s) + s * R(dtheta) * v

    with  R(a) = [[cos a, -sin a], [sin a, cos a]].

Each match therefore contributes one point (c_x, c_y, log2 s, dtheta) in a 4D
pose space.  Correct matches from one instance all predict nearly the same
centre, scale and rotation, so they pile into one bin; clutter matches predict
essentially random poses and spread out thinly.

BINNING
-------
Following Lowe (2004) section 7.1:

    orientation : 30 degree bins            (12 bins, wrapped)
    scale       : factor-of-2 bins, i.e. bins of width 1 in log2 s
    location    : bins of 0.25 x (projected template dimension)

The location bin width has to grow with the hypothesised scale - a 0.42-scale
instance is physically smaller, so its centre must be localised more tightly
than a 1.0-scale one.  We make the width a function of the *scale bin*, using
the geometric-mean scale of that bin, so the grid is still well defined:

    L_k = 0.25 * max(W, H) * 2^(k + 0.5)      for scale bin k

BROADENED (SOFT) VOTING
-----------------------
Hard binning suffers from boundary effects: a cluster sitting on a bin edge
gets split in half and may fall below the vote threshold.  Lowe's remedy, which
we implement, is to have each match vote for the 2 nearest bins in *each* of
the 4 dimensions - 2^4 = 16 bins per match.  This guarantees that a coherent
cluster is fully contained in at least one bin, at the cost of a factor-16
inflation of the table, which is irrelevant at this problem size because the
table is a sparse dictionary rather than a dense 4D array.

A dense 4D accumulator would be about
    (1320/L) x (960/L) x 7 x 12
cells and mostly empty; a hash map keyed by the integer 4-tuple costs memory
proportional to the number of votes instead.
"""

from collections import defaultdict

import numpy as np

# Bin geometry
THETA_BIN_DEG = 30.0      # 12 orientation bins around the circle
LOG2_SCALE_BIN = 1.0      # factor-of-2 scale bins
LOC_BIN_FRACTION = 0.25   # location bin = 0.25 x projected template dimension


def pose_votes(scene_feats, tmpl_feats, matches, tmpl_wh):
    """Turn every surviving match into one 4D pose hypothesis.

    Returns
    -------
    cx, cy   : (M,) predicted object-centre coordinates in the scene
    log2s    : (M,) log2 of the predicted scale
    dtheta   : (M,) predicted rotation in degrees, in [0, 360)
    """
    W, H = tmpl_wh
    ct = np.array([W / 2.0, H / 2.0])

    t_xy = tmpl_feats.xy[matches.tmpl_idx]
    t_sig = tmpl_feats.sigma[matches.tmpl_idx]
    t_th = tmpl_feats.theta[matches.tmpl_idx]

    s_xy = scene_feats.xy[matches.scene_idx]
    s_sig = scene_feats.sigma[matches.scene_idx]
    s_th = scene_feats.theta[matches.scene_idx]

    s = s_sig / np.maximum(t_sig, 1e-9)
    dtheta = np.mod(s_th - t_th, 360.0)

    v = ct[None, :] - t_xy                      # template kp -> template centre
    a = np.deg2rad(dtheta)
    ca, sa = np.cos(a), np.sin(a)

    # s * R(dtheta) * v
    rx = s * (ca * v[:, 0] - sa * v[:, 1])
    ry = s * (sa * v[:, 0] + ca * v[:, 1])

    cx = s_xy[:, 0] + rx
    cy = s_xy[:, 1] + ry
    return cx, cy, np.log2(np.maximum(s, 1e-9)), dtheta


def _neighbour_pair(value, width):
    """Return the two bin indices and how the vote splits between them.

    For a continuous coordinate `value` and bin `width`, the vote goes to the
    bin it falls in and to the nearer neighbouring bin.  This is the 1-D
    building block of Lowe's broadened voting.
    """
    u = value / width
    k = int(np.floor(u))
    frac = u - k                       # position inside the bin, in [0, 1)
    k2 = k + 1 if frac >= 0.5 else k - 1
    return k, k2


def build_hough_table(cx, cy, log2s, dtheta, tmpl_wh,
                      scale_range=(-2.5, 2.0), scene_wh=None, margin=0.6):
    """Accumulate broadened votes into a sparse 4D table.

    Returns
    -------
    table : dict  (k_scale, k_theta, ix, iy) -> list of match indices
    """
    W, H = tmpl_wh
    maxdim = max(W, H)
    n_theta = int(round(360.0 / THETA_BIN_DEG))
    table = defaultdict(list)

    lo, hi = scale_range
    for i in range(len(cx)):
        if not (lo <= log2s[i] <= hi):
            continue                                    # implausible scale
        if scene_wh is not None:
            sw, sh = scene_wh
            # A predicted centre far outside the image cannot be a real object.
            if not (-margin * sw <= cx[i] <= (1 + margin) * sw and
                    -margin * sh <= cy[i] <= (1 + margin) * sh):
                continue

        ks1, ks2 = _neighbour_pair(log2s[i], LOG2_SCALE_BIN)
        kt1, kt2 = _neighbour_pair(dtheta[i], THETA_BIN_DEG)

        for ks in (ks1, ks2):
            if not (lo - 1 <= ks <= hi + 1):
                continue
            # Location bin width follows the scale bin (see module docstring).
            L = LOC_BIN_FRACTION * maxdim * (2.0 ** (ks + 0.5))
            L = max(L, 8.0)                              # never finer than 8 px
            ix1, ix2 = _neighbour_pair(cx[i], L)
            iy1, iy2 = _neighbour_pair(cy[i], L)
            for kt in (kt1, kt2):
                ktw = kt % n_theta                       # orientation wraps
                for ix in (ix1, ix2):
                    for iy in (iy1, iy2):
                        table[(ks, ktw, ix, iy)].append(i)

    # A match can land twice in the same bin only through duplicated
    # neighbour indices (frac exactly 0.5 etc.); de-duplicate defensively.
    return {k: np.unique(np.asarray(v, np.int64)) for k, v in table.items()}


def ranked_bins(table, min_votes=3):
    """Bins with at least `min_votes` matches, most-voted first.

    The assignment's threshold of >= 3 votes is the natural one: 3 point
    correspondences are the minimum needed to determine a 2D affine transform,
    so a bin with fewer votes cannot produce a solvable system anyway.
    """
    items = [(k, v) for k, v in table.items() if len(v) >= min_votes]
    items.sort(key=lambda kv: -len(kv[1]))
    return items


def vote_histogram_2d(cx, cy, scene_wh, bin_px=24):
    """Marginalise the 4D vote space onto the (x, y) plane, for visualisation.

    This is only used to draw the Hough-space figure in the report; the
    pipeline itself never builds a dense array.
    """
    sw, sh = scene_wh
    nx, ny = int(np.ceil(sw / bin_px)), int(np.ceil(sh / bin_px))
    acc = np.zeros((ny, nx), np.float32)
    ix = np.floor(cx / bin_px).astype(int)
    iy = np.floor(cy / bin_px).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(acc, (iy[ok], ix[ok]), 1.0)
    return acc
