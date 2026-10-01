"""
mior.geometry
=============
Stage 4a: the 2D affine transform, solved from scratch.

Banned here: cv2.estimateAffine2D, cv2.findHomography, cv2.getAffineTransform.
None are imported.  We build the linear system Ax = b ourselves and solve the
normal equations with NumPy's general-purpose linear algebra (explicitly
allowed as a "matrix operation").

THE MODEL
---------
A 2D affine transform maps a template point p = (x, y) to a scene point
p' = (x', y') as

    [x']   [a  b][x]   [tx]
    [y'] = [c  d][y] + [ty]

Six unknowns.  Each point correspondence gives two scalar equations, so three
non-collinear correspondences determine the transform exactly and more than
three over-determine it.

Written as a linear system in the unknown vector u = [a, b, tx, c, d, ty]^T,
each correspondence contributes two rows:

    [x  y  1  0  0  0] u = x'
    [0  0  0  x  y  1] u = y'

Stacking n correspondences gives A (2n x 6) and b (2n).  The least-squares
solution minimising ||Au - b||^2 satisfies the normal equations

    (A^T A) u = A^T b

which is a 6x6 symmetric system.  We solve it directly, falling back to a
pseudo-inverse when A^T A is singular (which happens exactly when the
correspondences are degenerate).

WHY AFFINE AND NOT A FULL HOMOGRAPHY
------------------------------------
Six parameters need three points; a homography needs four and is much more
willing to bend into a nonsense shape when driven by a handful of noisy,
clutter-contaminated matches.  For a roughly planar object viewed at a modest
tilt the affine model is an excellent approximation, and its rigidity is a
feature: it is far harder for outliers to produce a plausible-looking affine
transform than a plausible-looking homography.

DEGENERACY
----------
An affine fit from three points is degenerate when those points are collinear -
the 6x6 matrix loses rank and the solution shears the template into a sliver.
Two guards are used:

  1. Collinearity / triangle-area test on the sample, normalised by the size of
     the sample so it is scale independent.
  2. Sanity of the linear part M = [[a, b], [c, d]] once solved:
       * det(M) > 0            (no mirror image - a real packet cannot flip)
       * sqrt(det M) in a plausible scale range
       * the two singular values of M are of similar magnitude, i.e. the
         anisotropy sigma_max / sigma_min is bounded.  This is the shear guard:
         a sheared, sliver-like box has a huge singular-value ratio.
"""

import numpy as np


def solve_affine_lsq(src, dst):
    """Least-squares affine fit mapping `src` (N,2) onto `dst` (N,2).

    Returns
    -------
    (2, 3) array [[a, b, tx], [c, d, ty]] or None if the system is unsolvable.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 3:
        return None

    # Build A (2n x 6) and b (2n) exactly as written in the docstring.
    A = np.zeros((2 * n, 6))
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = src[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 3] = src[:, 0]
    A[1::2, 4] = src[:, 1]
    A[1::2, 5] = 1.0

    b = np.empty(2 * n)
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]

    # Normal equations (A^T A) u = A^T b
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        u = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        u, *_ = np.linalg.lstsq(A, b, rcond=None)     # rank-deficient fallback

    if not np.all(np.isfinite(u)):
        return None
    return np.array([[u[0], u[1], u[2]],
                     [u[3], u[4], u[5]]])


def apply_affine(M, pts):
    """Map (N,2) points through a (2,3) affine matrix."""
    pts = np.asarray(pts, float)
    return pts @ M[:, :2].T + M[:, 2][None, :]


def triangle_area(p0, p1, p2):
    """Unsigned area of the triangle p0 p1 p2 (half the cross product)."""
    return 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                     (p2[0] - p0[0]) * (p1[1] - p0[1]))


def is_well_spread(pts, min_area_frac=0.02):
    """Collinearity guard for a minimal sample.

    The absolute triangle area is meaningless on its own - three points 2 px
    apart always have a tiny area.  We normalise by the squared spread of the
    sample (its bounding-box diagonal), giving a scale-free measure of how far
    the sample is from collinear.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return False
    span = pts.max(axis=0) - pts.min(axis=0)
    diag2 = float(span[0] ** 2 + span[1] ** 2)
    if diag2 < 1e-6:
        return False
    area = triangle_area(pts[0], pts[1], pts[2])
    return (area / diag2) > min_area_frac


def decompose(M):
    """Geometric quantities of the linear part of an affine transform.

    Returns a dict with
        det        determinant of M (negative => reflection)
        scale      sqrt(|det|), the average linear magnification
        sv_ratio   sigma_max / sigma_min from the SVD - the anisotropy, i.e.
                   how much the transform stretches one direction relative to
                   the perpendicular one.  1.0 is a pure similarity.
        rotation   rotation angle in degrees of the closest rotation matrix
    """
    A = np.asarray(M, float)[:, :2]
    det = float(np.linalg.det(A))
    sv = np.linalg.svd(A, compute_uv=False)
    sv_ratio = float(sv[0] / max(sv[1], 1e-12))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return dict(det=det, scale=float(np.sqrt(abs(det))),
                sv_ratio=sv_ratio, rotation=rot % 360.0,
                sv=(float(sv[0]), float(sv[1])))


def passes_sanity(M, scale_range=(0.15, 3.0), max_sv_ratio=1.55,
                  allow_reflection=False):
    """Reject transforms that are geometrically impossible for a rigid packet.

    This is what stops the bounding box from shearing into a spike when a
    cluster is contaminated by outliers.
    """
    if M is None or not np.all(np.isfinite(M)):
        return False, dict(reason="non-finite")
    g = decompose(M)
    if not allow_reflection and g["det"] <= 0:
        return False, dict(reason="reflection", **g)
    if not (scale_range[0] <= g["scale"] <= scale_range[1]):
        return False, dict(reason="scale-out-of-range", **g)
    if g["sv_ratio"] > max_sv_ratio:
        return False, dict(reason="shear", **g)
    return True, g


def corners_of(w, h):
    """The four template corners in template pixel coordinates."""
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])


def project_corners(M, w, h):
    """Project the template corner rectangle into the scene through M."""
    return apply_affine(M, corners_of(w, h))


def reprojection_errors(M, src, dst):
    """Per-correspondence Euclidean residual ||M(src) - dst||."""
    pred = apply_affine(M, src)
    return np.linalg.norm(pred - np.asarray(dst, float), axis=1)
