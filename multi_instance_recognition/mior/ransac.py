"""
mior.ransac
===========
Stage 4b: robust estimation of the affine transform inside one Hough bin,
implemented from scratch.

A Hough bin is *mostly* clean but not perfectly clean.  Broadened voting, a
coarse 30-degree orientation quantisation and sheer bad luck all let a few
clutter matches share a bin with a real instance.  Those few outliers are
enough to wreck a plain least-squares fit, because least squares minimises the
sum of SQUARED residuals and a single match that is 300 px wrong contributes as
much as 900 matches that are 10 px wrong.  Hence RANSAC.

THE ALGORITHM
-------------
  repeat N times:
      1. draw a minimal sample of 3 correspondences from the bin
      2. reject the sample immediately if the 3 template points are nearly
         collinear (degenerate - see mior.geometry.is_well_spread)
      3. solve the exact 6x6 affine system on that sample
      4. reject the model if it fails the geometric sanity checks
         (reflection / implausible scale / excessive shear)
      5. count inliers: correspondences with reprojection error < tau
      6. keep the model with the most inliers
  finally:
      refit by least squares on ALL inliers, re-classify, and repeat that
      refinement a few times so the final transform uses every good match
      rather than the three lucky ones.

ADAPTIVE ITERATION COUNT
------------------------
The classic formula: to see at least one all-inlier sample with probability p,
given an inlier fraction w and a sample size m = 3,

    N = log(1 - p) / log(1 - w^m)

We recompute this each time the best inlier count improves, so a clean bin
terminates after a handful of iterations while a dirty one keeps trying up to
`max_iters`.

THE INLIER THRESHOLD
--------------------
A fixed pixel threshold is wrong here, because the instances differ in size by
more than a factor of two.  Eight pixels is tight on a 1.0-scale instance and
absurdly loose on a 0.42-scale one.  We therefore make tau proportional to the
projected size of the object under the hypothesis:

    tau = clip(tau_frac * max(W, H) * s_hypothesis, tau_min, tau_max)

with s_hypothesis read off the model itself via sqrt(det M).  This is also what
makes the threshold meaningful in the presence of the small perspective tilt in
the data: the affine model cannot fit a projective warp exactly, so residuals
grow with object size and the threshold must grow with it too.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import (apply_affine, decompose, is_well_spread, passes_sanity,
                       reprojection_errors, solve_affine_lsq)


@dataclass
class AffineFit:
    M: np.ndarray                       # (2,3) affine, template -> scene
    inliers: np.ndarray                 # indices into the correspondence set
    mean_error: float                   # mean reprojection error over inliers
    max_error: float
    n_candidates: int                   # size of the cluster it was fitted in
    geom: dict = field(default_factory=dict)
    iterations: int = 0

    @property
    def n_inliers(self):
        return len(self.inliers)

    @property
    def inlier_ratio(self):
        return self.n_inliers / max(self.n_candidates, 1)


def _tau_for(M, tmpl_wh, tau_frac, tau_min, tau_max):
    """Scale-adaptive inlier threshold (see module docstring)."""
    s = decompose(M)["scale"]
    return float(np.clip(tau_frac * max(tmpl_wh) * s, tau_min, tau_max))


def ransac_affine(src, dst, tmpl_wh, rng,
                  max_iters=600, min_iters=30, confidence=0.995,
                  tau_frac=0.045, tau_min=3.5, tau_max=22.0,
                  scale_range=(0.15, 3.0), max_sv_ratio=1.55,
                  refine_rounds=4):
    """Robustly fit template->scene affine to a set of correspondences.

    Parameters
    ----------
    src, dst : (N,2) template points and their scene counterparts
    tmpl_wh  : (W, H) of the template, used for the adaptive threshold
    rng      : numpy Generator, for reproducibility

    Returns
    -------
    AffineFit or None if no acceptable model was found.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 3:
        return None

    best_M, best_inl, best_count = None, np.empty(0, int), 0
    N = max_iters
    it = 0

    while it < N and it < max_iters:
        it += 1
        idx = rng.choice(n, size=3, replace=False)

        # (2) degeneracy guard on the minimal sample, on BOTH sides
        if not is_well_spread(src[idx]) or not is_well_spread(dst[idx]):
            continue

        # (3) exact fit on 3 points
        M = solve_affine_lsq(src[idx], dst[idx])
        if M is None:
            continue

        # (4) geometric sanity
        ok, _ = passes_sanity(M, scale_range, max_sv_ratio)
        if not ok:
            continue

        # (5) inlier count
        tau = _tau_for(M, tmpl_wh, tau_frac, tau_min, tau_max)
        err = reprojection_errors(M, src, dst)
        inl = np.flatnonzero(err < tau)

        if len(inl) > best_count:
            best_count, best_inl, best_M = len(inl), inl, M
            # adaptive iteration count
            w = best_count / n
            if 0 < w < 1:
                denom = np.log(max(1.0 - w ** 3, 1e-12))
                N = int(np.clip(np.log(1.0 - confidence) / denom,
                                min_iters, max_iters))

    if best_M is None or best_count < 3:
        return None

    # ---- (final) least-squares refinement on all inliers ------------------ #
    M = best_M
    inl = best_inl
    for _ in range(refine_rounds):
        if len(inl) < 3:
            break
        M_ref = solve_affine_lsq(src[inl], dst[inl])
        if M_ref is None:
            break
        ok, _ = passes_sanity(M_ref, scale_range, max_sv_ratio)
        if not ok:
            break
        tau = _tau_for(M_ref, tmpl_wh, tau_frac, tau_min, tau_max)
        err = reprojection_errors(M_ref, src, dst)
        new_inl = np.flatnonzero(err < tau)
        if len(new_inl) < 3:
            break
        # Converged when the inlier set stops changing.
        if len(new_inl) == len(inl) and np.array_equal(new_inl, inl):
            M = M_ref
            break
        M, inl = M_ref, new_inl

    ok, geom = passes_sanity(M, scale_range, max_sv_ratio)
    if not ok:
        return None

    err = reprojection_errors(M, src[inl], dst[inl])
    return AffineFit(M=M, inliers=inl,
                     mean_error=float(err.mean()), max_error=float(err.max()),
                     n_candidates=n, geom=geom, iterations=it)
