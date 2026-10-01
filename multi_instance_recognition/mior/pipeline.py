"""
mior.pipeline
=============
Stage 6: the greedy extraction loop that ties everything together, plus the
false-positive verification test.

THE GREEDY EXTRACTION LOOP (sequential inlier subtraction)
----------------------------------------------------------
One RANSAC fit can only ever describe ONE instance, because a single affine
transform maps the template to a single place.  To find all of them we run the
loop the assignment describes:

    1. build the 4D Hough table from the currently ACTIVE matches
    2. take the bins in order of vote count
    3. for the first bin that yields a verified affine fit:
         - emit a detection
         - DEACTIVATE that fit's inlier matches
    4. go back to step 1 with the reduced match set
    5. stop when no remaining bin produces a verified detection

Subtracting the inliers is what makes the loop terminate and what stops the
strongest instance from being re-discovered over and over.  Rebuilding the
Hough table after each extraction (rather than reusing the first one) matters
too: removing 60 matches changes which of the remaining bins is now the
strongest, and lets a weak, heavily occluded instance rise to the top once the
dominant one is out of the way.

VERIFICATION - REJECTING BACKGROUND CLUTTER
-------------------------------------------
A bin that clears the 3-vote threshold is a hypothesis, not a detection.  In a
scene with thousands of clutter keypoints, some bins collect three mutually
consistent accidents.  A detection is only emitted if it passes ALL of:

  1. support        n_inliers >= min_inliers.  Three points can always be fit
                    exactly, so a 3-inlier "fit" carries no evidence at all.
  2. consistency    inlier_ratio within the cluster >= min_inlier_ratio, i.e.
                    the bin really was dominated by one coherent pose.
  3. residual       mean reprojection error <= max_mean_error (scaled by the
                    hypothesised object size) - the explicit error threshold.
  4. geometry       no reflection, plausible scale, bounded shear
                    (enforced inside mior.ransac via mior.geometry).
  5. spread         the inliers must not all clump in one corner of the
                    template.  A fit supported only by a 20x20 px patch says
                    nothing about where the other three corners are, and is the
                    classic source of wildly sheared boxes.
  6. appearance     zero-mean normalised cross-correlation between the template
                    and the scene region back-warped into template coordinates.
                    This is the appearance-level check that finally separates
                    the real packet from the decoy Basmati box, which can share
                    a few text-like features but does not correlate.

ZNCC, computed from scratch
---------------------------
    ZNCC(a, b) = sum_i (a_i - mean a)(b_i - mean b)
                 / sqrt( sum_i (a_i - mean a)^2 * sum_i (b_i - mean b)^2 )

evaluated over the template support after sampling the scene at the affine
pre-image of every template pixel (inverse mapping + bilinear interpolation,
implemented here rather than with cv2.warpAffine).  It ranges from -1 to 1 and
is invariant to any linear brightness/contrast change, which is essential
because each instance in the scene has a different exposure.
"""

from dataclasses import dataclass, field

import numpy as np

from . import hough
from .geometry import (apply_affine, decompose, project_corners,
                       reprojection_errors)
from .nms import nms, polygon_area
from .ransac import ransac_affine


# --------------------------------------------------------------------------- #
#  Appearance check
# --------------------------------------------------------------------------- #
def back_warp_zncc(template_gray, scene_gray, M, step=2):
    """ZNCC between the template and the scene region it maps onto.

    We walk the TEMPLATE pixel grid (sub-sampled by `step` for speed), push
    each pixel through the affine M to find where it lands in the scene, and
    sample the scene there by bilinear interpolation.  Pixels that land outside
    the scene are dropped.
    """
    th, tw = template_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]

    ys, xs = np.mgrid[0:th:step, 0:tw:step]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    dst = apply_affine(M, pts)

    u, v = dst[:, 0], dst[:, 1]
    ok = (u >= 0) & (u <= sw - 1.001) & (v >= 0) & (v <= sh - 1.001)
    if ok.sum() < 50:
        return 0.0, 0.0

    u, v = u[ok], v[ok]
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = u - x0, v - y0

    g = scene_gray.astype(np.float64)
    top = g[y0, x0] * (1 - fx) + g[y0, x1] * fx
    bot = g[y1, x0] * (1 - fx) + g[y1, x1] * fx
    b = top * (1 - fy) + bot * fy

    a = template_gray.astype(np.float64)[ys.ravel()[ok], xs.ravel()[ok]]

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float(a @ a) * float(b @ b))
    if denom < 1e-9:
        return 0.0, float(ok.mean())
    return float((a @ b) / denom), float(ok.mean())


def inlier_spread(tmpl_pts, tmpl_wh):
    """Fraction of the template's area spanned by the inliers' bounding box."""
    if len(tmpl_pts) < 3:
        return 0.0
    span = tmpl_pts.max(axis=0) - tmpl_pts.min(axis=0)
    W, H = tmpl_wh
    return float((span[0] * span[1]) / max(W * H, 1e-9))


# --------------------------------------------------------------------------- #
#  Detection record
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    M: np.ndarray
    corners: np.ndarray
    inlier_scene_idx: np.ndarray
    inlier_tmpl_idx: np.ndarray
    n_inliers: int
    inlier_ratio: float
    mean_error: float
    scale: float
    rotation: float
    sv_ratio: float
    zncc: float
    spread: float
    bin_key: tuple
    bin_votes: int
    score: float
    checks: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(M=self.M.tolist(), corners=self.corners.tolist(),
                    n_inliers=int(self.n_inliers),
                    inlier_ratio=round(float(self.inlier_ratio), 4),
                    mean_error=round(float(self.mean_error), 3),
                    scale=round(float(self.scale), 4),
                    rotation_deg=round(float(self.rotation), 2),
                    sv_ratio=round(float(self.sv_ratio), 4),
                    zncc=round(float(self.zncc), 4),
                    inlier_spread=round(float(self.spread), 4),
                    bin=list(self.bin_key), bin_votes=int(self.bin_votes),
                    score=round(float(self.score), 3))


# --------------------------------------------------------------------------- #
#  Verification
# --------------------------------------------------------------------------- #
def verify(fit, tmpl_pts, tmpl_wh, template_gray, scene_gray, cfg):
    """Apply every acceptance test.  Returns (ok, info_dict)."""
    W, H = tmpl_wh
    g = decompose(fit.M)
    tau = float(np.clip(cfg["max_error_frac"] * max(W, H) * g["scale"],
                        cfg["max_error_min"], cfg["max_error_max"]))
    spread = inlier_spread(tmpl_pts, tmpl_wh)
    z, cover = back_warp_zncc(template_gray, scene_gray, fit.M)

    quad = project_corners(fit.M, W, H)
    area = polygon_area(quad)

    checks = {
        "n_inliers":      (fit.n_inliers, cfg["min_inliers"],
                           fit.n_inliers >= cfg["min_inliers"]),
        "inlier_ratio":   (round(fit.inlier_ratio, 3), cfg["min_inlier_ratio"],
                           fit.inlier_ratio >= cfg["min_inlier_ratio"]),
        "mean_error":     (round(fit.mean_error, 2), round(tau, 2),
                           fit.mean_error <= tau),
        "inlier_spread":  (round(spread, 3), cfg["min_spread"],
                           spread >= cfg["min_spread"]),
        "zncc":           (round(z, 3), cfg["min_zncc"], z >= cfg["min_zncc"]),
        "area":           (int(area), cfg["min_area"], area >= cfg["min_area"]),
    }
    ok = all(c[2] for c in checks.values())
    return ok, dict(checks=checks, geom=g, zncc=z, coverage=cover,
                    spread=spread, quad=quad, tau=tau)


DEFAULT_CFG = dict(
    ratio=0.80,             # Lowe's rho
    min_votes=3,            # Hough bin threshold (assignment: >= 3)
    min_inliers=6,          # 3 would be a tautology; 6 is real evidence
    min_inlier_ratio=0.30,
    max_error_frac=0.045,   # error threshold as a fraction of projected size
    max_error_min=3.5,
    max_error_max=18.0,
    min_spread=0.05,        # inliers must span >= 5% of the template area
    min_zncc=0.12,          # appearance floor; survives >50% occlusion
    min_area=400.0,         # px^2
    max_detections=12,
    bins_per_round=250,
    iou_nms=0.30,
    ransac_iters=800,
    scale_range=(0.18, 2.6),
    max_sv_ratio=1.55,
)


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #
def detect_instances(scene_feats, tmpl_feats, matches, tmpl_wh, scene_wh,
                     template_gray, scene_gray, cfg=None, seed=7, verbose=True):
    """Run the full greedy multi-instance extraction.

    Returns
    -------
    dict with keys:
        detections  : list[Detection] surviving NMS
        suppressed  : list[dict] detections removed as duplicates
        rejected    : list[dict] hypotheses that failed verification
        rounds      : per-round bookkeeping for the report
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rng = np.random.default_rng(seed)

    cx, cy, log2s, dtheta = hough.pose_votes(scene_feats, tmpl_feats,
                                             matches, tmpl_wh)
    scene_xy = scene_feats.xy[matches.scene_idx]
    tmpl_xy = tmpl_feats.xy[matches.tmpl_idx]

    active = np.ones(len(matches), bool)
    raw_detections, rejected, rounds = [], [], []

    for rnd in range(cfg["max_detections"] * 2):
        idx = np.flatnonzero(active)
        if len(idx) < cfg["min_votes"]:
            break

        table = hough.build_hough_table(cx[idx], cy[idx], log2s[idx],
                                        dtheta[idx], tmpl_wh,
                                        scene_wh=scene_wh)
        bins = hough.ranked_bins(table, cfg["min_votes"])
        if not bins:
            break

        emitted = None
        for bin_key, local in bins[:cfg["bins_per_round"]]:
            gidx = idx[local]
            src, dst = tmpl_xy[gidx], scene_xy[gidx]

            # A bin of n matches admits only C(n,3) distinct minimal samples;
            # there is no point running more RANSAC iterations than that.
            n = len(gidx)
            max_it = int(min(cfg["ransac_iters"], max(30, n * (n - 1) * (n - 2))))

            fit = ransac_affine(src, dst, tmpl_wh, rng, max_iters=max_it,
                                scale_range=cfg["scale_range"],
                                max_sv_ratio=cfg["max_sv_ratio"])
            if fit is None:
                continue

            ok, info = verify(fit, src[fit.inliers], tmpl_wh,
                              template_gray, scene_gray, cfg)
            if not ok:
                rejected.append(dict(bin=list(bin_key), votes=int(n),
                                     n_inliers=int(fit.n_inliers),
                                     checks=info["checks"]))
                continue

            det = Detection(
                M=fit.M, corners=info["quad"],
                inlier_scene_idx=matches.scene_idx[gidx[fit.inliers]],
                inlier_tmpl_idx=matches.tmpl_idx[gidx[fit.inliers]],
                n_inliers=fit.n_inliers, inlier_ratio=fit.inlier_ratio,
                mean_error=fit.mean_error, scale=info["geom"]["scale"],
                rotation=info["geom"]["rotation"],
                sv_ratio=info["geom"]["sv_ratio"], zncc=info["zncc"],
                spread=info["spread"], bin_key=tuple(int(b) for b in bin_key),
                bin_votes=int(n),
                # Score favours a fit with many, well-spread, low-residual
                # inliers that also looks like the template.
                score=fit.n_inliers * (0.5 + 0.5 * max(info["zncc"], 0.0)),
                checks=info["checks"])
            raw_detections.append(det)

            # sequential inlier subtraction
            active[gidx[fit.inliers]] = False
            emitted = det
            break

        rounds.append(dict(round=rnd, active_matches=int(len(idx)),
                           bins=int(len(bins)),
                           emitted=None if emitted is None else
                           dict(inliers=emitted.n_inliers,
                                scale=round(emitted.scale, 3),
                                rot=round(emitted.rotation, 1))))
        if emitted is None:
            break
        if verbose:
            print(f"  round {rnd}: {len(idx):5d} active matches, {len(bins):5d} bins"
                  f"  -> detection with {emitted.n_inliers:3d} inliers "
                  f"(s={emitted.scale:.2f}, rot={emitted.rotation:6.1f} deg, "
                  f"zncc={emitted.zncc:.2f})")

    if not cfg.get("apply_nms", True):
        return dict(detections=raw_detections, suppressed=[],
                    raw=raw_detections, rejected=rejected, rounds=rounds, cfg=cfg)

    kept, suppressed = nms([d.__dict__ | {"corners": d.corners,
                                          "score": d.score, "_obj": d}
                            for d in raw_detections],
                           iou_threshold=cfg["iou_nms"])

    return dict(detections=[k["_obj"] for k in kept],
                suppressed=[s["_obj"] for s in suppressed],
                raw=raw_detections, rejected=rejected, rounds=rounds, cfg=cfg)


# --------------------------------------------------------------------------- #
#  Ablation helper: every bin fitted independently, no inlier subtraction
# --------------------------------------------------------------------------- #
def detect_all_bins_no_subtraction(scene_feats, tmpl_feats, matches, tmpl_wh,
                                   scene_wh, template_gray, scene_gray,
                                   cfg=None, seed=7, max_bins=400):
    """Fit and verify EVERY Hough bin once, without removing any inliers.

    This is the ablation that shows why non-maximum suppression is needed.
    Broadened voting means a strong instance owns a whole neighbourhood of
    bins, and each of those bins independently converges to essentially the
    same affine transform.  Without the greedy subtraction loop the raw output
    is a pile of near-coincident boxes; IoU-NMS is what turns it back into one
    box per physical object.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rng = np.random.default_rng(seed)

    cx, cy, log2s, dtheta = hough.pose_votes(scene_feats, tmpl_feats,
                                             matches, tmpl_wh)
    scene_xy = scene_feats.xy[matches.scene_idx]
    tmpl_xy = tmpl_feats.xy[matches.tmpl_idx]

    table = hough.build_hough_table(cx, cy, log2s, dtheta, tmpl_wh,
                                    scene_wh=scene_wh)
    out = []
    for bin_key, local in hough.ranked_bins(table, cfg["min_votes"])[:max_bins]:
        src, dst = tmpl_xy[local], scene_xy[local]
        n = len(local)
        max_it = int(min(cfg["ransac_iters"], max(30, n * (n - 1) * (n - 2))))
        fit = ransac_affine(src, dst, tmpl_wh, rng, max_iters=max_it,
                            scale_range=cfg["scale_range"],
                            max_sv_ratio=cfg["max_sv_ratio"])
        if fit is None:
            continue
        ok, info = verify(fit, src[fit.inliers], tmpl_wh,
                          template_gray, scene_gray, cfg)
        if not ok:
            continue
        out.append(Detection(
            M=fit.M, corners=info["quad"],
            inlier_scene_idx=matches.scene_idx[local[fit.inliers]],
            inlier_tmpl_idx=matches.tmpl_idx[local[fit.inliers]],
            n_inliers=fit.n_inliers, inlier_ratio=fit.inlier_ratio,
            mean_error=fit.mean_error, scale=info["geom"]["scale"],
            rotation=info["geom"]["rotation"], sv_ratio=info["geom"]["sv_ratio"],
            zncc=info["zncc"], spread=info["spread"],
            bin_key=tuple(int(b) for b in bin_key), bin_votes=int(n),
            score=fit.n_inliers * (0.5 + 0.5 * max(info["zncc"], 0.0)),
            checks=info["checks"]))
    return out
