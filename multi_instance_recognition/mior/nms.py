"""
mior.nms
========
Stage 5: duplicate suppression by polygon IoU, implemented from scratch.

WHY DUPLICATES HAPPEN
---------------------
Broadened Hough voting deliberately puts every match into 16 bins.  A strong
instance therefore produces several neighbouring bins that all clear the vote
threshold, and each of them can converge to (nearly) the same affine transform.
Without suppression the final image carries three or four almost-coincident
boxes on the same packet.

WHY POLYGON IoU AND NOT AXIS-ALIGNED IoU
----------------------------------------
Our detections are the projected template *quadrilaterals*, which are rotated
by arbitrary angles.  Collapsing them to axis-aligned boxes before computing
overlap badly overestimates the area of a 45-degree-rotated detection and
therefore merges instances that are actually distinct.  We compute the true
quadrilateral IoU:

    IoU(P, Q) = area(P and Q) / (area(P) + area(Q) - area(P and Q))

using two classical primitives written out below.

1. SHOELACE FORMULA for the area of a simple polygon with vertices
   (x_0,y_0) ... (x_{n-1},y_{n-1}):

       A = 1/2 | sum_i ( x_i * y_{i+1} - x_{i+1} * y_i ) |

2. SUTHERLAND-HODGMAN CLIPPING for the intersection of a subject polygon with a
   CONVEX clip polygon.  The subject is clipped successively against each edge
   of the clip polygon; for each edge we walk the subject's vertices and keep
   the ones inside the half-plane, inserting the edge-crossing point wherever
   the polygon crosses from inside to outside or back.  Since a projected
   rectangle under a non-degenerate affine map is always convex, this is exact
   here.

The half-plane test uses the sign of the 2D cross product

       cross = (b - a) x (p - a)

which is positive when p lies to the left of the directed edge a->b.  We
normalise every polygon to counter-clockwise order first so "inside" is
consistently "left of every edge".
"""

import numpy as np


def polygon_area_signed(poly):
    """Signed shoelace area.  Positive for counter-clockwise vertex order."""
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly):
    return abs(polygon_area_signed(poly))


def ensure_ccw(poly):
    """Return the polygon with counter-clockwise vertex order."""
    p = np.asarray(poly, float)
    return p if polygon_area_signed(p) >= 0 else p[::-1]


def _inside(p, a, b):
    """True if p is on the left of (or on) the directed edge a -> b."""
    return ((b[0] - a[0]) * (p[1] - a[1]) -
            (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9


def _edge_intersection(p, q, a, b):
    """Intersection of segment p->q with the infinite line a->b."""
    r = q - p
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return q
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / denom
    return p + t * r


def clip_polygon(subject, clip):
    """Sutherland-Hodgman: intersect `subject` with the CONVEX polygon `clip`."""
    out = [np.asarray(v, float) for v in ensure_ccw(subject)]
    clip = ensure_ccw(clip)

    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if not out:
            break
        inp, out = out, []
        prev = inp[-1]
        prev_in = _inside(prev, a, b)
        for cur in inp:
            cur_in = _inside(cur, a, b)
            if cur_in:
                if not prev_in:
                    out.append(_edge_intersection(prev, cur, a, b))
                out.append(cur)
            elif prev_in:
                out.append(_edge_intersection(prev, cur, a, b))
            prev, prev_in = cur, cur_in
    return np.array(out) if out else np.zeros((0, 2))


def polygon_iou(p, q):
    """Intersection-over-union of two convex polygons."""
    inter = clip_polygon(p, q)
    if len(inter) < 3:
        return 0.0
    ai, ap, aq = polygon_area(inter), polygon_area(p), polygon_area(q)
    union = ap + aq - ai
    return float(ai / union) if union > 1e-9 else 0.0


def centre_of(poly):
    return np.asarray(poly, float).mean(axis=0)


def nms(detections, iou_threshold=0.30, score_key=lambda d: d["score"]):
    """Greedy non-maximum suppression over quadrilateral detections.

    detections : list of dicts, each with a "corners" (4,2) entry.
    Highest-scoring detection wins; any remaining detection overlapping it by
    more than `iou_threshold` is discarded as a duplicate of the same physical
    object.

    Returns (kept, suppressed) as two lists, so the report can show exactly how
    many duplicate boxes the stage removed.
    """
    order = sorted(range(len(detections)), key=lambda i: -score_key(detections[i]))
    kept, suppressed = [], []
    kept_polys = []

    for i in order:
        poly = np.asarray(detections[i]["corners"], float)
        dup_of = None
        for j, kp in enumerate(kept_polys):
            if polygon_iou(poly, kp) > iou_threshold:
                dup_of = j
                break
        if dup_of is None:
            kept.append(detections[i])
            kept_polys.append(poly)
        else:
            d = dict(detections[i])
            d["suppressed_by"] = dup_of
            suppressed.append(d)
    return kept, suppressed
