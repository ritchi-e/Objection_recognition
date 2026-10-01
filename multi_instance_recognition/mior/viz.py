"""
mior.viz
========
Drawing helpers for the report figures.

Only primitive drawing calls are used (line / circle / polylines / putText /
rectangle) plus Matplotlib for the Hough-space heat map.  Nothing here affects
the recognition result.

A note on the final figure, per the assignment's "Final Cleanup" requirement:
the delivered detection image draws ONLY the projected quadrilaterals and a
small label.  Keypoint crosses and match lines are confined to the diagnostic
figures, because a final image sprayed with thousands of markers hides exactly
the thing it is supposed to show.
"""

import cv2
import numpy as np

# A colour-blind-safe qualitative palette (BGR), used for per-instance colours.
PALETTE = [
    (66, 133, 244),    # blue
    (52, 168, 83),     # green
    (26, 115, 232),    # deep blue
    (245, 124, 0),     # orange
    (211, 47, 47),     # red
    (156, 39, 176),    # purple
    (0, 172, 193),     # cyan
    (124, 179, 66),    # lime
]


def bgr(idx):
    return PALETTE[idx % len(PALETTE)]


def side_by_side(img_a, img_b, gap=24, bg=(28, 28, 30)):
    """Place two images side by side on a common canvas, vertically centred.

    Returns (canvas, offset_a, offset_b) where the offsets are the (x, y)
    translations that map each source image into the canvas.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    H, W = max(ha, hb), wa + gap + wb
    canvas = np.full((H, W, 3), bg, np.uint8)
    ya, yb = (H - ha) // 2, (H - hb) // 2
    canvas[ya:ya + ha, 0:wa] = img_a
    canvas[yb:yb + hb, wa + gap:wa + gap + wb] = img_b
    return canvas, (0, ya), (wa + gap, yb)


def draw_matches(img_t, img_s, tmpl_xy, scene_xy, colors=None, thickness=1,
                 radius=2, max_lines=None, rng=None, alpha=1.0):
    """Classic two-panel match visualisation, drawn by hand."""
    canvas, (ox_t, oy_t), (ox_s, oy_s) = side_by_side(img_t, img_s)
    n = len(tmpl_xy)
    order = np.arange(n)
    if max_lines is not None and n > max_lines:
        rng = rng or np.random.default_rng(0)
        order = rng.choice(n, size=max_lines, replace=False)

    layer = canvas.copy()
    for i in order:
        p = (int(round(tmpl_xy[i][0])) + ox_t, int(round(tmpl_xy[i][1])) + oy_t)
        q = (int(round(scene_xy[i][0])) + ox_s, int(round(scene_xy[i][1])) + oy_s)
        c = colors[i] if colors is not None else (0, 215, 255)
        cv2.line(layer, p, q, c, thickness, cv2.LINE_AA)
        cv2.circle(layer, p, radius, c, -1, cv2.LINE_AA)
        cv2.circle(layer, q, radius, c, -1, cv2.LINE_AA)

    if alpha < 1.0:
        canvas = cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)
    else:
        canvas = layer
    return canvas


def draw_keypoints(img, xy, color=(0, 200, 255), radius=2, thickness=-1):
    out = img.copy()
    for p in xy:
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), radius, color,
                   thickness, cv2.LINE_AA)
    return out


def draw_quad(img, corners, color, thickness=3, label=None, label_bg=True,
              corner_dots=True):
    """Draw one detection quadrilateral with an optional label chip."""
    pts = np.round(np.asarray(corners, float)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)

    if corner_dots:
        for p in pts.reshape(-1, 2):
            cv2.circle(img, tuple(p), thickness + 2, color, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), thickness, (255, 255, 255), -1, cv2.LINE_AA)

    # Mark the template's top edge so the recovered orientation is visible.
    c = pts.reshape(-1, 2)
    cv2.line(img, tuple(c[0]), tuple(c[1]), (255, 255, 255), max(1, thickness - 1),
             cv2.LINE_AA)

    if label:
        anchor = c[np.argmin(c[:, 1])]
        x, y = int(anchor[0]), int(anchor[1]) - 12
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        x = max(4, min(x, img.shape[1] - tw - 10))
        y = max(th + 8, y)
        if label_bg:
            cv2.rectangle(img, (x - 6, y - th - 7), (x + tw + 6, y + base + 1),
                          color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_dashed_quad(img, corners, color, thickness=2, dash=14):
    """Dashed outline - used for ground-truth polygons."""
    c = np.asarray(corners, float)
    for i in range(len(c)):
        p, q = c[i], c[(i + 1) % len(c)]
        d = np.linalg.norm(q - p)
        n = max(int(d / dash), 1)
        for k in range(n):
            if k % 2:
                continue
            a = p + (q - p) * (k / n)
            b = p + (q - p) * min((k + 1) / n, 1.0)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     color, thickness, cv2.LINE_AA)
    return img


def banner(img, text, height=46, color=(24, 24, 26), fg=(255, 255, 255),
           scale=0.8):
    """Add a caption strip above an image."""
    h, w = img.shape[:2]
    strip = np.full((height, w, 3), color, np.uint8)
    # Shrink the font until the caption fits, so long captions are never
    # silently clipped at the right-hand edge of the figure.
    thick = 2
    while scale > 0.32:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= w - 28:
            break
        scale -= 0.04
        if scale < 0.55:
            thick = 1
    cv2.putText(strip, text, (14, int(height * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
    return np.vstack([strip, img])


def stack_v(imgs, gap=10, bg=(28, 28, 30)):
    w = max(i.shape[1] for i in imgs)
    parts = []
    for k, im in enumerate(imgs):
        pad = np.full((im.shape[0], w - im.shape[1], 3), bg, np.uint8)
        parts.append(np.hstack([im, pad]) if pad.size else im)
        if k != len(imgs) - 1:
            parts.append(np.full((gap, w, 3), bg, np.uint8))
    return np.vstack(parts)
