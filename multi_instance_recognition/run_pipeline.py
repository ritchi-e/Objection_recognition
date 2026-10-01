"""
run_pipeline.py
===============
End-to-end driver for Assignment 1: Multi-Instance Object Recognition in
Cluttered Scenes.

    python run_pipeline.py                                  # default dataset
    python run_pipeline.py --template mine.jpg --scene desk.jpg --no-gt

Everything after SIFT feature extraction is implemented from scratch inside the
`mior` package:

    mior.features   SIFT (OpenCV, permitted) -> keypoints + 128-D descriptors
    mior.matching   Euclidean distance matrix + Lowe ratio test, scene->template
    mior.hough      4D Generalised Hough Transform with broadened voting
    mior.geometry   affine Ax=b least squares + degeneracy / shear guards
    mior.ransac     robust iterative estimation with adaptive iteration count
    mior.pipeline   greedy extraction loop + multi-test verification
    mior.nms        polygon IoU + non-maximum suppression

The script writes every figure the report needs into ./outputs and a machine
readable results.json next to them.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from mior import features as F
from mior import hough as HG
from mior import matching as MT
from mior import viz as VZ
from mior.geometry import project_corners
from mior.nms import polygon_iou
from mior.pipeline import DEFAULT_CFG, detect_instances

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", default=os.path.join(HERE, "data", "template.png"))
    p.add_argument("--scene", default=os.path.join(HERE, "data", "scene.png"))
    p.add_argument("--gt", default=os.path.join(HERE, "data", "ground_truth.json"))
    p.add_argument("--no-gt", action="store_true",
                   help="skip the quantitative evaluation (use for your own photos)")
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    p.add_argument("--ratio", type=float, default=DEFAULT_CFG["ratio"])
    p.add_argument("--min-votes", type=int, default=DEFAULT_CFG["min_votes"])
    p.add_argument("--min-inliers", type=int, default=DEFAULT_CFG["min_inliers"])
    p.add_argument("--min-zncc", type=float, default=DEFAULT_CFG["min_zncc"])
    p.add_argument("--iou-nms", type=float, default=DEFAULT_CFG["iou_nms"])
    p.add_argument("--rootsift", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()

    template = cv2.imread(a.template, cv2.IMREAD_COLOR)
    scene = cv2.imread(a.scene, cv2.IMREAD_COLOR)
    if template is None or scene is None:
        raise SystemExit(f"could not read {a.template} / {a.scene}")

    th, tw = template.shape[:2]
    sh, sw = scene.shape[:2]
    tmpl_gray = F.to_gray(template)
    scene_gray = F.to_gray(scene)

    print("=" * 74)
    print("MULTI-INSTANCE OBJECT RECOGNITION IN CLUTTERED SCENES")
    print("=" * 74)
    print(f"template : {os.path.basename(a.template)}  {tw} x {th}")
    print(f"scene    : {os.path.basename(a.scene)}  {sw} x {sh}")

    # ---- 1. SIFT --------------------------------------------------------- #
    t0 = time.time()
    ft = F.extract_sift(template)
    fs = F.extract_sift(scene)
    t_sift = time.time() - t0
    print(f"\n[1] SIFT           template {len(ft):6d} keypoints"
          f"   scene {len(fs):6d} keypoints    ({t_sift:.2f}s)")

    td, sd = ft.desc, fs.desc
    if a.rootsift:
        td, sd = F.rootsift(td), F.rootsift(sd)

    # ---- 2. correspondence ----------------------------------------------- #
    t0 = time.time()
    matches, raw = MT.match_scene_to_template(sd, td, ratio=a.ratio)
    idx1, d1, d2, r = raw
    naive = MT.all_nearest_matches(idx1, d1, d2, r)
    t_match = time.time() - t0
    print(f"[2] Matching       {len(naive):6d} nearest-neighbour matches "
          f"(no test)  ->  {len(matches):5d} survive Lowe ratio < {a.ratio}"
          f"   ({t_match:.2f}s)")
    print(f"                   rejection rate "
          f"{100 * (1 - len(matches) / max(len(naive), 1)):.1f}%")

    # ---- 3. Hough -------------------------------------------------------- #
    t0 = time.time()
    cx, cy, log2s, dtheta = HG.pose_votes(fs, ft, matches, (tw, th))
    table = HG.build_hough_table(cx, cy, log2s, dtheta, (tw, th),
                                 scene_wh=(sw, sh))
    bins = HG.ranked_bins(table, a.min_votes)
    t_hough = time.time() - t0
    print(f"[3] 4D Hough       {len(table):6d} occupied bins, "
          f"{len(bins):5d} with >= {a.min_votes} votes; "
          f"top bin holds {len(bins[0][1]) if bins else 0} votes  ({t_hough:.2f}s)")

    # ---- 4-6. greedy extraction ------------------------------------------ #
    print(f"[4] Greedy extraction (RANSAC + LSQ affine + verification)")
    t0 = time.time()
    cfg = dict(ratio=a.ratio, min_votes=a.min_votes, min_inliers=a.min_inliers,
               min_zncc=a.min_zncc, iou_nms=a.iou_nms)
    res = detect_instances(fs, ft, matches, (tw, th), (sw, sh),
                           tmpl_gray, scene_gray, cfg=cfg, seed=a.seed)
    t_det = time.time() - t0
    dets = res["detections"]
    print(f"    -> {len(res['raw'])} raw detections, "
          f"{len(res['suppressed'])} removed by IoU-NMS, "
          f"{len(res['rejected'])} hypotheses failed verification"
          f"   ({t_det:.2f}s)")

    # order detections left-to-right, top-to-bottom for stable labelling
    dets = sorted(dets, key=lambda d: (d.corners.mean(axis=0)[1] // 200,
                                       d.corners.mean(axis=0)[0]))

    print(f"\n[5] FINAL DETECTIONS: {len(dets)}")
    print(f"    {'#':<3}{'centre (x,y)':>16}{'scale':>8}{'rot deg':>9}"
          f"{'inliers':>9}{'ratio':>7}{'err px':>8}{'ZNCC':>7}{'shear':>7}")
    for i, d in enumerate(dets):
        c = d.corners.mean(axis=0)
        print(f"    {i:<3}{f'({c[0]:.0f}, {c[1]:.0f})':>16}{d.scale:>8.3f}"
              f"{d.rotation:>9.1f}{d.n_inliers:>9d}{d.inlier_ratio:>7.2f}"
              f"{d.mean_error:>8.2f}{d.zncc:>7.2f}{d.sv_ratio:>7.3f}")

    # ---- figures --------------------------------------------------------- #
    print("\n[6] Writing figures ...")
    figs = make_figures(a.out, template, scene, ft, fs, matches, naive,
                        cx, cy, dets, res, (tw, th), (sw, sh))

    # ---- evaluation ------------------------------------------------------ #
    evaluation = None
    if not a.no_gt and os.path.exists(a.gt):
        evaluation = evaluate(a.gt, dets, scene, a.out, (tw, th))

    # ---- results.json ---------------------------------------------------- #
    out = dict(
        template=os.path.basename(a.template), scene=os.path.basename(a.scene),
        template_size=[tw, th], scene_size=[sw, sh],
        keypoints=dict(template=len(ft), scene=len(fs)),
        matching=dict(ratio=a.ratio, nearest_neighbour_matches=len(naive),
                      after_ratio_test=len(matches),
                      rejection_rate=round(1 - len(matches) / max(len(naive), 1), 4)),
        hough=dict(occupied_bins=len(table), bins_over_threshold=len(bins),
                   min_votes=a.min_votes,
                   top_bin_votes=int(len(bins[0][1])) if bins else 0),
        detections=[d.as_dict() for d in dets],
        n_suppressed_by_nms=len(res["suppressed"]),
        n_rejected_by_verification=len(res["rejected"]),
        rejected_examples=res["rejected"][:12],
        rounds=res["rounds"],
        timing=dict(sift=round(t_sift, 3), matching=round(t_match, 3),
                    hough=round(t_hough, 3), detection=round(t_det, 3),
                    total=round(time.time() - t_all, 3)),
        evaluation=evaluation, figures=figs,
    )
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTotal runtime {time.time() - t_all:.2f}s")
    print(f"Outputs in {a.out}")


# --------------------------------------------------------------------------- #
def make_figures(outdir, template, scene, ft, fs, matches, naive,
                 cx, cy, dets, res, tmpl_wh, scene_wh):
    tw, th = tmpl_wh
    sw, sh = scene_wh
    rng = np.random.default_rng(1)
    written = {}

    def save(name, img):
        p = os.path.join(outdir, name)
        cv2.imwrite(p, img)
        written[name] = os.path.relpath(p, outdir)
        print(f"    {name}")

    # --- 00 originals ----------------------------------------------------- #
    # side_by_side handles either image being taller/wider than the other
    # (common with phone photos where the template is a portrait close-up).
    inputs, _, _ = VZ.side_by_side(template, scene, gap=40)
    save("fig00_inputs.png",
         VZ.banner(inputs,
                   "Left: TEMPLATE (isolated)  |  Right: QUERY SCENE (cluttered)"))

    # --- 01 raw keypoints ------------------------------------------------- #
    save("fig01_sift_keypoints.png",
         VZ.banner(VZ.draw_keypoints(scene, fs.xy, (0, 200, 255), 1),
                   f"[1] SIFT keypoints in the scene: {len(fs)} "
                   f"(template has {len(ft)})"))

    # --- 02 naive matching, NO ratio test --------------------------------- #
    n_show = 1500
    sel = rng.choice(len(naive), size=min(n_show, len(naive)), replace=False)
    naive_img = VZ.draw_matches(
        template, scene, ft.xy[naive.tmpl_idx[sel]], fs.xy[naive.scene_idx[sel]],
        colors=[(0, 90, 255)] * len(sel), thickness=1, radius=1, alpha=0.75)
    save("fig02_naive_all_nn_matches.png",
         VZ.banner(naive_img,
                   f"[2a] NAIVE: every nearest-neighbour match, no ratio test "
                   f"({len(naive)} total, {len(sel)} drawn)"))

    # --- 03 after the ratio test ------------------------------------------ #
    ratio_img = VZ.draw_matches(
        template, scene, ft.xy[matches.tmpl_idx], fs.xy[matches.scene_idx],
        colors=[(0, 215, 255)] * len(matches), thickness=1, radius=2, alpha=0.9)
    save("fig03_after_lowe_ratio.png",
         VZ.banner(ratio_img,
                   f"[2b] After Lowe ratio test: {len(matches)} matches "
                   f"- still dominated by clutter, no grouping yet"))

    # --- 04 Hough vote space ---------------------------------------------- #
    acc = HG.vote_histogram_2d(cx, cy, (sw, sh), bin_px=24)
    heat = acc / max(acc.max(), 1e-6)
    heat_img = cv2.applyColorMap((np.power(heat, 0.45) * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
    heat_img = cv2.resize(heat_img, (sw, sh), interpolation=cv2.INTER_NEAREST)
    blend = cv2.addWeighted(scene, 0.40, heat_img, 0.60, 0)
    for d in dets:
        c = d.corners.mean(axis=0).astype(int)
        cv2.circle(blend, tuple(c), 16, (255, 255, 255), 2, cv2.LINE_AA)
    save("fig04_hough_vote_space.png",
         VZ.banner(np.hstack([heat_img, blend]),
                   "[3] 4D Hough space marginalised onto (x,y): predicted object "
                   "centres.  Right: overlaid on the scene, circles = extracted peaks"))

    # --- 05 per-instance inlier clusters ---------------------------------- #
    clus = scene.copy()
    for i, d in enumerate(dets):
        c = VZ.bgr(i)
        for p in fs.xy[d.inlier_scene_idx]:
            cv2.circle(clus, (int(p[0]), int(p[1])), 4, c, -1, cv2.LINE_AA)
            cv2.circle(clus, (int(p[0]), int(p[1])), 4, (255, 255, 255), 1, cv2.LINE_AA)
        VZ.draw_quad(clus, d.corners, c, 2, corner_dots=False)
    save("fig05_inlier_clusters.png",
         VZ.banner(clus, "[4] RANSAC inliers per extracted instance "
                         "(one colour per detection)"))

    # --- 06 raw detections before NMS ------------------------------------- #
    pre = scene.copy()
    for i, d in enumerate(res["raw"]):
        VZ.draw_quad(pre, d.corners, VZ.bgr(i), 2, corner_dots=False)
    post = scene.copy()
    for i, d in enumerate(dets):
        VZ.draw_quad(post, d.corners, VZ.bgr(i), 3, corner_dots=False)
    save("fig06_nms_before_after.png",
         np.hstack([VZ.banner(pre, f"[5a] Before NMS: {len(res['raw'])} raw boxes"),
                    VZ.banner(post, f"[5b] After IoU-NMS: {len(dets)} boxes")]))

    # --- 07 FINAL clean output -------------------------------------------- #
    final = scene.copy()
    for i, d in enumerate(dets):
        VZ.draw_quad(final, d.corners, VZ.bgr(i), 3,
                     label=f"#{i+1}  s={d.scale:.2f}  {d.rotation:.0f}deg")
    save("fig07_final_detections.png",
         VZ.banner(final, f"[6] FINAL: {len(dets)} instances localised "
                          f"(white edge = template top edge)"))
    cv2.imwrite(os.path.join(outdir, "final_detections_clean.png"),
                final)     # unbannered copy

    # --- 08 the headline comparison for the report ------------------------ #
    naive_small = cv2.resize(naive_img, (final.shape[1] * 2, int(
        naive_img.shape[0] * final.shape[1] * 2 / naive_img.shape[1])))
    left = VZ.banner(cv2.resize(naive_img, (sw, int(naive_img.shape[0] * sw /
                                                    naive_img.shape[1]))),
                     "NAIVE feature matching  ->  chaos")
    right = VZ.banner(cv2.resize(final, (sw, int(final.shape[0]))),
                      "FULL PIPELINE  ->  clean localisation")
    save("fig08_comparison_naive_vs_final.png", VZ.stack_v([left, right]))
    del naive_small

    return written


# --------------------------------------------------------------------------- #
def evaluate(gt_path, dets, scene, outdir, tmpl_wh):
    """Quantitative comparison against the dataset's ground-truth polygons."""
    with open(gt_path) as f:
        gt = json.load(f)
    inst = gt["instances"]
    tw, th = tmpl_wh

    print(f"\n[7] EVALUATION against {len(inst)} ground-truth instances")
    iou_mat = np.zeros((len(inst), len(dets)))
    for i, g in enumerate(inst):
        for j, d in enumerate(dets):
            iou_mat[i, j] = polygon_iou(np.array(g["corners"]), d.corners)

    rows, used = [], set()
    for i, g in enumerate(inst):
        j = int(np.argmax(iou_mat[i])) if len(dets) else -1
        iou = float(iou_mat[i, j]) if j >= 0 else 0.0
        if iou < 0.30 or j in used:
            rows.append(dict(instance=g["name"], matched=False, iou=round(iou, 3),
                             corner_error_px=None,
                             gt_scale=g["scale"],
                             gt_rotation=g["rotation_deg"] % 360.0,
                             occlusion_pct=g.get("occlusion_pct")))
            continue
        used.add(j)
        ce = float(np.linalg.norm(np.array(g["corners"]) - dets[j].corners,
                                  axis=1).mean())
        rows.append(dict(instance=g["name"], matched=True, detection=j,
                         iou=round(iou, 3), corner_error_px=round(ce, 2),
                         gt_scale=g["scale"], est_scale=round(dets[j].scale, 3),
                         gt_rotation=g["rotation_deg"] % 360.0,
                         est_rotation=round(dets[j].rotation, 1),
                         occlusion_pct=g.get("occlusion_pct"),
                         n_inliers=dets[j].n_inliers))

    tp = sum(1 for r in rows if r["matched"])
    fn = len(inst) - tp
    fp = len(dets) - tp
    prec = tp / max(len(dets), 1)
    rec = tp / max(len(inst), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"    {'inst':<6}{'occl%':>7}{'gt scale':>10}{'est':>8}"
          f"{'gt rot':>9}{'est':>9}{'IoU':>8}{'corner err':>12}")
    for r in rows:
        if r["matched"]:
            print(f"    {r['instance']:<6}{r['occlusion_pct']:>7.1f}"
                  f"{r['gt_scale']:>10.2f}{r['est_scale']:>8.2f}"
                  f"{r['gt_rotation']:>9.1f}{r['est_rotation']:>9.1f}"
                  f"{r['iou']:>8.3f}{r['corner_error_px']:>12.2f}")
        else:
            print(f"    {r['instance']:<6}{r['occlusion_pct']:>7.1f}"
                  f"{r['gt_scale']:>10.2f}{'--':>8}{r['gt_rotation']:>9.1f}"
                  f"{'--':>9}{r['iou']:>8.3f}{'MISSED':>12}")
    print(f"\n    TP={tp}  FP={fp}  FN={fn}   precision={prec:.3f}  "
          f"recall={rec:.3f}  F1={f1:.3f}")

    # Explicit false-positive-trap accounting.
    decoys = gt.get("decoys", [])
    dec_hits = 0
    for d in decoys:
        p = np.array(d["corners"])
        if any(polygon_iou(p, x.corners) > 0.20 for x in dets):
            dec_hits += 1
    if decoys:
        print(f"    decoy packets present: {len(decoys)}   "
              f"falsely detected: {dec_hits}")
    ious = [r["iou"] for r in rows if r["matched"]]
    ces = [r["corner_error_px"] for r in rows if r["matched"]]
    if ious:
        print(f"    mean IoU={np.mean(ious):.3f}   "
              f"mean corner error={np.mean(ces):.2f} px")

    # overlay figure
    ov = scene.copy()
    for d in decoys:
        VZ.draw_dashed_quad(ov, np.array(d["corners"]), (90, 90, 255), 2)
    for g in inst:
        VZ.draw_dashed_quad(ov, np.array(g["corners"]), (255, 255, 255), 2)
    for j, d in enumerate(dets):
        VZ.draw_quad(ov, d.corners, VZ.bgr(j), 3, corner_dots=False)
    for r in rows:
        if r["matched"]:
            g = next(x for x in inst if x["name"] == r["instance"])
            c = np.array(g["corners"]).mean(axis=0).astype(int)
            cv2.putText(ov, f"IoU {r['iou']:.2f}", (c[0] - 46, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(ov, f"IoU {r['iou']:.2f}", (c[0] - 46, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    p = os.path.join(outdir, "fig09_evaluation_vs_ground_truth.png")
    cv2.imwrite(p, VZ.banner(ov, "[7] Dashed white = ground truth, dashed red = "
                                 "decoy packet (must NOT be detected), solid = detection"))
    print(f"    fig09_evaluation_vs_ground_truth.png")

    return dict(per_instance=rows, tp=tp, fp=fp, fn=fn,
                n_decoys=len(decoys), decoys_falsely_detected=dec_hits,
                precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                mean_iou=round(float(np.mean(ious)), 4) if ious else None,
                mean_corner_error_px=round(float(np.mean(ces)), 3) if ces else None)


if __name__ == "__main__":
    main()
