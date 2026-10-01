"""
make_report.py
==============
Build the assignment report from outputs/results.json and the pipeline figures.

Writes:
  report/report.tex   — LaTeX source
  report/report.pdf   — PDF (via pdflatex if available, otherwise ReportLab)
  MIOR_report.pdf     — copy at the package root / workspace root
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import cv2
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
FIGSRC = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(FIGSRC, "results.json")))
mt = R["matching"]
hg = R["hough"]
dets = R["detections"]
timing = R.get("timing", {})


def f(x, n=2):
    return f"{float(x):.{n}f}"


# --------------------------------------------------------------------------- #
#  Shared numbers
# --------------------------------------------------------------------------- #
stats = {
    "tmpl_w": R["template_size"][0],
    "tmpl_h": R["template_size"][1],
    "scene_w": R["scene_size"][0],
    "scene_h": R["scene_size"][1],
    "nkpt": R["keypoints"]["template"],
    "nkps": R["keypoints"]["scene"],
    "rho": mt["ratio"],
    "nraw": mt["nearest_neighbour_matches"],
    "nmatch": mt["after_ratio_test"],
    "rej": 100 * mt["rejection_rate"],
    "nocc": hg["occupied_bins"],
    "nover": hg["bins_over_threshold"],
    "ntop": hg["top_bin_votes"],
    "ndet": len(dets),
    "nrawdet": len(dets) + int(R.get("n_suppressed_by_nms", 0)),
    "nsupp": int(R.get("n_suppressed_by_nms", 0)),
    "nrej": int(R.get("n_rejected_by_verification", 0)),
    "runtime": timing.get("total", 0),
}

det_rows_tex = "\n".join(
    f"    {i+1} & {d['scale']:.3f} & {d['rotation_deg']:.1f} & "
    f"{d['n_inliers']} & {d['inlier_ratio']:.2f} & {d['mean_error']:.2f} & "
    f"{d['zncc']:.2f} & {d['sv_ratio']:.3f} \\\\"
    for i, d in enumerate(dets)
)


# --------------------------------------------------------------------------- #
#  Figure preparation (JPEG embeds for both LaTeX and ReportLab)
# --------------------------------------------------------------------------- #
FIGDIR = os.path.join(OUT, "figs")
os.makedirs(FIGDIR, exist_ok=True)


def prepare_figure(src_name, max_edge=1800):
    src = os.path.join(FIGSRC, src_name)
    if not os.path.isfile(src):
        # allow .jpg fallbacks written by the pipeline
        alt = os.path.splitext(src)[0] + ".jpg"
        src = alt if os.path.isfile(alt) else src
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"missing figure: {src}")
    long_edge = max(img.shape[:2])
    if long_edge > max_edge:
        s = max_edge / long_edge
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out_name = os.path.splitext(src_name)[0] + ".jpg"
    out_path = os.path.join(FIGDIR, out_name)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out_path


FIGURES = {
    "fig00": prepare_figure("fig00_inputs.png"),
    "fig02": prepare_figure("fig02_naive_all_nn_matches.png"),
    "fig03": prepare_figure("fig03_after_lowe_ratio.png"),
    "fig04": prepare_figure("fig04_hough_vote_space.png"),
    "fig05": prepare_figure("fig05_inlier_clusters.png"),
    "fig06": prepare_figure("fig06_nms_before_after.png"),
    "fig07": prepare_figure("fig07_final_detections.png"),
    "fig08": prepare_figure("fig08_comparison_naive_vs_final.png"),
}


# --------------------------------------------------------------------------- #
#  LaTeX source
# --------------------------------------------------------------------------- #
TEX = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs}
\usepackage{caption,float,enumitem}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}

\captionsetup{font=small,labelfont=bf}
\setlist{itemsep=2pt,topsep=4pt}
\newcommand{\code}[1]{\texttt{\small #1}}

\title{Assignment 1: Multi-Instance Object Recognition in Cluttered Scenes}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-0.8cm}

\section{Objective}

The goal of this assignment is to detect and localise multiple instances of a
known object in a single cluttered photograph. Instances may differ in scale
and orientation and may be partly occluded by other objects. The pipeline uses
OpenCV for image I/O and SIFT feature extraction. Feature matching,
parameter-space clustering, geometric estimation, verification and duplicate
suppression are implemented in the accompanying code.

\section{Data}

\begin{itemize}
  \item \textbf{Template.} An isolated photograph of a Kurkure packet
        (\code{data/template.png}, TMPLW$\times$TMPLH).
  \item \textbf{Query scene.} A photograph of my lab desk
        (\code{data/scene.png}, SCENEW$\times$SCENEH) containing several
        Kurkure packets among other clutter, at different distances and
        orientations, with partial occlusion.
\end{itemize}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figs/fig00_inputs.jpg}
\caption{Left: isolated Kurkure template. Right: cluttered lab-desk query.}
\end{figure}

\section{Recognition pipeline}

\subsection{SIFT features}

SIFT keypoints and 128-D descriptors are extracted on both images
(NKPT template keypoints, NKPS scene keypoints). Descriptors are
$L_1$-normalised and square-rooted (RootSIFT) before matching so that
Euclidean distance corresponds to the Hellinger distance between gradient
histograms.

\subsection{Feature correspondence}

For each scene descriptor the two nearest template descriptors are found by
explicit Euclidean distance. A match is kept only if it passes Lowe's ratio
test
\[
  \frac{d_1}{d_2} < \rho, \qquad \rho = RHO.
\]
Matching is performed from the scene to the template so that each instance can
accumulate its own correspondences rather than competing for a fixed budget of
template features. This reduced NRAW nearest-neighbour pairs to NMATCH
matches (rejection rate REJPCT\%).

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig02_naive_all_nn_matches.jpg}
\caption{Naive nearest-neighbour matches without a ratio test. Correspondences
spread across background clutter and do not identify object instances.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig03_after_lowe_ratio.jpg}
\caption{Matches remaining after Lowe's ratio test. Clutter is reduced, but
the output is still an unordered set of correspondences.}
\end{figure}

\subsection{4D generalised Hough transform}

A single SIFT match implies a similarity pose (centre, scale, orientation).
Each match therefore casts a vote in a four-dimensional parameter space
$(c_x, c_y, \log_2 s, \Delta\theta)$. Votes are broadened to neighbouring bins
so that clusters straddling a bin boundary are not split. Bins with at least
three votes form candidate object hypotheses. In this run the vote table
contained NOCC occupied bins, of which NOVER cleared the vote threshold; the
strongest bin held NTOP votes.

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig04_hough_vote_space.jpg}
\caption{Hough votes marginalised onto the image plane. Peaks correspond to
likely object centres.}
\end{figure}

\subsection{Robust affine estimation}

For each candidate bin, a 2D affine transform
\[
  \begin{bmatrix}x'\\y'\end{bmatrix}
  =
  \begin{bmatrix}a&b\\c&d\end{bmatrix}
  \begin{bmatrix}x\\y\end{bmatrix}
  +
  \begin{bmatrix}t_x\\t_y\end{bmatrix}
\]
is estimated with RANSAC. Minimal samples of three correspondences are solved
exactly; surviving models are refined by least squares on the inlier set
($A\mathbf{x}=\mathbf{b}$). Degenerate samples are rejected by a collinearity
check, and fitted transforms are discarded if they imply a reflection or
excessive shear.

\subsection{Iterative extraction}

One affine model describes one instance. After a verified detection is emitted,
its inlier matches are removed and the Hough table is rebuilt. The loop repeats
until no remaining bin yields a verified fit. This sequential inlier
subtraction recovers multiple instances without rediscovering the strongest one
repeatedly.

\subsection{Verification and non-maximum suppression}

A hypothesis is accepted only if it has sufficient inliers, a coherent inlier
ratio, acceptable mean reprojection error, geometric sanity, adequate inlier
spread across the template, and a positive appearance check based on zero-mean
normalised cross-correlation. Overlapping boxes that describe the same physical
packet are merged with polygon IoU non-maximum suppression.

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig05_inlier_clusters.jpg}
\caption{RANSAC inliers for each accepted instance after greedy extraction.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig06_nms_before_after.jpg}
\caption{Duplicate suppression: overlapping hypotheses before and after IoU NMS.}
\end{figure}

\section{Results}

The pipeline localised \textbf{NDET} Kurkure packets on the lab desk.
NRAWDET raw hypotheses were produced before NMS; NSUPP were removed as
duplicates; NREJ bins failed verification. Total runtime was RUNTIME\,s.

\begin{table}[H]\centering\small
\caption{Accepted detections.}
\begin{tabular}{@{}rrrrrrrr@{}}
\toprule
\# & Scale & Rotation ($^\circ$) & Inliers & Inlier ratio & Mean err.\ (px) & ZNCC & $\sigma_{\max}/\sigma_{\min}$ \\
\midrule
DETROWS
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.96\textwidth]{figs/fig07_final_detections.jpg}
\caption{Final localisations. Each quadrilateral is the template rectangle
projected through the estimated affine transform.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{figs/fig08_comparison_naive_vs_final.jpg}
\caption{Comparison required by the brief: chaotic naive feature matching
versus the final processed bounding-box detections.}
\end{figure}

\section{Discussion}

Naive matching alone is not usable in a cluttered desk scene: most
correspondences attach to background texture. Lowe's ratio test removes a large
fraction of ambiguous pairs, but still leaves an unordered match set. Clustering
in pose space groups those matches into instance hypotheses; RANSAC and the
geometric checks keep the projected boxes from shearing; verification rejects
spurious peaks; and IoU NMS leaves one box per packet.

The recovered scales (about 0.23--0.32 relative to the close-up template) and
rotations near $0^\circ$ and $90^\circ$ match the layout visible in the lab-desk
photograph. Mean reprojection errors on the accepted models are a few pixels on
a 4K scene.

\section{How to reproduce}

\begin{verbatim}
pip install -r requirements.txt
python run_pipeline.py --template data/template.png \
                       --scene data/scene.png --no-gt --out outputs
python make_report.py
\end{verbatim}

\end{document}
"""

subs = {
    "TMPLW": str(stats["tmpl_w"]),
    "TMPLH": str(stats["tmpl_h"]),
    "SCENEW": str(stats["scene_w"]),
    "SCENEH": str(stats["scene_h"]),
    "NKPT": f"{stats['nkpt']:,}",
    "NKPS": f"{stats['nkps']:,}",
    "RHO": f(stats["rho"], 2),
    "NRAW": f"{stats['nraw']:,}",
    "NMATCH": f"{stats['nmatch']:,}",
    "REJPCT": f(stats["rej"], 1),
    "NOCC": f"{stats['nocc']:,}",
    "NOVER": f"{stats['nover']:,}",
    "NTOP": f"{stats['ntop']:,}",
    "NDET": str(stats["ndet"]),
    "NRAWDET": str(stats["nrawdet"]),
    "NSUPP": str(stats["nsupp"]),
    "NREJ": str(stats["nrej"]),
    "RUNTIME": f(stats["runtime"], 1),
    "DETROWS": det_rows_tex,
}

tex = TEX
for k in sorted(subs, key=len, reverse=True):
    tex = tex.replace(k, str(subs[k]))

tex_path = os.path.join(OUT, "report.tex")
with open(tex_path, "w") as fh:
    fh.write(tex)
print(f"wrote {tex_path}")


# --------------------------------------------------------------------------- #
#  PDF via ReportLab (always available in this environment)
# --------------------------------------------------------------------------- #
def fitted_image(path, max_w, max_h=None):
    img = Image(path)
    w, h = img.imageWidth, img.imageHeight
    scale = max_w / float(w)
    if max_h is not None:
        scale = min(scale, max_h / float(h))
    img.drawWidth = w * scale
    img.drawHeight = h * scale
    return img


def build_pdf_reportlab(pdf_path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=16,
        leading=20,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        "CaptionCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceBefore=3,
        spaceAfter=10,
    )
    mono = ParagraphStyle(
        "MonoCustom",
        parent=styles["Code"],
        fontSize=8.5,
        leading=11,
        fontName="Courier",
        spaceBefore=4,
        spaceAfter=8,
        leftIndent=8,
    )

    story = []
    story.append(Paragraph(
        "Assignment 1: Multi-Instance Object Recognition in Cluttered Scenes",
        title,
    ))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("1. Objective", h1))
    story.append(Paragraph(
        "The goal of this assignment is to detect and localise multiple "
        "instances of a known object in a single cluttered photograph. "
        "Instances may differ in scale and orientation and may be partly "
        "occluded by other objects. The pipeline uses OpenCV for image I/O "
        "and SIFT feature extraction. Feature matching, parameter-space "
        "clustering, geometric estimation, verification and duplicate "
        "suppression are implemented in the accompanying code.",
        body,
    ))

    story.append(Paragraph("2. Data", h1))
    story.append(ListFlowable([
        ListItem(Paragraph(
            "<b>Template.</b> An isolated photograph of a Kurkure packet "
            f"(<font face='Courier'>data/template.png</font>, "
            f"{stats['tmpl_w']}&times;{stats['tmpl_h']}).",
            body,
        ), leftIndent=12, bulletColor=colors.black),
        ListItem(Paragraph(
            "<b>Query scene.</b> A photograph of my lab desk "
            f"(<font face='Courier'>data/scene.png</font>, "
            f"{stats['scene_w']}&times;{stats['scene_h']}) containing several "
            "Kurkure packets among other clutter, at different distances and "
            "orientations, with partial occlusion.",
            body,
        ), leftIndent=12, bulletColor=colors.black),
    ], bulletType="bullet", start="•"))
    story.append(Spacer(1, 0.2 * cm))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig00"], 16.5 * cm, 9 * cm),
        Paragraph(
            "Figure 1. Left: isolated Kurkure template. "
            "Right: cluttered lab-desk query.",
            caption,
        ),
    ]))

    story.append(Paragraph("3. Recognition pipeline", h1))

    story.append(Paragraph("3.1 SIFT features", h2))
    story.append(Paragraph(
        f"SIFT keypoints and 128-D descriptors are extracted on both images "
        f"({stats['nkpt']:,} template keypoints, {stats['nkps']:,} scene "
        f"keypoints). Descriptors are L1-normalised and square-rooted "
        f"(RootSIFT) before matching so that Euclidean distance corresponds "
        f"to the Hellinger distance between gradient histograms.",
        body,
    ))

    story.append(Paragraph("3.2 Feature correspondence", h2))
    story.append(Paragraph(
        f"For each scene descriptor the two nearest template descriptors are "
        f"found by explicit Euclidean distance. A match is kept only if it "
        f"passes Lowe's ratio test d1/d2 &lt; &rho; with &rho; = "
        f"{f(stats['rho'], 2)}. Matching is performed from the scene to the "
        f"template so that each instance can accumulate its own "
        f"correspondences rather than competing for a fixed budget of "
        f"template features. This reduced {stats['nraw']:,} nearest-neighbour "
        f"pairs to {stats['nmatch']:,} matches "
        f"(rejection rate {f(stats['rej'], 1)}%).",
        body,
    ))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig02"], 16.5 * cm, 8.5 * cm),
        Paragraph(
            "Figure 2. Naive nearest-neighbour matches without a ratio test. "
            "Correspondences spread across background clutter and do not "
            "identify object instances.",
            caption,
        ),
    ]))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig03"], 16.5 * cm, 8.5 * cm),
        Paragraph(
            "Figure 3. Matches remaining after Lowe's ratio test. Clutter is "
            "reduced, but the output is still an unordered set of "
            "correspondences.",
            caption,
        ),
    ]))

    story.append(Paragraph("3.3 4D generalised Hough transform", h2))
    story.append(Paragraph(
        f"A single SIFT match implies a similarity pose (centre, scale, "
        f"orientation). Each match therefore casts a vote in a "
        f"four-dimensional parameter space "
        f"(c<sub>x</sub>, c<sub>y</sub>, log<sub>2</sub> s, &Delta;&theta;). "
        f"Votes are broadened to neighbouring bins so that clusters "
        f"straddling a bin boundary are not split. Bins with at least three "
        f"votes form candidate object hypotheses. In this run the vote table "
        f"contained {stats['nocc']:,} occupied bins, of which "
        f"{stats['nover']:,} cleared the vote threshold; the strongest bin "
        f"held {stats['ntop']:,} votes.",
        body,
    ))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig04"], 16.5 * cm, 8 * cm),
        Paragraph(
            "Figure 4. Hough votes marginalised onto the image plane. Peaks "
            "correspond to likely object centres.",
            caption,
        ),
    ]))

    story.append(Paragraph("3.4 Robust affine estimation", h2))
    story.append(Paragraph(
        "For each candidate bin a 2D affine transform is estimated with "
        "RANSAC. Minimal samples of three correspondences are solved exactly; "
        "surviving models are refined by least squares on the inlier set "
        "(A x = b). Degenerate samples are rejected by a collinearity check, "
        "and fitted transforms are discarded if they imply a reflection or "
        "excessive shear.",
        body,
    ))

    story.append(Paragraph("3.5 Iterative extraction", h2))
    story.append(Paragraph(
        "One affine model describes one instance. After a verified detection "
        "is emitted, its inlier matches are removed and the Hough table is "
        "rebuilt. The loop repeats until no remaining bin yields a verified "
        "fit. This sequential inlier subtraction recovers multiple instances "
        "without rediscovering the strongest one repeatedly.",
        body,
    ))

    story.append(Paragraph("3.6 Verification and non-maximum suppression", h2))
    story.append(Paragraph(
        "A hypothesis is accepted only if it has sufficient inliers, a "
        "coherent inlier ratio, acceptable mean reprojection error, geometric "
        "sanity, adequate inlier spread across the template, and a positive "
        "appearance check based on zero-mean normalised cross-correlation. "
        "Overlapping boxes that describe the same physical packet are merged "
        "with polygon IoU non-maximum suppression.",
        body,
    ))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig05"], 16.5 * cm, 8 * cm),
        Paragraph(
            "Figure 5. RANSAC inliers for each accepted instance after greedy "
            "extraction.",
            caption,
        ),
    ]))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig06"], 16.5 * cm, 7.5 * cm),
        Paragraph(
            "Figure 6. Duplicate suppression: overlapping hypotheses before "
            "and after IoU NMS.",
            caption,
        ),
    ]))

    story.append(Paragraph("4. Results", h1))
    story.append(Paragraph(
        f"The pipeline localised <b>{stats['ndet']}</b> Kurkure packets on "
        f"the lab desk. {stats['nrawdet']} raw hypotheses were produced "
        f"before NMS; {stats['nsupp']} were removed as duplicates; "
        f"{stats['nrej']} bins failed verification. Total runtime was "
        f"{f(stats['runtime'], 1)} s.",
        body,
    ))

    header = ["#", "Scale", "Rot (°)", "Inliers", "Ratio",
              "Err (px)", "ZNCC", "σmax/σmin"]
    rows = [header]
    for i, d in enumerate(dets):
        rows.append([
            str(i + 1),
            f(d["scale"], 3),
            f(d["rotation_deg"], 1),
            str(d["n_inliers"]),
            f(d["inlier_ratio"], 2),
            f(d["mean_error"], 2),
            f(d["zncc"], 2),
            f(d["sv_ratio"], 3),
        ])
    table = Table(rows, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)
    story.append(Paragraph("Table 1. Accepted detections.", caption))

    story.append(KeepTogether([
        fitted_image(FIGURES["fig07"], 16.5 * cm, 9 * cm),
        Paragraph(
            "Figure 7. Final localisations. Each quadrilateral is the "
            "template rectangle projected through the estimated affine "
            "transform.",
            caption,
        ),
    ]))
    story.append(KeepTogether([
        fitted_image(FIGURES["fig08"], 15.5 * cm, 14 * cm),
        Paragraph(
            "Figure 8. Comparison required by the brief: chaotic naive "
            "feature matching versus the final processed bounding-box "
            "detections.",
            caption,
        ),
    ]))

    story.append(Paragraph("5. Discussion", h1))
    story.append(Paragraph(
        "Naive matching alone is not usable in a cluttered desk scene: most "
        "correspondences attach to background texture. Lowe's ratio test "
        "removes a large fraction of ambiguous pairs, but still leaves an "
        "unordered match set. Clustering in pose space groups those matches "
        "into instance hypotheses; RANSAC and the geometric checks keep the "
        "projected boxes from shearing; verification rejects spurious peaks; "
        "and IoU NMS leaves one box per packet.",
        body,
    ))
    story.append(Paragraph(
        "The recovered scales (about 0.23–0.32 relative to the close-up "
        "template) and rotations near 0° and 90° match the layout visible in "
        "the lab-desk photograph. Mean reprojection errors on the accepted "
        "models are a few pixels on a 4K scene.",
        body,
    ))

    story.append(Paragraph("6. How to reproduce", h1))
    story.append(Paragraph(
        "pip install -r requirements.txt<br/>"
        "python run_pipeline.py --template data/template.png "
        "--scene data/scene.png --no-gt --out outputs<br/>"
        "python make_report.py",
        mono,
    ))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Assignment 1: Multi-Instance Object Recognition",
    )
    doc.build(story)
    print(f"wrote {pdf_path}")


pdf_path = os.path.join(OUT, "report.pdf")

# Prefer pdflatex when present; otherwise ReportLab.
compiled = False
for name in ("pdflatex", "xelatex"):
    path = shutil.which(name)
    mac = f"/Library/TeX/texbin/{name}"
    engine = path or (mac if os.path.isfile(mac) else None)
    if not engine:
        continue
    try:
        for _ in range(2):
            subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error",
                 "report.tex"],
                cwd=OUT,
                check=True,
                capture_output=True,
            )
        print(f"compiled with {name}: {pdf_path}")
        compiled = True
        break
    except subprocess.CalledProcessError as exc:
        print(f"{name} failed ({exc}); falling back to ReportLab")

if not compiled:
    build_pdf_reportlab(pdf_path)

for dest in (
    os.path.join(HERE, "MIOR_report.pdf"),
    os.path.join(os.path.dirname(HERE), "MIOR_report.pdf"),
):
    shutil.copy2(pdf_path, dest)
    print(f"copied {dest}")
