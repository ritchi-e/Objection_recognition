# Assignment 1 — Multi-Instance Object Recognition in Cluttered Scenes

Detect and localise multiple instances of a known object in one cluttered
photograph (scale / rotation / partial occlusion).

**Object:** Kurkure packet  
**Scene:** lab desk (`data/template.png`, `data/scene.png`)  
**Result:** 3 instances localised (~50 s on 4K photos)

## What this project contains

| Piece | Role |
|---|---|
| `data/template.png` | Isolated Kurkure template photo |
| `data/scene.png` | Cluttered lab-desk query (≥3 packets) |
| `mior/` | Recognition pipeline modules |
| `run_pipeline.py` | End-to-end run → figures + `outputs/results.json` |
| `make_report.py` | Builds `report/report.pdf` from those outputs |
| `verify_compliance.py` | Checks banned OpenCV / sklearn APIs are not used |
| `requirements.txt` | Python dependencies |
| `report/report.pdf` / `MIOR_report.pdf` | Assignment report |

### Pipeline stages (`mior/`)

1. **`features.py`** — SIFT keypoints + 128-D descriptors (OpenCV), RootSIFT  
2. **`matching.py`** — Euclidean distances, Lowe ratio test, scene → template  
3. **`hough.py`** — 4D generalised Hough transform (centre, scale, orientation)  
4. **`geometry.py`** / **`ransac.py`** — affine fit (`Ax = b`), RANSAC, collinearity / shear checks  
5. **`pipeline.py`** — greedy extraction with sequential inlier subtraction + verification  
6. **`nms.py`** — polygon IoU non-maximum suppression  
7. **`viz.py`** — figures (boxes / matches; final image is clean boxes only)

OpenCV is used for image I/O and `cv2.SIFT_create`. Matching, clustering,
geometry, verification and NMS are written in this repo 


## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# detect instances and write figures
python run_pipeline.py --template data/template.png --scene data/scene.png \
                       --no-gt --out outputs

# rebuild the PDF report
python make_report.py

# optional: confirm library restrictions
python verify_compliance.py
```

Useful knobs if a weak instance is missed: `--ratio 0.85`, `--min-inliers 5`,
`--min-zncc 0.05`.

## Outputs

```
outputs/
  fig00_inputs.png
  fig02_naive_all_nn_matches.png      # naive matching (chaotic)
  fig03_after_lowe_ratio.png
  fig04_hough_vote_space.png
  fig05_inlier_clusters.png
  fig06_nms_before_after.png
  fig07_final_detections.png          # clean boxes
  fig08_comparison_naive_vs_final.png # report comparison
  final_detections_clean.png
  results.json
report/report.pdf
MIOR_report.pdf
```

## Report

The report covers the data (Kurkure + lab desk), the mathematical pipeline,
and the required naive-matching vs final-bounding-box comparison, using the
figures above.
