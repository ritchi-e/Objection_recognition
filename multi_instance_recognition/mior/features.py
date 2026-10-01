"""
mior.features
=============
Stage 1 of the pipeline: SIFT keypoint detection and descriptor extraction.

This is the ONLY stage where the assignment permits a high-level library call
(`cv2.SIFT_create`).  Everything downstream - matching, clustering, geometry -
is implemented from scratch in the other modules of this package.

A SIFT keypoint carries four numbers that the rest of the pipeline depends on:

    (x, y)   sub-pixel location of the feature in its image
    sigma    the "size" of the keypoint, i.e. the scale of the DoG octave/layer
             at which it was found.  The RATIO of two matched sigmas is a
             direct estimate of the relative scale between template and scene.
    theta    the dominant gradient orientation in degrees.  The DIFFERENCE of
             two matched thetas is a direct estimate of the relative rotation.

Those two derived quantities (sigma ratio, theta difference) are what make the
4D Generalised Hough Transform in `mior.hough` possible: every single match
casts a complete pose hypothesis on its own, which is exactly the property we
need when occlusion leaves us with only a handful of correct matches.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureSet:
    """SIFT output for one image, stored as flat NumPy arrays.

    Attributes
    ----------
    xy    : (N, 2) float64 - keypoint locations
    sigma : (N,)   float64 - keypoint scales
    theta : (N,)   float64 - keypoint orientations, degrees in [0, 360)
    desc  : (N,128) float32 - L2 descriptors
    """
    xy: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray
    desc: np.ndarray

    def __len__(self):
        return len(self.xy)


def to_gray(img):
    """BGR (or already-gray) uint8 image -> single-channel uint8."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def extract_sift(img, n_features=0, contrast_threshold=0.03,
                 edge_threshold=10.0, n_octave_layers=3, sigma=1.6):
    """Detect SIFT keypoints and compute 128-D descriptors.

    Parameters
    ----------
    n_features : int
        0 keeps every keypoint.  In a cluttered scene we deliberately keep all
        of them - throwing keypoints away by a global response ranking is
        exactly what loses a small, half-occluded instance.
    contrast_threshold : float
        Lowered slightly from the OpenCV default (0.04) so that low-contrast
        features on a shadowed or partially occluded instance survive.

    Returns
    -------
    FeatureSet
    """
    gray = to_gray(img)
    sift = cv2.SIFT_create(nfeatures=n_features,
                           nOctaveLayers=n_octave_layers,
                           contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold,
                           sigma=sigma)
    kps, desc = sift.detectAndCompute(gray, None)

    if desc is None or len(kps) == 0:
        empty = np.zeros((0, 2))
        return FeatureSet(empty, np.zeros(0), np.zeros(0),
                          np.zeros((0, 128), np.float32))

    xy = np.array([k.pt for k in kps], dtype=np.float64)
    sig = np.array([k.size for k in kps], dtype=np.float64)
    th = np.array([k.angle for k in kps], dtype=np.float64) % 360.0
    return FeatureSet(xy, sig, th, np.asarray(desc, dtype=np.float32))


def rootsift(desc):
    """Convert raw SIFT descriptors to RootSIFT.

    RootSIFT (Arandjelovic & Zisserman, CVPR 2012) is a one-line change with a
    consistently better matching behaviour: L1-normalise the descriptor, then
    take the element-wise square root.  The Euclidean distance between two
    RootSIFT vectors is then the Hellinger distance between the original
    histograms, which is far less dominated by a few large gradient bins.

    We still compute plain Euclidean distances in `mior.matching`; only the
    vectors being compared change.  This is a descriptor normalisation, not a
    matching API, so it stays inside the rules of the assignment.
    """
    if len(desc) == 0:
        return desc
    d = desc.astype(np.float32)
    d /= (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)     # L1 normalise
    return np.sqrt(d)                                       # Hellinger map
