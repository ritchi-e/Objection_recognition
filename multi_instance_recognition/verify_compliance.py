"""
verify_compliance.py
====================
Asserts that the assignment's library restrictions are actually respected.

The assignment permits OpenCV/NumPy/Matplotlib for image I/O, matrix operations
and SIFT feature extraction, but forbids high-level automated matching,
clustering and geometric APIs.  Grepping for the forbidden names is not enough,
because every module documents in its docstring which APIs it deliberately does
NOT use, and those comments would trip a naive grep.

This script therefore parses each file into an abstract syntax tree and looks at
real attribute accesses and imports only, so docstrings and comments cannot
produce either a false alarm or a false clearance.

    python verify_compliance.py        # exits non-zero if anything is violated
"""

import ast
import glob
import os
import sys

BANNED_CV2 = {
    "BFMatcher", "FlannBasedMatcher", "DescriptorMatcher_create",
    "estimateAffine2D", "estimateAffinePartial2D", "findHomography",
    "getAffineTransform", "getPerspectiveTransform",
    "warpAffine", "warpPerspective", "matchTemplate", "kmeans",
    "drawMatches", "drawMatchesKnn", "drawKeypoints",
    "solvePnP", "solvePnPRansac", "connectedComponents",
}
BANNED_MODULES = ("sklearn", "scipy")

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    files = sorted(glob.glob(os.path.join(HERE, "**", "*.py"), recursive=True))
    violations = []
    cv2_used = {}

    for path in files:
        rel = os.path.relpath(path, HERE)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        except SyntaxError as e:
            violations.append(f"{rel}: cannot parse ({e})")
            continue

        for node in ast.walk(tree):
            # attribute accesses of the form cv2.<something>
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "cv2"):
                cv2_used.setdefault(node.attr, set()).add(rel)
                if node.attr in BANNED_CV2:
                    violations.append(f"{rel}:{node.lineno}: cv2.{node.attr}")

            # imports of forbidden third-party packages
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in BANNED_MODULES:
                        violations.append(f"{rel}:{node.lineno}: import {a.name}")
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in BANNED_MODULES:
                    violations.append(f"{rel}:{node.lineno}: from {node.module} import ...")

    print(f"Scanned {len(files)} Python files.\n")
    print("cv2 attributes actually referenced in code:")
    for name in sorted(cv2_used):
        print(f"    cv2.{name:<24} {', '.join(sorted(cv2_used[name]))}")

    print()
    if violations:
        print("COMPLIANCE FAILED - banned API used:")
        for v in violations:
            print("   ", v)
        return 1

    print("COMPLIANCE OK")
    print("  no BFMatcher / FlannBasedMatcher   -> matching is from scratch")
    print("  no sklearn.cluster / cv2.kmeans    -> clustering is from scratch")
    print("  no estimateAffine2D / findHomography / getAffineTransform")
    print("                                     -> geometry is from scratch")
    print("  no warpAffine / warpPerspective / matchTemplate")
    print("                                     -> warping and ZNCC are from scratch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
