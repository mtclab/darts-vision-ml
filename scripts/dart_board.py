"""
Dart board geometry constants and keypoint-to-score conversion.

Standard dart board layout used across all DartsVision scripts.
Mirrors the logic in the Android app's BoardLayout.kt for consistency.

Segment order (clockwise from 12 o'clock):
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5

Ring radii (mm from center):
    Bullseye (Double Bull):     0  -  6.35
    Bull (Single Bull):         6.35 - 15.9
    Inner Single:              15.9 - 99
    Triple:                    99  - 107
    Outer Single:             107  - 162
    Double:                   162  - 170
    Outside:                  > 170
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SEGMENT_ANGLE = 18.0  # degrees per segment

BULLSEYE_OUTER = 6.35
BULL_OUTER = 15.9
INNER_SINGLE_OUTER = 99.0
TRIPLE_OUTER = 107.0
OUTER_SINGLE_OUTER = 162.0
DOUBLE_OUTER = 170.0

DEEPDARTS_CAL_INDICES = {0: "D20", 1: "D6", 2: "D3", 3: "D11"}
DEEPDARTS_CAL_MM = {
    "D20": (0.0, -170.0),
    "D6": (170.0, 0.0),
    "D3": (0.0, 170.0),
    "D11": (-170.0, 0.0),
}


@dataclass
class DartScore:
    segment: int
    ring: str  # "bullseye", "bull", "single", "triple", "double", "single_outer", "outside"
    angle_deg: float
    radius_mm: float
    label: str  # e.g., "T20", "D16", "5", "Bull", "Bullseye", "Miss"
    points: int

    def to_string(self) -> str:
        return self.label


def segment_from_angle(angle_deg: float) -> int:
    normalized = angle_deg % 360.0
    if normalized < 0:
        normalized += 360.0
    offset = normalized + SEGMENT_ANGLE / 2.0 - 1e-9
    index = int(offset % 360.0 / SEGMENT_ANGLE)
    return SEGMENT_ORDER[min(index, 19)]


def ring_from_radius(radius_mm: float) -> str:
    if radius_mm <= BULLSEYE_OUTER:
        return "bullseye"
    elif radius_mm <= BULL_OUTER:
        return "bull"
    elif radius_mm <= INNER_SINGLE_OUTER:
        return "single"
    elif radius_mm <= TRIPLE_OUTER:
        return "triple"
    elif radius_mm <= OUTER_SINGLE_OUTER:
        return "single_outer"
    elif radius_mm <= DOUBLE_OUTER:
        return "double"
    else:
        return "outside"


def score_from_polar(angle_deg: float, radius_mm: float) -> int:
    ring = ring_from_radius(radius_mm)
    seg = segment_from_angle(angle_deg)
    if ring == "bullseye":
        return 50
    elif ring == "bull":
        return 25
    elif ring == "single" or ring == "single_outer":
        return seg
    elif ring == "triple":
        return seg * 3
    elif ring == "double":
        return seg * 2
    else:
        return 0


def format_score(segment: int, ring: str) -> str:
    if ring == "bullseye":
        return "Bullseye"
    elif ring == "bull":
        return "Bull"
    elif ring == "triple":
        return f"T{segment}"
    elif ring == "double":
        return f"D{segment}"
    elif ring in ("single", "single_outer"):
        return f"{segment}"
    else:
        return "Miss"


def classify_dart(angle_deg: float, radius_mm: float) -> DartScore:
    ring = ring_from_radius(radius_mm)
    seg = segment_from_angle(angle_deg)
    points = score_from_polar(angle_deg, radius_mm)
    label = format_score(seg, ring)
    return DartScore(
        segment=seg if ring not in ("bullseye", "bull", "outside") else 0,
        ring=ring,
        angle_deg=angle_deg,
        radius_mm=radius_mm,
        label=label,
        points=points,
    )


def keypoints_to_scores(
    cal_keypoints: List[Optional[Tuple[float, float]]],
    dart_keypoints: List[Optional[Tuple[float, float]]],
    img_width: int,
    img_height: int,
) -> List[DartScore]:
    """
    Convert YOLO keypoint coordinates to dart scores using homography.

    Args:
        cal_keypoints: List of 4 calibration keypoints [(x, y), ...] in normalized [0,1] coords.
                       Indices: 0=D20, 1=D6, 2=D3, 3=D11. None if not detected.
        dart_keypoints: List of dart tip keypoints [(x, y), ...] in normalized coords. None if not detected.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of DartScore objects.
    """
    import numpy as np

    cal_points = [kp for kp in cal_keypoints if kp is not None]
    if len(cal_points) < 4:
        return []

    cal_names = ["D20", "D6", "D3", "D11"]
    src_pts = []
    dst_pts_mm = []
    for i, name in enumerate(cal_names):
        if cal_keypoints[i] is not None:
            px, py = cal_keypoints[i]
            src_pts.append([px * img_width, py * img_height])
            dst_pts_mm.append(list(DEEPDARTS_CAL_MM[name]))

    if len(src_pts) < 4:
        return []

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts_mm = np.array(dst_pts_mm, dtype=np.float32)

    try:
        H, _ = __import__("cv2").findHomography(src_pts, dst_pts_mm)
        if H is None:
            return []
    except Exception:
        return []

    scores = []
    for kp in dart_keypoints:
        if kp is None:
            continue
        px, py = kp
        pt_pixels = np.array([px * img_width, py * img_height, 1.0])
        pt_mm = H @ pt_pixels
        pt_mm = pt_mm[:2] / pt_mm[2]

        radius_mm = math.sqrt(pt_mm[0] ** 2 + pt_mm[1] ** 2)
        angle_deg = math.degrees(math.atan2(pt_mm[0], -pt_mm[1])) % 360.0

        scores.append(classify_dart(angle_deg, radius_mm))

    return scores


def labels_to_ground_truth(
    label_line: str,
    img_width: int = 800,
    img_height: int = 800,
) -> List[DartScore]:
    """
    Parse a YOLO pose label line and compute ground truth dart scores.

    Label format (7 keypoints = 4 cal + 3 dart tips):
        class cx cy w h kpt0_x kpt0_y kpt0_v kpt1_x kpt1_y kpt1_v ... kpt6_x kpt6_y kpt6_v

    Returns list of DartScore for each visible dart tip.
    """
    parts = label_line.strip().split()
    if len(parts) < 5 + 7 * 3:
        return []

    cal_keypoints = []
    for i in range(4):
        x = float(parts[5 + i * 3])
        y = float(parts[5 + i * 3 + 1])
        v = float(parts[5 + i * 3 + 2])
        cal_keypoints.append((x, y) if v > 0.5 else None)

    dart_keypoints = []
    for i in range(3):
        idx = 5 + (4 + i) * 3
        if idx + 2 < len(parts):
            x = float(parts[idx])
            y = float(parts[idx + 1])
            v = float(parts[idx + 2])
            dart_keypoints.append((x, y) if v > 0.5 else None)

    return keypoints_to_scores(cal_keypoints, dart_keypoints, img_width, img_height)


SCORE_DEFINITIONS = """
Standard dart board scoring reference for VLM prompts:

Segment numbers clockwise from top: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5

Score format:
- T20 = triple 20 (60 points)
- D20 = double 20 (40 points)
- 20 = single 20 (20 points) — use just the number
- Bull = 25 points (outer bull, green circle)
- Bullseye = 50 points (inner bull, red center)
- Miss = off the board (0 points)
"""