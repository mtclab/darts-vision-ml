# Training Approaches Comparison

Detailed comparison of each detection approach that can be trained from the DeepDarts dataset.

---

## Approach 1: Darts Pose (Dart Tips + Calibration Keypoints)

**Model:** `yolo11n-pose.pt` → `darts_pose_float32.tflite` (~6 MB)

### What it does
Detects the dartboard as a single object with 7 keypoints:
- 4 calibration points (double-20/6/3/11 corners) → compute homography
- Up to 3 dart tip positions → precise scoring

### Pros
- **Pixel-precise** dart tip positions (heatmap+regression head, ±2° accuracy)
- **Board + darts in one inference** — single forward pass
- **Homography from keypoints** — replaces CV BoardDetector
- **Small model** — 2.9M params, ~6 MB TFLite
- **Real-time** — 30-80ms on mobile CPU, <10ms with GPU/NPU

### Cons
- **Needs keypoints as objects training format** — conversion from DeepDarts required
- **Dart tips vanish after removal** — doesn't help with frame-diff
- **Fixed keypoint count** — YOLO pose expects consistent keypoint topology
  - Workaround: zero-pad invisible dart tips (visibility=0)
- **Accuracy depends on dataset diversity** — 16k face-on images, only 830 multi-angle

### When to use
**Fallback detection method.** When no reference frame is available for frame-differencing. Single-frame dart detection with good accuracy.

---

## Approach 2: Darts Detect (Bounding Boxes)

**Model:** `yolo11n.pt` → `darts_detect_float32.tflite` (~4 MB)

### What it does
Detects two classes with bounding boxes:
- `dartboard`: large bbox around the board
- `dart_tip`: small bbox (~25×25 normalized) around each dart tip

### Pros
- **Simpler training** — no keypoint format, standard YOLO detect
- **Easier to extend** — add new classes (dart_flight, dart_barrel)
- **Smaller model** — no keypoint head, ~4 MB
- **Good for board detection** — robust bbox around board

### Cons
- **Less precise** — bbox center ≠ dart tip (±5-15px error)
- **No homography** — can't compute perspective correction from bbox alone
- **Still need CV BoardDetector** for geometry (or separate calibration model)

### When to use
Alternative to darts pose when keypoint detection is unavailable. Less precise but simpler.

---

## Approach 3: Board Calibration (Keypoints Only)

**Model:** `yolo11n-pose.pt` → `board_calibration_float32.tflite` (~5 MB)

### What it does
Detects the dartboard with exactly 4 calibration keypoints (no dart tips).
Computes homography from the 4 corner points.

### Pros
- **Replaces CV BoardDetector** — ML is more robust to angle, lighting, board style
- **Homography from keypoints** → precise geometry → accurate scoring
- **Simpler than full pose model** — fewer keypoints, faster convergence
- **Works with ANY dart detection method** — independent of dart location

### Cons
- **Only finds the board** — still need frame-diff or another method for darts
- **4 points minimum for homography** — if any point is missed, falls back to CV
- **Less training signal** — no dart tip annotations used

### When to use
**Primary model.** Pair with frame-differencing. Board calibration gives geometry, frame-diff finds darts.
This is the **best combo**: #3 for board + frame-diff for darts = no per-frame CV needed.

---

## Approach 4: Frame-Differencing + CV (No Training Needed)

**No model** — pure CV pipeline using OpenCV.

### What it does
1. User captures "empty board" reference frame
2. Each subsequent frame is diffed against reference
3. Changes (darts) are isolated via thresholding + morphology
4. Contour detection finds dart positions in the diff image

### Pros
- **No training data needed** — works immediately
- **No model to deploy** — 0 MB, no TFLite inference
- **Eliminates wire confusion** — board wires are static, they cancel out in diff
- **Simple contour detection works** — no HSV filtering, no dark/bright heuristics
- **Fast** — absdiff + threshold is ~5ms per frame

### Cons
- **Requires reference frame** — user must capture empty board before each turn
- **Reference goes stale** — lighting changes, camera shift invalidate it
- **No perspective correction** — still need BoardDetector for geometry
- **Can't distinguish darts from other changes** — hand in frame, shadow shift, etc.
- **Board vibration** — dart impact causes micro-shift, creating false diff regions

### When to use
**Always.** This is the current primary dart detection method. Combine with ML board calibration (#3)
for best results. Frame-diff + ML geometry = "point camera at board → instant score."

---

## Recommended Combination

```
┌──────────────────────────────────────────────┐
│              RECOMMENDED PIPELINE              │
│                                                │
│  1. Board Calibration (#3)                    │
│     → 4 keypoints → homography → geometry      │
│     → Replaces CV BoardDetector                │
│                                                │
│  2. Frame Differencing (#4)                    │
│     → Reference frame (empty board)            │
│     → Diff current frame → isolate darts       │
│     → Contour detection in diff image          │
│                                                │
│  3. Optional: Darts Pose (#1)                 │
│     → When no reference frame available        │
│     → Single-frame dart detection              │
│     → Lower accuracy than frame-diff           │
│                                                │
│  Result: ~90%+ accuracy, real-time, mobile     │
└──────────────────────────────────────────────┘
```

### Why this combo?

1. **Frame-diff is the proven technique** (Autodarts, Domazet 2023)
2. **ML board detection is more robust** than CV ellipse fitting
3. **Darts pose as fallback** when reference isn't available
4. **No model required for primary path** — frame-diff is pure CV
5. **ML models are optional** — app works without them, better with them