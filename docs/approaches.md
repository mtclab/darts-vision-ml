# Training Approaches Comparison

Comparison of detection and recognition approaches for the DartsVision hand-held darts scoring app.

**Key constraint:** Phone is held in hand (not statically mounted). Frame-differencing requires a stable reference frame and is therefore **not viable** for this use case.

---

## 3-Mode Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DartsVision Scoring Modes                   │
│                                                                │
│  Quick Mode (offline, real-time)                              │
│    YOLO board calibration (4 kp) → homography → geometry      │
│    YOLO darts pose (7 kp) → dart tip positions                │
│    → ScoringEngine → scores                                    │
│    Speed: ~30ms  |  Accuracy: ~2 deg  |  Offline: Yes         │
│                                                                │
│  Precise Mode (VLM, on-demand)                               │
│    Camera frame → VLM → "T20, S5, D16"                       │
│    → ScoringEngine → scores                                    │
│    VLM options:                                                │
│      1. Qwen3.5-0.8B on-device via LiteRT-LM (~1.2GB)       │
│      2. Gemini 2.5 Flash cloud API (free, 1,500 req/day)     │
│      3. User's own Ollama (self-hosted)                       │
│    Speed: 5-30s  |  Accuracy: Best  |  Offline: Depends      │
│                                                                │
│  Manual Mode (always available)                              │
│    User taps score on board UI                                 │
│    Speed: Instant  |  Accuracy: Perfect  |  Offline: Yes     │
│                                                                │
│  Fallback chain: Quick → Precise → Manual                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Approach 1: YOLO Board Calibration (Trained, Active)

**Model:** YOLO11n-pose → `board_calibration_float32.tflite` (~11 MB)
**Status:** Trained — mAP50-95=0.995 (50 epochs)

### What it does
Detects the dartboard with exactly 4 calibration keypoints (double-20/6/3/11 corners). Computes homography → precise geometry → segment/ring identification.

### Pros
- **Proven** — mAP50-95=0.995 on DeepDarts
- **Replaces CV BoardDetector** — more robust to angle, lighting, board style
- **Small** — ~11 MB, bundled in APK, no download needed
- **Real-time** — ~30ms on mobile CPU
- **Works with any dart detection** — independent of how darts are found

### Cons
- **Only finds the board** — still need another method for dart positions
- **94% face-on training data** — DeepDarts d1 split is mostly front-facing

### When to use
**Always.** This is the primary board detection method for all modes.

---

## Approach 2: YOLO Darts Pose (Trained, Active)

**Model:** YOLO11n-pose → `darts_pose_float32.tflite` (~11 MB)
**Status:** Trained — mAP50-95=0.912 (62 epochs)

### What it does
Detects the dartboard with 7 keypoints: 4 calibration corners + up to 3 dart tip positions. Board + darts in single inference pass.

### Pros
- **Trained and working** — mAP50-95=0.912
- **Single-frame** — no reference frame needed (works hand-held)
- **Real-time** — ~30ms on mobile CPU
- **Offline** — no network needed
- **Bundled in APK** — no download, works on any device

### Cons
- **3-dart limit** — YOLO pose has fixed keypoint topology (zero-pad invisible tips)
- **~2 deg accuracy** — less than VLM or frame-diff
- **94% face-on training data** — limited hand-held angle diversity
- **Dart tips are small** — after removal, no persistent detection

### When to use
**Quick Mode.** Default offline scoring. Instant feedback, good accuracy.

---

## Approach 3: VLM Score Reading (Under Evaluation)

**Models:** Qwen3.5-0.8B / Qwen3.5-2B (on-device) or Gemini 2.5 Flash (cloud)
**Status:** Baseline evaluation in progress

### What it does
Sends full camera frame to a Vision Language Model. VLM reads the board directly and outputs scores like "T20, S5, D16". No keypoint detection or geometry computation needed.

### Pros
- **Best accuracy** — VLM understands the scene, reads segment labels
- **No geometry pipeline** — VLM does the visual reasoning end-to-end
- **Handles angles** — VLM understands perspective without explicit homography
- **Simple pipeline** — camera frame in, scores out

### Cons
- **Slow** — 5-30 seconds per inference (on-device VLM) or network round-trip (cloud)
- **On-device VLM unproven** — Qwen3.5 accuracy for dart boards is unknown
- **Cloud requires network** — Gemini free tier works but needs internet
- **Large on-device model** — Qwen3.5-0.8B ~1.2GB via LiteRT, 2B ~4GB

### On-Device (Qwen3.5-VL)

| Model | Size | Notes |
|-------|------|-------|
| Qwen3.5-0.8B | ~1.2 GB LiteRT | Existing LiteRT conversion available. OCRBench 79.1%, RefCOCO 77.8% |
| Qwen3.5-2B | ~4 GB LiteRT | Better benchmarks (MMStar 68.1% vs 55.9%) but larger |
| Qwen3.5 LoRA | ~1.2 GB + LoRA adapter | If baseline is 50-80%, LoRA fine-tune on DeepDarts |

Model class: `Qwen3_5ForConditionalGeneration` (not `Qwen2_5_VL`). Requires `transformers>=4.57.0`.

### Cloud (Gemini 2.5 Flash)

| Detail | Value |
|--------|-------|
| Free tier | 15 RPM, 1,500 RPD |
| Latency | 2-5 seconds |
| Android SDK | Official Google AI client library |
| Image input | Base64 inline |
| Data policy (free) | Content may improve Google products; opt out on paid |

### When to use
**Precise Mode.** When user wants best accuracy and is willing to wait. Cloud VLM if on-device VLM is insufficient.

---

## Approach 4: Frame-Differencing (Not Viable for Hand-Held)

**No model** — pure CV pipeline using OpenCV.

### What it does
1. Capture "empty board" reference frame
2. Diff each subsequent frame against reference
3. Changes (darts) isolated via thresholding + morphology

### Why it doesn't work hand-held
- **Requires stable camera** — impossible when phone is held in hand
- **Reference goes stale** — any hand movement invalidates the diff
- **Board vibration** — dart impact causes micro-shift, false diff regions
- **Can't distinguish darts from other changes** — hand in frame, shadow shift

### When to use
**Only with static camera mount.** Not suitable for DartsVision's hand-held use case.

---

## Approach 5: YOLO Darts Detect (Bounding Boxes, Alternative)

**Model:** YOLO11n → `darts_detect_float32.tflite` (~4 MB)
**Status:** Trained but less precise than pose model

### What it does
Detects two classes with bounding boxes: `dartboard` (large) and `dart_tip` (small ~20x20 px).

### Pros
- Simpler training, standard YOLO detect format
- Smaller model (~4 MB)
- Easier to extend (add dart_flight, dart_barrel classes)

### Cons
- **Less precise** — bbox center ≠ dart tip (±5-15px error)
- **No homography** — can't compute geometry from bbox alone

### When to use
Alternative when keypoint detection is unavailable. Less precise than darts pose.

---

## Recommended Combination

```
Quick Mode:  Approach 1 (board calibration) + Approach 2 (darts pose)
             → YOLO inference → keypoint → homography → scoring
             → Real-time, offline, ~30ms

Precise Mode: Approach 3 (VLM)
              → Full frame → Qwen3.5 on-device OR Gemini cloud → scores
              → Best accuracy, 5-30s, requires network (cloud) or memory (on-device)

Manual Mode: User taps score on board UI
             → Always available as fallback
```

### Why this combo?

1. **Hand-held constraint** — frame-differencing is dead, YOLO single-frame is the only real-time option
2. **YOLO is proven** — mAP50-95=0.995 (calibration) and 0.912 (pose)
3. **VLM is complementary** — YOLO finds keypoints, VLM reads scores (scene understanding)
4. **Cloud VLM is free** — Gemini 2.5 Flash, 1,500 req/day, sufficient for darts
5. **Graceful fallback** — Quick → Precise → Manual