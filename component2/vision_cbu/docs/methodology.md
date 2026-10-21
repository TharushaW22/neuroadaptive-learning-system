# Methodology

## 1. Research question
**To what extent can visual behaviour features extracted from a single ordinary RGB webcam,
modelled over time, estimate learner engagement (as annotated in DAiSEE), and does temporal
modelling (GRU/LSTM) outperform a non-temporal baseline built on window statistics?**

Sub-questions
1. Which feature groups (eye/blink, gaze proxy, head pose, facial geometry) contribute most? (ablation)
2. How sensitive is performance to window length, stride and sampling rate?
3. Is inference fast enough on a CPU for live use? (latency)

The target in the first experiment is the **original DAiSEE engagement level (0-3)**. It is not
re-interpreted as attention or distraction (see `labeling_protocol.md`).

## 2. Data
* **DAiSEE** (to be added): short clips of students watching online lecture material, each with
  crowd-annotated intensity levels (0-3) for engagement, boredom, confusion and frustration, and
  official train/validation/test splits. Check details (counts, recording conditions, licence)
  against the official page and paper before citing them.
* **Synthetic data** (now): generated feature tables used *only* to verify the software.

## 3. Visual features (per sampled frame, schema v1.0.0)
| Group | Features | Method |
|---|---|---|
| eye | `ear_left`, `ear_right`, `ear_mean`, `blink_event`, `blink_rate_per_min` | Eye Aspect Ratio (Soukupova & Cech, 2016) on 6 eye landmarks; blink = EAR below threshold for 2-15 consecutive samples then re-opening; rate over a trailing 60 s window (NaN for the first 10 s). |
| gaze | `gaze_h`, `gaze_v` | Iris-centre position within the eye opening, normalised to about [-1, 1]. **Uncalibrated proxy**, not gaze estimation. |
| head_pose | `head_yaw`, `head_pitch`, `head_roll` | `cv2.solvePnP` matching all 468 Face Mesh landmarks to MediaPipe's canonical 3D face model, with approximate camera intrinsics. Signs: yaw + = face turns to image right, pitch + = looking down, roll + = clockwise. Verified by recovering known rotations of the model (< 0.5 deg) and by a mirror-image check on a real photo (yaw sign flips, pitch within ~1 deg). |
| facial | `mouth_aspect_ratio`, `brow_raise`, `face_scale`, `face_center_x`, `face_center_y` | Geometric ratios. Not FACS action units. |
| quality | `face_detected`, `num_faces`, `multiple_faces`, `brightness`, `low_light`, `over_exposed`, `face_too_small`, `landmarks_complete`, `quality_ok` | Rule-based checks per frame. |

Landmarks come from MediaPipe Face Mesh / Face Landmarker (478 points with iris). Frames are
sampled at `extraction.target_fps` (default 5 fps) for dataset videos; the webcam processes one
in every `camera.sample_every_n_frames` frames.

## 4. Temporal processing
1. Frames failing `quality_ok` are treated as missing.
2. Gaps of up to `max_interpolate_gap` samples are forward-filled **within the same video only**.
3. Windows of `window_size` samples (default 20 = 4 s at 5 fps) with `stride` (default 10) are
   built per video; windows never span two videos. Short videos give one zero-padded window
   with a mask. Windows with fewer than `min_valid_ratio` quality-OK samples are dropped.
4. **Label policy `broadcast_video_label`**: every window inherits its clip's video-level label.
   These are therefore *weak* window labels, not independently verified frame/window ground truth.
   Video-level metrics (averaging window probabilities per clip) are reported alongside.
5. Remaining missing values are filled with **training-set medians** and features are
   standardised with statistics **fitted on the training split only**.

## 5. Models
* **Baseline**: per-window mean, std, min, max of each feature (4 x 15 = 60 inputs) ->
  median imputer -> standard scaler -> Logistic Regression (default) or Random Forest; class
  weights balanced.
* **Temporal model**: GRU (default) or LSTM over the normalised sequence; representation =
  [masked mean of hidden states ; hidden state at last valid step] -> dropout -> linear.
  Cross-entropy with balanced class weights, Adam, early stopping on validation loss,
  gradient clipping. Configurable sequence length, hidden size, layers, batch size, learning rate,
  epochs, stride and sampling rate.
* Seeds are fixed (`project.seed`) for Python, NumPy and PyTorch.
* Checkpoints store: label mapping, feature schema version + fingerprint, feature list, window
  settings, normaliser parameters, model hyper-parameters, training-data identity (video-ID hash,
  per-split participant/video counts), library versions. Loading validates these.

## 6. Split protocol (leakage prevention)
* Official DAiSEE train/validation/test split files are preserved.
* If no validation split exists, it is carved from the training split **by participant**
  (GroupShuffleSplit).
* A video (and all its windows) belongs to exactly one split - enforced (error otherwise).
* Participant overlap between official splits is detected by the audit; it is reported as a
  limitation (and raises an error if official splits are disabled and participant-level splitting is on).
* Normalisation and imputation statistics are fitted on training data only.

## 7. Metrics
Accuracy; macro precision, recall, F1; weighted F1; per-class precision/recall/F1/support;
confusion matrix; majority-class accuracy (reference point for the imbalanced labels);
video-level macro F1; training/validation curves; per-window inference latency on CPU;
baseline vs. temporal comparison; feature-group ablation (baseline retrained without each group).
**Macro F1 is the primary metric** because DAiSEE engagement levels are strongly imbalanced.

## 8. Limitations
See `limitations_and_ethics.md`. Key points: weak (video-level) labels; uncalibrated gaze;
canonical (not person-specific) head model and approximate camera intrinsics; crowd-sourced subjective labels; dataset demographics and recording
conditions may not match the deployment population; a webcam can only observe visible behaviour.
