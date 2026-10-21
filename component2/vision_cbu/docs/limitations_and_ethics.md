# Limitations and ethics

## Technical limitations
* **Gaze**: iris-in-eye ratios are uncalibrated proxies. They depend on head pose, eyelid shape,
  glasses and camera angle, and do not tell where on the screen a person is looking. No gaze
  accuracy is claimed.
* **Head pose**: MediaPipe's canonical (average, not person-specific) 3D face model and approximate camera intrinsics (focal length = image
  width, no lens distortion) - angles are approximate and unvalidated in absolute terms.
* **Blinks**: fixed EAR threshold; varies by person, eye shape, glasses, resolution
  and frame rate. At 5 fps very short blinks can be missed. Calibrate per user where possible.
* **Facial features**: geometric ratios only; **no FACS action-unit recognition** is implemented or claimed.
* **Labels**: DAiSEE labels are crowd-annotated, subjective and clip-level; window labels are
  inherited (weak supervision). Engagement is not attention or distraction.
* **Generalisation**: dataset participants, environments and cameras may differ from the target
  learners; results may not transfer and may differ across demographic groups (not analysed
  unless metadata is available and ethically usable).
* **Single camera**: only visible behaviour is observed; occlusion, lighting and multiple people
  degrade quality (flagged as UNKNOWN rather than guessed).
* **Synthetic data** verifies software only and has no scientific meaning.

## Ethical use
* Outputs must not be used for grading, discipline, surveillance or any high-stakes decision.
* A visual signal is never treated as proof of a mental state.
* Participants must know what is measured, why, and how to stop.

## Privacy by design (implemented)
| Measure | Where |
|---|---|
| No raw video/frames written (no code path; tested) | `tests/test_privacy_and_exports.py` |
| Session logging off by default; explicit consent required to save | `privacy.enable_session_logging`, `save_session()` |
| Logging to file disabled unless requested | `setup_logging(enable_file=False)` |
| Pseudonymous IDs: salted SHA-256 (`P_xxxxxxxxxx`); salt in env var `VCBU_PSEUDONYM_SALT` or `outputs/.pseudonym_salt` (git-ignored) | `src/utils/privacy.py` |
| Dataset and outputs git-ignored | `.gitignore` |

## Data deletion
* Saved webcam sessions: Dashboard > Settings & privacy > "Delete all saved session files",
  or delete `outputs\sessions\`.
* In-memory live data: "Clear in-memory live session" (also cleared when the browser session ends).
* Derived dataset features: delete `data\processed\features\*` and `data\processed\sequences\*`.
* Dataset: delete `data\raw\daisee\videos\*` and `annotations\*` when your licence/retention period ends.
* Deleting the pseudonym salt makes existing pseudonyms unlinkable to new sessions.

## Dataset licence
DAiSEE is distributed by its authors under their own terms (see the official page). Typically
this means research use only, no redistribution, and citation of the dataset paper. Verify the
current terms yourself, keep a copy of the agreement, and do not share videos, frames or
identifiable derivatives.
