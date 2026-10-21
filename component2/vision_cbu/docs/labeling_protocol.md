# Labelling protocol - engagement is not (in)attention

## 1. What DAiSEE labels mean
DAiSEE provides, per clip, crowd-annotated **intensity levels 0-3** of four affective states:
engagement, boredom, confusion, frustration. They describe how an annotator perceived the
learner over the whole clip. They are *video-level, subjective, perceived* labels.

## 2. Why engagement != attention != distraction
| Construct | Meaning | Observable by webcam? |
|---|---|---|
| Engagement (DAiSEE) | Annotator-perceived involvement with the learning activity | Partly (posture, gaze, expression), subjectively |
| Attention | Allocation of cognitive resources to the task | Only indirectly; covert attention is invisible |
| Distraction | Attention captured by something task-irrelevant | Only indirectly (e.g. sustained look-away), and ambiguous |

A learner can be low-engagement but not distracted (tired, bored, waiting), or looking away
while thinking about the task. Therefore **low engagement is never automatically mapped to
DISTRACTED**, and a single blink, glance or head turn is never interpreted as distraction.

## 3. Schemes supported by the software
| Scheme | Classes | Status |
|---|---|---|
| `daisee_original` | `engagement_0 .. engagement_3` | **Default and first experiment** |
| `daisee_binary` | low (< threshold) / high (>= threshold) | Allowed; threshold must be reported |
| `operational` | FOCUSED / REDUCED_ATTENTION / DISTRACTED | **Refused by code** unless both a mapping and an existing justification document are configured |

The original DAiSEE labels stay in every feature row (`label_engagement`, `label_boredom`,
`label_confusion`, `label_frustration`) regardless of scheme.

## 4. Requirements before using operational categories
Either
* **A. Defensible mapping**: a written document (e.g. `docs/operational_mapping_justification.md`)
  citing evidence that specific DAiSEE level combinations correspond to the operational
  categories, approved by the supervisor, with known limitations; or
* **B. New annotation study**: collect your own labels with the protocol below.

Then set in a config file:
```yaml
labels:
  scheme: operational
  operational_mapping: {0: ..., 1: ..., 2: ..., 3: ...}
  operational_mapping_document: docs/operational_mapping_justification.md
```

## 5. Template for a new annotation study (option B)
* **Definitions** given to annotators with examples and counter-examples for each category.
* **Unit**: fixed windows (e.g. 4 s) rather than whole clips, so labels match model windows.
* **Annotators**: >= 2 independent, trained, blind to model output; report Cohen's/Fleiss' kappa or
  Krippendorff's alpha; resolve disagreements by documented rule.
* **Context**: record the task (video lecture, quiz), whether looking away is task-related
  (taking notes), and environment.
* **Self-report / task performance** (optional) as complementary ground truth.
* **Ethics**: informed consent, pseudonymous IDs, retention and deletion plan, ethics approval.

## 6. Heuristic attention / fatigue / readiness scores (EXPERIMENTAL, `heuristic_v1`)

> **Status: experimental / heuristic.** Attention, fatigue, and readiness are heuristic scores in
> [0,1] derived from engagement and visual proxies. They are not validated cognitive-state
> measurements.

### 6.1 What is and is not trained
* The **only trained model** predicts the DAiSEE **engagement level 0-3** (`daisee_original`
  scheme). All reported accuracy / F1 / confusion matrices refer to engagement only.
* `attention_score`, `fatigue_score` and `readiness_score` are **fixed weighted formulas**. They
  were **not trained**, have **no ground-truth labels**, and **no accuracy is claimed or reported**
  for them. They are not part of the evaluation reports.
* Code: `src/models/interpretation.py`. Weights: `configs/default.yaml` -> `interpretation`.

### 6.2 Fields per window / sample
| Field | Type | Meaning |
|---|---|---|
| `engagement_level` | `0..3` or null | argmax class of the trained engagement model (null if no model, model UNKNOWN, or a non-0-3 label scheme) |
| `attention_score` | float in [0,1] or null | heuristic, see 6.4 |
| `fatigue_score` | float in [0,1] or null | heuristic, see 6.4 |
| `readiness_score` | float in [0,1] or null | heuristic, see 6.4 |
| `quality_state` | `OK` / `UNKNOWN` | see 6.5 |
| `interpretation_source` | `"heuristic_v1"` (or `"disabled"`) | formula version |
| `interpretation_reason` | text | why scores are null, else `ok` |

Stored values are continuous floats. The UI rounds to 2 decimals (`0.10`, `0.40`) and may show
percent (`score * 100`, e.g. `0.72` -> `72%`).

### 6.3 Visual proxies (computed over the same window the model sees; each clipped to [0,1])
The window is the model's input window (default 20 samples = 4 s at 5 fps). Only quality-OK frames
are used for eye and pose proxies.

| Proxy | Formula | Config (`interpretation.proxies`) |
|---|---|---|
| `face_vis` | fraction of window frames with `face_detected` | - |
| `ear_open` | `clip((mean EAR - ear_closed) / (ear_open - ear_closed))` | `ear_closed: 0.15`, `ear_open: 0.30` |
| `blink_high` | `clip((mean blink_rate_per_min - normal) / (high - normal))` | `blink_rate_normal: 17`, `blink_rate_high: 35` |
| `pose_stable` | `1 - clip(mean(std(yaw), std(pitch), std(roll)) / pose_motion_max_deg)` | `pose_motion_max_deg: 15` |
| `quality` | 1 if fraction of quality-OK frames >= `min_valid_frames`, else 0 | `min_valid_frames: 0.6` |

`blink_rate_per_min` is a trailing 60 s rate and is undefined for the first ~10 s of a stream, so
scores are UNKNOWN during that warm-up.

### 6.4 Scoring formulas (`heuristic_v1`)
First normalise the predicted engagement level: `e = engagement_level / 3`.

```
attention = clip01( 0.55*e + 0.20*face_vis + 0.15*pose_stable + 0.10*ear_open )
fatigue   = clip01( 0.40*(1-e) + 0.30*blink_high + 0.20*(1-ear_open) + 0.10*(1-pose_stable) )
readiness = clip01( 0.50*e + 0.30*attention + 0.20*(1-fatigue) )
```

Higher attention = more attentive-looking; higher fatigue = more fatigue-like; higher readiness =
more ready-to-learn looking. Config names of the weights:

```yaml
interpretation:
  enabled: true
  version: "heuristic_v1"
  attention_weights: {engagement: 0.55, face_vis: 0.20, pose_stable: 0.15, ear_open: 0.10}
  fatigue_weights:   {low_engagement: 0.40, blink_high: 0.30, eyes_closed: 0.20, pose_unstable: 0.10}
  readiness_weights: {engagement: 0.50, attention: 0.30, low_fatigue: 0.20}
```
Weights must be non-negative (enforced). Each default set sums to 1, so scores are in [0,1] even
before clipping. If you change any weight or proxy range, change `version` too (e.g.
`heuristic_v2`) so exported data stays traceable.

### 6.5 When scores are null (UNKNOWN)
All three scores are null, and nothing is guessed, when any of these is true:
* window face quality is poor (`quality_state = UNKNOWN`): fewer than `min_valid_frames` quality-OK
  frames, or below `dashboard.min_quality_for_prediction`;
* any proxy cannot be measured (e.g. face-box-only backend without landmarks, blink-rate warm-up);
* the engagement model returned UNKNOWN (too few samples, low quality) or no model is loaded;
* the label scheme does not produce a 0-3 engagement level (binary / operational);
* `interpretation.enabled` is false (`interpretation_source = "disabled"`).

### 6.6 Session summary and exports
* Session analytics keeps the engagement class counts (`engagement_0..3`, `UNKNOWN`) and adds, per
  score: n valid, n unknown, mean, median, min, max, mean %, and counts in the display bins
  `0.00-0.33`, `0.33-0.66`, `0.66-1.00` (bins are display-only; stored values stay continuous).
* CSV export: the score columns are floats in [0,1], empty when UNKNOWN; `interpretation_source`
  identifies the formula version.
* JSON export: scores are floats or `null`, plus a `heuristic_scores` block with the status
  (`EXPERIMENTAL / HEURISTIC - not validated`), the disclaimer, and the exact weights used.

### 6.7 Limitations
* Each score inherits the errors of the engagement prediction (it has the largest weight).
* EAR and blink thresholds vary between people, glasses, camera angle and lighting; they are not
  calibrated per person.
* Blinking more or moving the head is not evidence of fatigue or inattention on its own (e.g. note
  taking, dry eyes, talking). Interpret session-level trends, not single windows.
* Validating these scores would need a separate labelled study (section 5): for example
  self-reported sleepiness (KSS) for fatigue or task performance for readiness. Until then they
  must not be reported as measurements or evaluated as if they were.
