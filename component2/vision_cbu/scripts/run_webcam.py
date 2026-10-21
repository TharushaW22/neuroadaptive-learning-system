"""Live webcam feature extractor (OpenCV window). Press Q or Esc to stop.

    python scripts/run_webcam.py                       # live webcam, features only
    python scripts/run_webcam.py --seconds 30 --no-display
    python scripts/run_webcam.py --mock --seconds 5 --no-display   # no camera needed
    python scripts/run_webcam.py --save --consent      # save FEATURES (never video) to outputs/sessions

Cognitive-state predictions are shown only if a validated real model exists; otherwise the
overlay reads 'NO VALIDATED MODEL AVAILABLE'.
"""
import argparse
import sys

import cv2
import pandas as pd

from _bootstrap import EXIT_ERROR, EXIT_OK
from src.models.inference import load_live_predictor
from src.models.interpretation import interpret_window, interpretation_config
from src.processing.quality_control import quality_warnings
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.utils.privacy import ConsentRequiredError, pseudonym_for, save_session
from src.vision.face_landmarks import draw_landmarks
from src.vision.feature_extractor import FrameFeatureExtractor
from src.webcam.camera import Camera, CameraError, MockCamera


def _fmt(v) -> str:
    return "UNKNOWN" if v is None else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="mock camera + mock landmarks (no hardware)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--participant", default="anonymous")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--consent", action="store_true", help="I consent to saving extracted features")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config, {"privacy": {"enable_session_logging": a.save}})

    detector = None
    if a.mock:
        from src.vision.mock_landmarks import MockLandmarkDetector
        detector = MockLandmarkDetector([{"eye_open": 1.0}] * 8 + [{"eye_open": 0.05}] * 3)
    pid = pseudonym_for(cfg, a.participant)
    extractor = FrameFeatureExtractor(cfg, detector=detector, source="webcam", participant_id=pid, video_id="live")
    predictor, model_msg = load_live_predictor(cfg)
    print(f"Face backend: {extractor.backend} | Model: {model_msg}")
    show_scores = bool(interpretation_config(cfg)["enabled"])
    win = predictor.window if predictor is not None else int(cfg["sequences"]["window_size"])
    if show_scores:
        print("Heuristic attention/fatigue/readiness scores (A/F/R, 0-1) are EXPERIMENTAL, not validated.")
    cam = MockCamera(sample_every_n=cfg["camera"]["sample_every_n_frames"]) if a.mock else Camera(cfg)
    rows = []
    try:
        with cam:
            for i, (ts, frame) in enumerate(cam.frames(max_seconds=a.seconds)):
                row, res = extractor.extract(frame, ts, i)
                pred = None if predictor is None else predictor.predict_frames(pd.DataFrame(rows[-200:] + [row]))
                state = model_msg if pred is None else pred["state"]
                if show_scores:      # EXPERIMENTAL heuristic scores on the same window
                    row.update(interpret_window(pred, pd.DataFrame(rows[-200:] + [row]).tail(win), cfg))
                rows.append(row)
                scores = (f"heuristic A {_fmt(row.get('attention_score'))} F {_fmt(row.get('fatigue_score'))} "
                          f"R {_fmt(row.get('readiness_score'))} [{row.get('quality_state', '-')}]")
                if not a.no_display:
                    vis = draw_landmarks(frame, res) if res is not None else frame.copy()
                    lines = [f"EAR {row['ear_mean']:.2f}  blinks/min {row['blink_rate_per_min']:.1f}",
                             f"yaw {row['head_yaw']:.0f} pitch {row['head_pitch']:.0f} roll {row['head_roll']:.0f}",
                             f"gaze proxy h {row['gaze_h']:.2f} v {row['gaze_v']:.2f}", f"State: {state}",
                             *([scores + " EXPERIMENTAL"] if show_scores else []),
                             *quality_warnings(row)]
                    for k, t in enumerate(lines):
                        cv2.putText(vis, t, (10, 50 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    cv2.imshow("Vision CBU - features (Q to quit)", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                elif i % 10 == 0:
                    print(f"t={ts:6.2f}s face={row['face_detected']} quality_ok={row['quality_ok']} "
                          f"EAR={row['ear_mean']:.3f} yaw={row['head_yaw']:.1f} state={state}"
                          + (f" {scores}" if show_scores else ""))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        pass
    finally:
        extractor.close()
        if not a.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {len(rows)} sampled frames. Camera released.")
    if a.save:
        try:
            p = save_session(pd.DataFrame(rows), cfg, consent=a.consent, session_label=pid)
            print(f"Saved extracted features (no video) to {p}")
        except ConsentRequiredError as exc:
            print(f"Not saved: {exc} Re-run with --consent if you agree.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
