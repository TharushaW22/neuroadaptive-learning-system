"""Check Python / library versions, face-landmark backend and (optionally) the webcam.

    python scripts/check_environment.py            # libraries + face backend
    python scripts/check_environment.py --camera   # also try to open the webcam
"""
import argparse
import importlib
import platform
import sys

import _bootstrap  # noqa: F401

REQUIRED = [("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("torch", "torch"),
            ("cv2", "opencv"), ("mediapipe", "mediapipe"), ("streamlit", "streamlit"),
            ("matplotlib", "matplotlib"), ("yaml", "pyyaml"), ("joblib", "joblib"), ("pytest", "pytest")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="try to open the configured webcam")
    args = ap.parse_args()
    ok = True
    from src import __version__
    print(f"Vision CBU project version {__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info[:2] != (3, 11):
        print("  NOTE: the pinned requirements target Python 3.11. Other versions may need different pins.")
    for mod, name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  [OK]      {name:13s} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            ok = ok and mod in ("mediapipe",)
            print(f"  [MISSING] {name:13s} {type(exc).__name__}: {exc}")
    try:
        import torch
        print(f"  torch CUDA available: {torch.cuda.is_available()} (CPU-only is fully supported)")
    except ImportError:
        pass

    from src.utils.config import load_config
    from src.vision.face_landmarks import FaceLandmarkDetector
    cfg = load_config()
    det = FaceLandmarkDetector(cfg)
    print(f"\nFace-landmark backend: {det.backend}")
    if det.backend == "none":
        print("  No face backend. Live features will be unavailable (tests still pass - they use a mock).")
    elif det.backend == "opencv_haar":
        print("  Face boxes only: EAR / gaze / head-pose features will be missing.\n"
              "  Fix: pip install mediapipe==0.10.14  (Python 3.11), or run scripts/download_face_model.py")
    det.close()

    if args.camera:
        from src.webcam.camera import probe_camera
        cam_ok, msg = probe_camera(cfg)
        print(f"\nCamera: {'OK' if cam_ok else 'FAILED'} - {msg}")

    from src.utils.status import get_status
    st = get_status(cfg)
    print(f"\nMode: {st['mode']} | videos: {st['n_videos']} | annotation files: {st['n_annotation_files']} | "
          f"audit: {st['audit_status']} | validated model: {'yes' if st['validated_model'] else 'NO VALIDATED MODEL AVAILABLE'}")
    print("\nEnvironment check", "PASSED" if ok else "FAILED (install missing packages: pip install -r requirements.txt)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
