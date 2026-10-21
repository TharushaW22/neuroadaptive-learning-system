"""Download MediaPipe's FaceLandmarker model (only needed when your MediaPipe build has no
legacy `mediapipe.solutions` API). Run explicitly; nothing is downloaded automatically.

    python scripts/download_face_model.py

Model card / licence: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
"""
import sys
import urllib.request

import _bootstrap  # noqa: F401
from src.utils.config import load_config, resolve_path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def main() -> int:
    target = resolve_path(load_config(), "face_landmarker_model")
    if target.exists():
        print(f"Already present: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n -> {target}")
    try:
        urllib.request.urlretrieve(URL, target)
    except Exception as exc:
        print(f"Download failed: {exc}\nDownload it manually from the URL above and save it to {target}")
        return 1
    print(f"Done ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
