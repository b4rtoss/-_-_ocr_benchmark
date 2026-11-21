"""Shared paths; OCR_BENCHMARK_DIR may point to an external experiment directory."""
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get('OCR_BENCHMARK_DIR', Path(__file__).resolve().parent)).resolve()
MANGA109_ROOT = Path(os.environ['MANGA109_ROOT']).resolve() if os.environ.get('MANGA109_ROOT') else None
IMAGE_DIR = MANGA109_ROOT / 'images' / 'Hamlet' if MANGA109_ROOT else BASE_DIR / 'Hamlet'
XML_PATH = MANGA109_ROOT / 'annotations' / 'Hamlet.xml' if MANGA109_ROOT else IMAGE_DIR / 'Hamlet.xml'
