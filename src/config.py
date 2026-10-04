from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "ckpt_40000"

PICK_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,
    "orange block": None,
    "red sphere": None,
    "blue sphere": None,
}

PLACE_TARGETS = {
    **PICK_TARGETS,
    "blue bowl": None,
    "red bowl": None,
    "green bowl": None,
    "yellow bowl": None,
    "orange bowl": None,
    "top left corner": (-0.25, -0.25, 0),
    "top right corner": (0.25, -0.25, 0),
    "middle": (0.0, -0.5, 0),
    "bottom left corner": (-0.25, -0.75, 0),
    "bottom right corner": (0.25, -0.75, 0),
    "top center": (0.0, -0.25, 0),
    "bottom center": (0.0, -0.75, 0),
    "left center": (-0.25, -0.5, 0),
    "right center": (0.25, -0.5, 0),
}

PIXEL_SIZE = 0.00267857
BOUNDS = __import__("numpy").float32([
    [-0.3, 0.3],
    [-0.8, -0.2],
    [0.0, 0.15],
])
