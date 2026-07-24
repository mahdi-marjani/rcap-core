from pathlib import Path
from ultralytics import YOLO
from .model_downloader import download_model_if_missing

ROOT_DIRECTORY = Path().absolute()

MODELS_DIRECTORY = ROOT_DIRECTORY / "models"
MODELS_DIRECTORY.mkdir(exist_ok=True)

download_model_if_missing("crosswalk.pt", MODELS_DIRECTORY)

YOLO_MODELS = {
    "yolo11x": YOLO(MODELS_DIRECTORY / "yolo11x.pt"),
    "crosswalk": YOLO(MODELS_DIRECTORY / "crosswalk.pt"),
    "yolo11x-seg": YOLO(MODELS_DIRECTORY / "yolo11x-seg.pt"),
    "yolov8x-oiv7": YOLO(MODELS_DIRECTORY / "yolov8x-oiv7.pt"),
}

AVAILABLE_MODELS = [
    "bicycle",
    "bus",
    "tractor",
    "boat",
    "car",
    "hydrant",
    "motorcycle",
    "traffic",
    "crosswalk",
    "stair",
    "taxi",
]