from pathlib import Path
from . import yolonnx
from .model_downloader import download_model_if_missing

ROOT_DIRECTORY = Path().absolute()

MODELS_DIRECTORY = ROOT_DIRECTORY / "models"
MODELS_DIRECTORY.mkdir(exist_ok=True)

models = ["yolo26l.onnx", "crosswalk.onnx", "yolo26l-seg.onnx"]
for model in models:
    download_model_if_missing(model, MODELS_DIRECTORY)

YOLO_MODELS = {
    "yolo": yolonnx.YOLOv26(MODELS_DIRECTORY / "yolo26l.onnx"),
    "crosswalk": yolonnx.YOLOv8(MODELS_DIRECTORY / "crosswalk.onnx"),
    "yolo-seg": yolonnx.YOLOv26Seg(MODELS_DIRECTORY / "yolo26l-seg.onnx"),
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
]