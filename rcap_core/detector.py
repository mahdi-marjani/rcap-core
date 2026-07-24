from .detection_3x3 import detect_cells_3x3
from .detection_4x4 import detect_cells_4x4
from .targets import get_target_num

def detect_cells(image, grid, target_text):
    target_num = get_target_num(target_text)
    
    if grid == "3x3":
        return detect_cells_3x3(image, target_num)
    if grid == "4x4":
        return detect_cells_4x4(image, target_num)