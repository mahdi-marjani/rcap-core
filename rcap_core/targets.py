import re

TARGET_MAPPING = {
    "bicycle": 1,
    "bus": 5,
    "tractor": 7,
    "boat": 8,
    "car": 2,
    "hydrant": 10,
    "motorcycle": 3,
    "traffic": 9,
    "crosswalk": 1001,
    "stair": 1002,
    "taxi": 1003,
}

def get_target_num(target_text):
    for key, value in TARGET_MAPPING.items():
        if re.search(key, target_text) is not None:
            return value
    return 1000