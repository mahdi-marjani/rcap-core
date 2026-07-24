# rcap-core

Core AI engine for solving reCAPTCHA v2 image challenges.

## Install

```bash
pip install rcap-core
````

## Quick example

```python
from rcap_core.detector import detect_cells
import numpy as np

# image as numpy array (H, W, 3)
image = np.zeros((300, 300, 3), dtype=np.uint8)

cells = detect_cells(
    image=image,
    grid="3x3",        # "3x3" or "4x4"
    target_text="cars"
)

print(cells)  # e.g. [1, 3, 7]
```
