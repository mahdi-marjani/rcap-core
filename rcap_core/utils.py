import numpy as np


def find_filled_cells(corners):
    """Find all cells filled by the given corner numbers in a 4x4 grid.

    It takes corner numbers (like [1,4,13,16]) and returns all cells inside the rectangle they form.
    Simple: find min/max rows/cols, then fill the square.
    """

    # Get row and col for each corner (0-based)
    positions = [((v - 1) // 4, (v - 1) % 4) for v in corners]
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    filled = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell_num = (r * 4) + c + 1
            filled.append(cell_num)

    return sorted(set(filled))  # Remove duplicates if any


def find_object_locations(image, rows=4, cols=4, min_pixels=100):
    """Find grid spots with objects (like cars or anything) by checking pixel count.

    Splits image into rows x cols grid, checks if each spot has > min_pixels bright pixels.
    Returns list of (row, col) spots with objects. Rows/cols start from 0.
    """
    height, width, _ = image.shape
    row_size = height // rows
    col_size = width // cols

    locations = []
    for r in range(rows):
        for c in range(cols):
            spot = image[
                r * row_size : (r + 1) * row_size,
                c * col_size : (c + 1) * col_size
            ]
            bright_count = np.sum(spot > 0)
            if bright_count > min_pixels:
                locations.append((r, c))

    return locations


def locations_to_numbers(locations, cols=4):
    """Turn (row, col) locations into simple numbers (1-based) for a grid.

    Example grid numbering (for 4 cols):
    +----+----+----+----+
    |  1 |  2 |  3 |  4 |
    +----+----+----+----+
    |  5 |  6 |  7 |  8 |
    +----+----+----+----+
    |  9 | 10 | 11 | 12 |
    +----+----+----+----+
    | 13 | 14 | 15 | 16 |
    +----+----+----+----+

    So (0,0) -> 1, (1,2) -> 7, etc.
    """
    numbers = []
    for row, col in locations:
        num = (row * cols) + col + 1
        numbers.append(num)
    return numbers