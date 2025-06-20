import cv2
import numpy as np


def show_image(_image: np.ndarray) -> None:
    im = cv2.resize(_image, fx=0.8, fy=0.8, dsize=None)
    cv2.imshow("Image", im)
    cv2.moveWindow("Image", 1920, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_image_grid(images, rows=None, cols=None):
    """
    Converts a list of images to a consistent format and concatenates them into a grid.
    Images are placed in a single row by default.

    Args:
        images: A list of images as NumPy arrays.
        rows: The number of rows in the grid.
        cols: The number of columns in the grid. If None, it will be
              calculated based on the number of rows.

    Returns:
        A NumPy array representing the image grid.
    """
    # Standardize images
    standardized_images = []
    max_h, max_w = 0, 0
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        standardized_images.append(img)
        h, w, _ = img.shape
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

    resized_images = []
    for img in standardized_images:
        resized = cv2.resize(img, (max_w, max_h))
        resized_images.append(resized)

    if not resized_images:
        return None

    num_images = len(resized_images)

    # Determine grid size
    if rows is not None:
        cols = int(np.ceil(num_images / rows))
    elif cols is not None:
        rows = int(np.ceil(num_images / cols))
    else:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

    # Create the grid
    grid_h = max_h * rows
    grid_w = max_w * cols
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Populate the grid
    for i, img in enumerate(resized_images):
        row_idx = i // cols
        col_idx = i % cols
        start_y = row_idx * max_h
        end_y = start_y + max_h
        start_x = col_idx * max_w
        end_x = start_x + max_w
        if row_idx < rows and col_idx < cols:
            grid[start_y:end_y, start_x:end_x, :] = img

    return grid
