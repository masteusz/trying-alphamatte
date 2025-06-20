import cv2
import numpy as np
from ultralytics import YOLO


def show_image(_image: np.ndarray) -> None:
    cv2.imshow("Image", _image)
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
        # Default to a single row
        rows = 1
        cols = num_images

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


def main():
    image = cv2.imread("images/1.png")

    show_image(image)

    model = YOLO("yolo11n-seg.pt")
    result = model(image)[0]

    annotated_image = result.plot()
    show_image(create_image_grid([image, annotated_image]))

    masks = result.masks.data.cpu()

    h, w, _ = image.shape
    mask_image = np.zeros((h, w), dtype=np.float32)

    for seg in masks.data.cpu().numpy():
        seg = cv2.resize(seg, (w, h))
        mask_image += seg

    show_image(create_image_grid([image, annotated_image, mask_image]))

    kernel = np.ones((7, 7))
    smaller_mask_image = cv2.erode(mask_image, kernel, iterations=5)
    bigger_mask_image = cv2.dilate(mask_image, kernel, iterations=5)

    show_image(create_image_grid([mask_image, smaller_mask_image, bigger_mask_image]))

    # outline = bigger_mask_image != smaller_mask_image
    # outline = outline.astype(np.uint8)
    #
    # mask_image[outline > 0] = 0.5
    #
    # cv2.imshow("mask", mask_image)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # img = load_image("images/2.png", "RGB", 1.0, "box")
    # # mask_image = image.astype(np.float32) / 255
    #
    # background = np.zeros(image.shape)
    # background[:, :] = [0.5, 0.5, 0.5]
    #
    # alpha = estimate_alpha_cf(img, mask_image)
    # foreground = estimate_foreground_ml(img, alpha)
    # new_image = blend(foreground, background, alpha)
    #
    # images = [img, mask_image, alpha, new_image]
    # grid = make_grid(images)
    # save_image("grid.png", grid)
    #
    # cutout = stack_images(foreground, alpha)
    # save_image("cutout.png", cutout)
    #
    # color_bleeding = blend(img, background, alpha)
    # grid = make_grid([color_bleeding, new_image])
    # save_image("color_bleeding.png", grid)


if __name__ == "__main__":
    main()
