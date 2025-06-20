import cv2
import numpy as np
from pymatting import blend, estimate_alpha_cf, estimate_foreground_ml, load_image, make_grid, save_image, stack_images
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")
    image = cv2.imread("images/2.png")
    h, w, _ = image.shape
    mask_image = np.zeros((h, w), dtype=np.float32)
    result = model(image)[0]

    masks = result.masks.data.cpu()

    for seg in masks.data.cpu().numpy():
        seg = cv2.resize(seg, (w, h))
        mask_image += seg

    kernel = np.ones((7, 7))
    smaller_mask_image = cv2.erode(mask_image, kernel, iterations=10)
    bigger_mask_image = cv2.dilate(mask_image, kernel, iterations=10)

    outline = bigger_mask_image != smaller_mask_image
    outline = outline.astype(np.uint8)

    mask_image[outline > 0] = 0.5

    # cv2.imshow("mask", mask_image)
    # cv2.imshow("image", image)
    # cv2.waitKey()

    img = load_image("images/2.png", "RGB", 1.0, "box")
    # mask_image = image.astype(np.float32) / 255

    background = np.zeros(image.shape)
    background[:, :] = [0.5, 0.5, 0.5]

    alpha = estimate_alpha_cf(img, mask_image)
    foreground = estimate_foreground_ml(img, alpha)
    new_image = blend(foreground, background, alpha)

    images = [img, mask_image, alpha, new_image]
    grid = make_grid(images)
    save_image("grid.png", grid)

    cutout = stack_images(foreground, alpha)
    save_image("cutout.png", cutout)

    color_bleeding = blend(img, background, alpha)
    grid = make_grid([color_bleeding, new_image])
    save_image("color_bleeding.png", grid)

    pass


if __name__ == "__main__":
    main()
