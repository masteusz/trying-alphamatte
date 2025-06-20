import cv2
import numpy as np
from pymatting import blend, estimate_alpha_cf, estimate_foreground_ml, load_image
from ultralytics import YOLO

from utils import create_image_grid, show_image


def main():
    image = cv2.imread("images/1.png")

    show_image(image)

    model = YOLO("yolo11n-seg.pt")
    result = model(image)[0]

    annotated_image = result.plot()
    show_image(create_image_grid([image, annotated_image]))

    masks = result.masks.data.cpu()

    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.float32)

    for seg in masks.data.cpu().numpy():
        seg = cv2.resize(seg, (w, h))
        mask += seg

    show_image(create_image_grid([image, annotated_image, mask]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smaller_mask = cv2.erode(mask, kernel, iterations=5)
    bigger_mask = cv2.dilate(mask, kernel, iterations=5)

    show_image(create_image_grid([mask, smaller_mask, bigger_mask]))

    outline = bigger_mask != smaller_mask
    outline = outline.astype(np.uint8)
    outline_image = outline * 255

    show_image(create_image_grid([mask, smaller_mask, bigger_mask, outline_image]))

    mask[outline > 0] = 0.5

    mask_image = (mask * 255).astype(np.uint8)

    show_image(create_image_grid([outline_image, mask_image]))

    img = load_image("images/1.png", "RGB", 1.0, "box")
    back_sf = load_image("images/back.jpg", "RGB", 1.0, "box")
    background = np.zeros(image.shape)
    background[:, :] = [0.5, 0.5, 0.5]

    alpha = estimate_alpha_cf(img, mask)
    foreground = estimate_foreground_ml(img, alpha)

    alpha_image = cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    foreground_image = cv2.cvtColor((foreground * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    show_image(create_image_grid([alpha_image, foreground_image]))

    color_bleeding = blend(img, back_sf, alpha)

    color_bleeding_image = cv2.cvtColor((color_bleeding * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    show_image(create_image_grid([image, color_bleeding_image]))


if __name__ == "__main__":
    main()
