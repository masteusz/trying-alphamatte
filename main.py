import cv2
import numpy as np
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")
    image = cv2.imread("images/1.png")
    h, w, _ = image.shape
    mask_image = np.zeros((h, w), dtype=np.uint8)
    result = model(image)[0]

    masks = result.masks.data.cpu()

    for seg in masks.data.cpu().numpy():
        seg = cv2.resize(seg, (w, h))
        seg = seg.astype(np.uint8) * 255
        mask_image += seg

    kernel = np.ones((7, 7))
    smaller_mask_image = mask_image.copy()
    smaller_mask_image = cv2.erode(smaller_mask_image, kernel, iterations=5)
    outline = mask_image != smaller_mask_image
    outline = outline.astype(np.uint8)

    cv2.imshow("mask", mask_image)
    cv2.waitKey()

    mask_image[outline > 0] = 127

    cv2.imshow("mask", mask_image)
    cv2.waitKey()

    pass


if __name__ == "__main__":
    main()
