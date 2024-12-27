import cv2
import numpy as np
from config import PERSPECTIVE_SRC, PERSPECTIVE_DST

def warp_perspective(image):
    """Perform perspective transformation."""
    img_size = (image.shape[1], image.shape[0])
    src = np.array(PERSPECTIVE_SRC, dtype=np.float32)
    dst = np.array(PERSPECTIVE_DST, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size)

    return warped, M_inv


