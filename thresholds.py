import cv2
import numpy as np
from config import THRESHOLDS

def binary_threshold(image):
    """
    Applies various thresholding techniques to the input image to highlight lane lines.
    
    Parameters:
        img (numpy.ndarray): The input BGR image.

    Returns:
        numpy.ndarray: A binary image combining Sobel, color, and intensity thresholds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel Thresholding
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_binary = (scaled_sobel >= THRESHOLDS["sobel_min"]) & (scaled_sobel <= THRESHOLDS["sobel_max"])

    # White Pixel Detection
    white_binary = (gray >= THRESHOLDS["white_min"]) & (gray <= THRESHOLDS["white_max"])

    # HLS Thresholding
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    sat_binary = (hls[:, :, 2] >= THRESHOLDS["saturation_min"]) & (hls[:, :, 2] <= THRESHOLDS["saturation_max"])
    hue_binary = (hls[:, :, 0] >= THRESHOLDS["hue_min"]) & (hls[:, :, 0] <= THRESHOLDS["hue_max"])

    # Combine Thresholds
    combined_binary = np.zeros_like(gray)
    combined_binary[sobel_binary | white_binary | sat_binary | hue_binary] = 255

    return combined_binary
