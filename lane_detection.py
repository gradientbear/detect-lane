import numpy as np
from config import WINDOW_PARAMS

def find_lane_pixels(binary_warped):
    """
    Identifies lane pixels using a histogram-based sliding window technique.

    Parameters:
        binary_warped (numpy.ndarray): Binary image with lane features highlighted.

    Returns:
        tuple: (leftx, lefty, rightx, righty) pixel coordinates of the detected lanes.
    """
    # Histogram for detecting peaks in the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding window parameters
    nwindows = WINDOW_PARAMS["nwindows"]
    margin = WINDOW_PARAMS["margin"]
    minpix = WINDOW_PARAMS["minpix"]
    window_height = binary_warped.shape[0] // nwindows

    # Extract non-zero pixel indices
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    # Initialize window starting positions
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Lists to store lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Define window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        # Identify non-zero pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append detected pixels to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter window based on mean pixel positions
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate pixel indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract pixel coordinates
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def find_lane_pixels_from_prev_poly(binary_warped, prev_left_fit, prev_right_fit, margin=100):
    """Find lane pixels using previously fitted polynomial."""
    # Get nonzero pixel positions
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Identify lane pixels within the margin around the previous polynomial
    left_lane_inds = ((nonzerox > (prev_left_fit[0] * (nonzeroy ** 2) +
                                   prev_left_fit[1] * nonzeroy +
                                   prev_left_fit[2] - margin)) &
                      (nonzerox < (prev_left_fit[0] * (nonzeroy ** 2) +
                                   prev_left_fit[1] * nonzeroy +
                                   prev_left_fit[2] + margin))).nonzero()[0]

    right_lane_inds = ((nonzerox > (prev_right_fit[0] * (nonzeroy ** 2) +
                                    prev_right_fit[1] * nonzeroy +
                                    prev_right_fit[2] - margin)) &
                       (nonzerox < (prev_right_fit[0] * (nonzeroy ** 2) +
                                    prev_right_fit[1] * nonzeroy +
                                    prev_right_fit[2] + margin))).nonzero()[0]

    # Extract lane pixel positions
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    """
    Fits a second-degree polynomial to the detected lane pixels.

    Parameters:
        binary_warped (numpy.ndarray): Binary warped image.
        leftx, lefty, rightx, righty (numpy.ndarray): Pixel coordinates for left and right lanes.

    Returns:
        tuple: Polynomial coefficients and fitted x, y coordinates for plotting.
    """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty

