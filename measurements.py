import numpy as np

def measure_curvature(ploty, left_fitx, right_fitx):
    """Calculate the curvature of the lane."""
    ym_per_pix = 30 / 720  # Meters per pixel in y-dimension
    xm_per_pix = 3.7 / 700  # Meters per pixel in x-dimension
    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Radius of curvature at the maximum y-value
    y_eval = np.max(ploty) * ym_per_pix
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / abs(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / abs(2 * right_fit_cr[0])

    return left_curverad, right_curverad

def measure_vehicle_position(image, left_fit, right_fit):
    """Calculate the vehicle's position relative to the center of the lane."""
    xm_per_pix = 3.7 / 700
    y_max = image.shape[0]

    left_x = np.polyval(left_fit, y_max) # left_fit[0] * y_max**2 + left_fit[1] * y_max + left_fit[2]
    right_x = np.polyval(right_fit, y_max) # right_fit[0] * y_max**2 + right_fit[1] * y_max + right_fit[2]
    
    lane_center = (left_x + right_x) / 2
    vehicle_center = image.shape[1] / 2

    # Vehicle position offset
    vehicle_position = (vehicle_center - lane_center) * xm_per_pix
    return vehicle_position
