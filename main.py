from lane_detection import fit_polynomial, find_lane_pixels, find_lane_pixels_from_prev_poly
from measurements import measure_curvature, measure_vehicle_position
from perspective import warp_image 
from thresholds import binary_threshold
from utils import draw_lane_info, draw_lane_lines
import cv2
import numpy as np

# Global variables for lane line history
left_fit_hist = []
right_fit_hist = []

def lane_finding_pipeline(image, draw_info = True, draw_lines = False):
    """Main pipeline for lane finding."""
    # Binary thresholding and warping
    binary_thresh = binary_threshold(image)
    binary_warped, M_inv = warp_image(binary_thresh)

    # Fit lane lines
    if len(left_fit_hist) == 0:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    else:
        prev_left_fit = np.mean(left_fit_hist, axis=0)
        prev_right_fit = np.mean(right_fit_hist, axis=0)
        leftx, lefty, rightx, righty = find_lane_pixels_from_prev_poly(binary_warped, prev_left_fit, prev_right_fit)
        if len(lefty) == 0 or len(righty) == 0:
            leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)

    # Update history
    left_fit_hist = np.vstack([left_fit_hist, left_fit])[-10:]
    right_fit_hist = np.vstack([right_fit_hist, right_fit])[-10:]

    # Calculate curvature and vehicle position
    left_curverad, right_curverad = measure_curvature(ploty, left_fitx, right_fitx)
    veh_pos = measure_vehicle_position(binary_warped, left_fit, right_fit)

    if draw_info:
        # Project lane and information back onto the original image
        result = draw_lane_info(image, ploty, left_fitx, right_fitx, left_curverad, right_curverad, veh_pos)
    elif draw_lines:
        result = draw_lane_lines(image, binary_warped, left_fitx, right_fitx, ploty)
    elif draw_info and draw_lines:
        result = draw_lane_info(image, ploty, left_fitx, right_fitx, left_curverad, right_curverad, veh_pos)
        result = draw_lane_lines(result, binary_warped, left_fitx, right_fitx, ploty)
    else:
        result = image
    return result, veh_pos


if __name__ == "__main__":
    
    # options: real_time, video
    REAL_TIME = False

    if REAL_TIME:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result, veh_pos = lane_finding_pipeline(frame)
            print(f"Vehicle position: {veh_pos:.2f} m")
            cv2.imshow('Video', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Load test Video
        cap = cv2.VideoCapture('project_video.mp4')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result, veh_pos = lane_finding_pipeline(frame)
            print(f"Vehicle position: {veh_pos:.2f} m")
            cv2.imshow('Video', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    