import cv2
import numpy as np 
from perspective import warp_perspective
def draw_lane_lines(binary_warped, left_fitx, right_fitx, ploty):
    """
    Overlays detected lane lines on the binary image.

    Parameters:
        binary_warped (numpy.ndarray): Binary warped image.
        left_fitx, right_fitx (numpy.ndarray): Fitted x-coordinates for left and right lanes.
        ploty (numpy.ndarray): y-coordinates for plotting.

    Returns:
        numpy.ndarray: Image with lane lines overlaid.
    """
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255 
    window_img = np.zeros_like(out_img)

    margin = 100
    left_line_pts = np.hstack((
        np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))]),
        np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])))
    right_line_pts = np.hstack((
        np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))]),
        np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

def draw_lane_info(img, ploty, left_fitx, right_fitx, left_curverad, right_curverad, veh_pos):
    """Overlay lane information onto the original image."""
    binary_warped, M_inv = warp_perspective(img)
    # Create blank image for lane visualization
    warp_zero = np.zeros(binary_warped.shape[:2], dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Points for lane polygon
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane polygon
    cv2.fillPoly(color_warp, [pts.astype(np.int32)], (0, 255, 0))

    # Warp back to original image space
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Display curvature and position
    cv2.putText(result, f'Curve Radius [m]: {(left_curverad + right_curverad) / 2:.2f}', (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Center Offset [m]: {veh_pos:.2f}', (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result
