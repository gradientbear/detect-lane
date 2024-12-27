# Advanced Lane Detection Based on Digital Image Processing

This repository provides a Python-based implementation of a lane detection and vehicle positioning System. It utilizes computer vision techniques to detect lane lines on a road, measure their curvature, and calculate the vehicle's position relative to the center of the lane. The system is capable of processing real-time video streams or prerecorded videos.

---
## Features
- **Binary Thresholding**: Applies multiple thresholds (color, gradient) to create a binary representation of the image.
- **Perspective Transform**: Warps the road to a bird's-eye view for better lane detection.
- **Sliding Window Search**: Locates lane lines using a histogram-based search or previous polynomial fits for efficiency.
- **Lane Line Fitting**: Fits a second-order polynomial to detected lane pixels.
- **Curvature Calculation**: Computes the radius of curvature for left and right lane lines.
- **Vehicle Positioning**: Determines the vehicle's offset from the lane center.
- **Visualization**: Projects lane information back onto the original image or video.

---

## Directory Structure
```
.
├── main.py               # Main execution script
├── config.py             # Configuration settings
├── lane_detection.py     # Functions for lane detection and polynomial fitting
├── measurements.py       # Functions to calculate curvature and vehicle position
├── perspective.py        # Perspective transformation functions
├── thresholds.py         # Binary thresholding functions
├── utils.py              # Utility functions for visualization
```
## Dependencies
- Python 3.8 or higher
- OpenCV
- NumPy
- Matplotlib
---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/eyadgad/Advanced-Lane-Detection-Based-on-Digital-Image-Processing.git
cd Advanced-Lane-Detection-Based-on-Digital-Image-Processing
```

### 2. Run the Code
#### Option 1: Process a prerecorded video
Set `REAL_TIME = False` and replace the `project_video.mp4` in `main.py` with your video file and run:
```bash
python main.py
```

#### Option 2: Real-Time Detection
Set `REAL_TIME = True` in `main.py` to enable real-time lane detection from your webcam:
```bash
python main.py
```

### 3. Outputs
- **Vehicle Position**: Printed to the terminal (e.g., `Vehicle position: 0.25 m`).
- **Lane Visualization**: Displays lane lines and curvature overlaid on the input video.

---

## Code Overview

### **`lane_detection.py`**
- `fit_polynomial(binary_warped, leftx, lefty, rightx, righty)`: Fits second-order polynomials to lane lines.
- `find_lane_pixels(binary_warped)`: Identifies lane pixels using a sliding window approach.
- `find_lane_pixels_from_prev_poly(binary_warped, left_fit, right_fit)`: Finds lane pixels near previous polynomial fits.

### **`measurements.py`**
- `measure_curvature(ploty, left_fitx, right_fitx)`: Calculates the curvature of the lane lines.
- `measure_vehicle_position(binary_warped, left_fit, right_fit)`: Computes the vehicle's offset from the center.

### **`perspective.py`**
- `warp_image(img)`: Warps the input image to a bird's-eye view.

### **`thresholds.py`**
- `binary_threshold(img)`: Applies gradient and color thresholding to generate a binary image.

### **`utils.py`**
- `draw_lane_info(image, ploty, left_fitx, right_fitx, left_curverad, right_curverad, veh_pos)`: Overlays lane information on the input image.
- `draw_lane_lines(image, binary_warped, left_fitx, right_fitx, ploty)`: Visualizes detected lane lines.
