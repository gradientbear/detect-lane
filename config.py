# Perspective transformation
PERSPECTIVE_SRC = [
    (190, 720), (596, 447), (685, 447), (1125, 720)
]
PERSPECTIVE_DST = [
    (300, 720), (300, 0), (980, 0), (980, 720)
]

# Thresholding values
THRESHOLDS = {
    "sobel_min": 30,
    "sobel_max": 255,
    "white_min": 200,
    "white_max": 255,
    "saturation_min": 90,
    "saturation_max": 255,
    "hue_min": 10,
    "hue_max": 25,
}

# Sliding window parameters
WINDOW_PARAMS = {
    "nwindows": 9,
    "margin": 100,
    "minpix": 50,
}

