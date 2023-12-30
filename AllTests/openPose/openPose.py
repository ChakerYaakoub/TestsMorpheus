import cv2
import numpy as np
# Import OpenPose module (adjust the import based on your OpenPose installation)
from openpose import pyopenpose as op

# Load image using OpenCV
image_path = 'your_image.jpg'
image = cv2.imread(image_path)

# Set up OpenPose
params = {
    # Adjust the path to your OpenPose models folder
    "model_folder": "path/to/openpose/models/",
    "hand": False,
    "face": False,
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process image with OpenPose
datum = op.Datum()
datum.cvInputData = image
opWrapper.emplaceAndPop([datum])

# Get keypoints
keypoints = datum.poseKeypoints[0]

# Identify larynx and neck points (adjust indices based on your OpenPose model)
larynx_index = 0  # Adjust based on the keypoint index for the larynx
neck_index = 1  # Adjust based on the keypoint index for the neck
larynx_point = keypoints[larynx_index]
neck_point = keypoints[neck_index]

# Add a new landmark point to the neck (example: midpoint)
new_landmark_point = (larynx_point + neck_point) / 2

# Draw keypoints on the original image
for point in keypoints:
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

# Draw the new landmark point
cv2.circle(image, (int(new_landmark_point[0]), int(
    new_landmark_point[1])), 5, (255, 0, 0), -1)

# Display the result
cv2.imshow('Landmark Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
