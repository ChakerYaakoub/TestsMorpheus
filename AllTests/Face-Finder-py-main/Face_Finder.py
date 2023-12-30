import cv2
import numpy as np

# Load the image
image = cv2.imread("image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the two points in the image
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Match the keypoints
matcher = cv2.DescriptorMatcher_create(
    cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors, descriptors)

# Convert the matches to a list
matches_list = list(matches)

# Sort the matches by distance
matches_list.sort(key=lambda x: x.distance)

# Get the two nearest matches
match1, match2 = matches_list[:2]

# Convert the keypoints to NumPy arrays
keypoint1 = np.array(keypoints[match1.queryIdx].pt)
keypoint2 = np.array(keypoints[match2.queryIdx].pt)

# Calculate the distance between the two points in pixels
distance_in_pixels = np.linalg.norm(keypoint1 - keypoint2)

# Calculate the pixel-to-cm conversion ratio
# Assume that the distance between the eyes is 6 cm
pixel_to_cm_ratio = 6 / distance_in_pixels

# Convert the distance between the two points from pixels to cm
distance_in_cm = distance_in_pixels * pixel_to_cm_ratio

# Print the distance in cm
print("Distance in cm:", distance_in_cm)

# Find the point from where to where
point_from = keypoint1
point_to = keypoint2

# Print the point from where to where
print("Point from:", point_from)
print("Point to:", point_to)

# Draw a line between the two points
cv2.line(image, (int(point_from[0]), int(point_from[1])), (int(
    point_to[0]), int(point_to[1])), (0, 0, 255), 2)

# Add text to the image
cv2.putText(image, str(distance_in_cm) + " cm", (int(point_from[0]), int(
    point_from[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the output image
cv2.imwrite("output.jpg", image)
