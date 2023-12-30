import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("video.mp4")

# Create a list of keypoints and distances
keypoints = []
distances = []

# While the video is playing
while cap.isOpened():

    # Capture the next frame
    ret, frame = cap.read()

    # If the frame is empty, break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the keypoints
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)

    # Match the keypoints
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des, des)

    # Convert the matches to a list
    matches_list = list(matches)

    # Sort the matches by distance
    matches_list.sort(key=lambda x: x.distance)

    # Get the two nearest matches
    match1, match2 = matches_list[:2]

    # Convert the keypoints to NumPy arrays
    keypoint1 = np.array(kp[match1.queryIdx].pt)
    keypoint2 = np.array(kp[match2.queryIdx].pt)

    # Calculate the distance between the two points in pixels
    distance_in_pixels = np.linalg.norm(keypoint1 - keypoint2)

    # Calculate the pixel-to-cm conversion ratio
    # Assume that the distance between the eyes is 6 cm
    pixel_to_cm_ratio = 6 / distance_in_pixels

    # Convert the distance between the two points from pixels to cm
    distance_in_cm = distance_in_pixels * pixel_to_cm_ratio

    # Draw a line between the two points
    cv2.line(frame, (int(keypoint1[0]), int(keypoint1[1])), (int(
        keypoint2[0]), int(keypoint2[1])), (0, 0, 255), 2)

    # Add text to the image
    cv2.putText(frame, str(distance_in_cm) + " cm", (int(keypoint1[0]), int(
        keypoint1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the output image
    cv2.imshow("Output", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
