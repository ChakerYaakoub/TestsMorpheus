import mediapipe as mp
import cv2
import math

# Read the input image
input_image = cv2.imread('test2.jpg')

# Estimate the focal length of the camera
focal_length = 1000.0

# Create a MediaPipe FaceMesh object
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Convert the image to RGB format
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe FaceMesh
results = face_mesh.process(rgb_image)

# If no face landmarks are detected, exit
if not results.multi_face_landmarks:
    print("No face landmarks detected in the image.")
    exit()

# Draw the facial landmarks on the image
mp.solutions.drawing_utils.draw_landmarks(
    input_image, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_CONTOURS, mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

# Calculate the normalized coordinates of the two points
landmark1_normalized_x = results.multi_face_landmarks[0].landmark[0].x
landmark1_normalized_y = results.multi_face_landmarks[0].landmark[0].y
landmark2_normalized_x = results.multi_face_landmarks[0].landmark[17].x
landmark2_normalized_y = results.multi_face_landmarks[0].landmark[17].y

# Calculate the normalized distance between the two points
normalized_distance = math.sqrt((landmark2_normalized_x - landmark1_normalized_x)**2 + (
    landmark2_normalized_y - landmark1_normalized_y)**2)

# Calculate the distance between the two points in centimeters
distance_in_cm = (focal_length * normalized_distance) / 10.0

# Get the first detected face landmarks
face_landmarks = results.multi_face_landmarks[0]
# Draw the specific facial landmarks on the image
specific_points = [0, 17]
for point_idx in specific_points:
    landmark = face_landmarks.landmark[point_idx]
    landmark_x = int(landmark.x * input_image.shape[1])
    landmark_y = int(landmark.y * input_image.shape[0])
    cv2.circle(input_image, (landmark_x, landmark_y), 5, (173, 255, 47), -1)

# Display the distance between points 17 and 11 on the image
cv2.putText(input_image, f"Distance: {distance_in_cm:.2f} cm",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Save the image with the distance displayed
cv2.imwrite('output_image2.jpg', input_image)
