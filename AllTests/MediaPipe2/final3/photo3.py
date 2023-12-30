import mediapipe as mp
import cv2

# Read the input image
input_image = cv2.imread('test2.jpg')

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

# Define the landmark indices for the two points you want to measure
landmark_index1 = 0  # Change these indices as per your requirement
landmark_index2 = 17

# If no face landmarks are detected, exit
if not results.multi_face_landmarks:
    print("No face landmarks detected in the image.")
    exit()

# Get the first detected face landmarks
face_landmarks = results.multi_face_landmarks[0]

# Get the specified landmarks
landmark1 = face_landmarks.landmark[landmark_index1]
landmark2 = face_landmarks.landmark[landmark_index2]

# Assuming a reference distance (in pixels) between landmarks 0 and 17
reference_distance_pixels = 100  # Change this value based on your estimate

# Assuming the reference distance corresponds to a certain length in centimeters
reference_distance_cm = 5  # Change this value based on your estimation

# Calculate the normalized distance between the two points
normalized_distance = ((landmark2.x - landmark1.x)**2 +
                       (landmark2.y - landmark1.y)**2)**0.5

estimated_distance_cm = (normalized_distance *
                         reference_distance_cm) / reference_distance_pixels

# Display the normalized distance between the specified landmarks
print(
    f"Normalized Distance between Landmark {landmark_index1} and Landmark {landmark_index2}: {estimated_distance_cm:.4f}")

# Draw circles on the specified landmarks for visualization
landmark1_x, landmark1_y = int(
    landmark1.x * input_image.shape[1]), int(landmark1.y * input_image.shape[0])
landmark2_x, landmark2_y = int(
    landmark2.x * input_image.shape[1]), int(landmark2.y * input_image.shape[0])
cv2.circle(input_image, (landmark1_x, landmark1_y), 5, (0, 255, 0), -1)
cv2.circle(input_image, (landmark2_x, landmark2_y), 5, (0, 255, 0), -1)
cv2.putText(input_image, f"Distance: {estimated_distance_cm:.2f} cm",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Save the image with the distance displayed
cv2.imwrite('output_image2.jpg', input_image)

# Show the image with circles around the specified landmarks
cv2.imshow('Image with Landmarks', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
