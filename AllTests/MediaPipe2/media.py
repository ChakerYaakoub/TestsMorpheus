import mediapipe as mp
import cv2
import math

# Create a VideoCapture object to read from the webcam.
cap = cv2.VideoCapture(0)

# Estimate the focal length of the camera.
focal_length = 1000.0

# Create a MediaPipe FaceMesh object.
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# While the webcam is open, read each frame and process it with MediaPipe FaceMesh.
while True:

    # Capture the next frame from the webcam.
    ret, frame = cap.read()

    # If the frame is not empty, process it with MediaPipe FaceMesh.
    if ret:

        # Convert the frame to RGB format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe FaceMesh.
        results = face_mesh.process(rgb_frame)

        # Draw the facial landmarks on the frame.
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_CONTOURS, mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

    # Calculate the normalized coordinates of the two points.
        landmark1_normalized_x = results.multi_face_landmarks[0].landmark[0].x
        landmark1_normalized_y = results.multi_face_landmarks[0].landmark[0].y
        landmark2_normalized_x = results.multi_face_landmarks[0].landmark[17].x
        landmark2_normalized_y = results.multi_face_landmarks[0].landmark[17].y
    # Calculate the normalized distance between the two points.
        normalized_distance = math.sqrt((landmark2_normalized_x - landmark1_normalized_x)**2 + (
            landmark2_normalized_y - landmark1_normalized_y)**2)

    # Calculate the distance between the two points in centimeters.
        distance_in_cm = (focal_length * normalized_distance) / 10.0
        # distance_in_cm = normalized_distance

        # Display the distance between points 17 and 11 on the frame.
        cv2.putText(frame, f"Distance: {distance_in_cm:.2f} cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame.
        cv2.imshow('MediaPipe FaceMesh', frame)

        # If the 'q' key is pressed, quit the program.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and destroy all windows.
cap.release()
cv2.destroyAllWindows()

# we have to read

# https://developers.google.com/mediapipe/api/solutions/python/mp
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/
