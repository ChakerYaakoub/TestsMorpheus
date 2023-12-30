import mediapipe as mp
import cv2
import numpy as np

# Create a MediaPipe FaceMesh object.
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define the new neck point.
neck_point = np.array([
    [0.5, 0.8],
])

# Capture the next frame from the webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

# If the frame is not empty, process it with MediaPipe FaceMesh.

    # Convert the frame to RGB format.
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh.
        results = face_mesh.process(rgb_frame)

    # Draw the facial landmarks on the frame.
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_CONTOURS, mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

    # Draw the new neck point on the frame.
        cv2.circle(frame, (int(neck_point[0][0] * frame.shape[1]),
                           int(neck_point[0][1] * frame.shape[0])), 5, (0, 255, 0), -1)

    # Display the frame.
        cv2.imshow('MediaPipe FaceMesh', frame)

    # If the 'q' key is pressed, quit the program.
     # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and destroy all windows.
cap.release()
cv2.destroyAllWindows()
