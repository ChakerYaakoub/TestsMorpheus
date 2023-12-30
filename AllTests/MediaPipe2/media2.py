import cv2
import mediapipe as mp
import numpy as np

# Define the focal length of the camera in millimeters
focal_length = 3.6

# Define the pixel size of the camera sensor in micrometers
pixel_size = 1.4

# Create a MediaPipe face detection pipeline
mp_face_detection = mp.solutions.face_detection

# Initialize the face detection pipeline
with mp_face_detection.FaceDetection() as face_detection:

    # Start the video capture
    cap = cv2.VideoCapture(0)

    while True:

        # Capture a frame
        ret, frame = cap.read()

        # If the frame is empty, break the loop
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the MediaPipe face detection pipeline
        results = face_detection.process(frame_rgb)

        # If no faces are detected, skip to the next frame
        if len(results.detections) == 0:
            continue

        # Get the first detected face
        detection = results.detections[0]

        # Calculate the distance between the two eyes
        distance_in_pixels = np.linalg.norm(np.squeeze(detection.location_data.relative_keypoints[0].x) - np.squeeze(detection.location_data.relative_keypoints[1].x), np.squeeze(
            detection.location_data.relative_keypoints[0].y) - np.squeeze(detection.location_data.relative_keypoints[1].y))

        # Convert the distance in pixels to centimeters
        distance_in_cm = distance_in_pixels * focal_length / pixel_size

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance_in_cm:0.2f} cm",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()
