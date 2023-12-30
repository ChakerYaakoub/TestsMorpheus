import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.5)

# Initialize MediaPipe Face Landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Open the camera (usually camera index 0 represents the default built-in camera)
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# You can change the output filename and parameters as needed
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cm_per_pixel = 0.025

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    results = face_detection.process(frame)
    if results.detections:
        face = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1, y1, x2, y2 = int(face.xmin * w), int(face.ymin * h), int(
            face.xmin * w + face.width * w), int(face.ymin * h + face.height * h)

        # Crop the face region
        face_region = frame[y1:y2, x1:x2]

        # Detect face landmarks
        face_landmarks = face_mesh.process(
            cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        if face_landmarks.multi_face_landmarks:
            landmarks = face_landmarks.multi_face_landmarks[0]
            # Extract the nose and mouth landmarks
            # Use appropriate landmark indices for your use case
            right = landmarks.landmark[168]  # Nose tip
            left = landmarks.landmark[54]  # Mouth right corner

            # Calculate the Euclidean distance between the nose and mouth
            distance_pixels = ((right.x * (x2 - x1) - left.x * (x2 - x1)) ** 2 +
                               (right.y * (y2 - y1) - left.y * (y2 - y1)) ** 2) ** 0.5
            distance_cm = distance_pixels * cm_per_pixel

            # Draw circles around the nose and mouth for visualization
            cv2.circle(frame, (int(right.x * (x2 - x1) + x1),
                       int(right.y * (y2 - y1) + y1)), 3, (0, 0, 255), -1)
            cv2.circle(frame, (int(left.x * (x2 - x1) + x1),
                       int(left.y * (y2 - y1) + y1)), 3, (0, 0, 255), -1)

            # Draw a line connecting the nose and mouth
            cv2.line(frame, (int(right.x * (x2 - x1) + x1), int(right.y * (y2 - y1) + y1),
                             (int(left.x * (x2 - x1) + x1), int(left.y * (y2 - y1) + y1)), (0, 0, 255), 2))

            # Display the distance on the frame
            cv2.putText(frame, f"Distance: {distance_cm:.2f} cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with landmarks and distance
    cv2.imshow("Facial Landmark Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, VideoWriter, and close the OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
