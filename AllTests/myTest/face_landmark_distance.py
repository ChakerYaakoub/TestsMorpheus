import cv2
import dlib

# Load the pre-trained facial landmark predictor model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

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

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # Extract the nose and mouth landmarks
        # 12 Landmark
        right = landmarks.part(12)
        # Landmark 4
        left = landmarks.part(4)

        # Calculate the Euclidean distance between the nose and mouth
        distance_pixels = ((right.x - left.x) ** 2 +
                           (right.y - left.y) ** 2) ** 0.5
        distance_cm = distance_pixels * cm_per_pixel

        # Draw circles around the nose and mouth for visualization
        cv2.circle(frame, (right.x, right.y), 3, (0, 0, 255), -1)
        cv2.circle(frame, (left.x, left.y), 3, (0, 0, 255), -1)

        # Draw a line connecting the nose and mouth
        cv2.line(frame, (right.x, right.y), (left.x, left.y), (0, 0, 255), 2)

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
