import cv2
import numpy as np

# Face detection model
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Video capture
cap = cv2.VideoCapture('video.mp4')

# Output video
output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(
    *'MJPG'), 25, (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Extract the two points from the face
    point1 = (faces[0][0], faces[0][1])
    point2 = (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3])

    # Calculate the distance between the two points
    distance = np.linalg.norm(point1 - point2)

    # Convert the distance from pixels to cm
    conversion_factor = 1.0  # cm/pixel
    distance_in_cm = distance * conversion_factor

    # Display the distance
    cv2.putText(frame, 'Distance: {} cm'.format(distance_in_cm),
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output video
    output_video.write(frame)

cap.release()
output_video.release()
