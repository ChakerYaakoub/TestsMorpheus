import cv2
import dlib
from scipy.spatial import distance as dist

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to convert dlib's rectangle to OpenCV style bounding box [x, y, w, h]


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Function to convert dlib's shape object to a list of (x, y) tuples


def shape_to_np(shape, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords

# Function to calculate distance between two points in cm


def calculate_distance(point1, point2, dpi):
    # Calculate the distance in pixels
    pixel_distance = dist.euclidean(point1, point2)
    # Convert pixels to cm using DPI (dots per inch) value
    inches = pixel_distance / dpi
    cm_distance = inches * 2.54
    return cm_distance


# Capture video from file
cap = cv2.VideoCapture('video.mp4')
dpi = 96  # This value will vary depending on your video and camera quality

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # Example calculation: distance between chin and nose bridge
        distance_cm = calculate_distance(
            shape[33], shape[8], dpi)  # Nose and chin points

        # Annotate the frame with distance
        cv2.putText(frame, "{:.1f}cm".format(distance_cm), (shape[33][0], shape[33][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
