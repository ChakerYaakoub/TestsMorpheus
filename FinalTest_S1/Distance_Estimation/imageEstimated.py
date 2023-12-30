import numpy as np
import mediapipe as mp
import cv2 as cv
import utils
from utils import FPS

# face detector function
# estimated from an image


def detect_face(frame):
    # convert the frame from BGR to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # process the rgb image to get face points
    results = face_detector.process(rgb_frame)

    # get the width and height of the image
    img_height, img_width = frame.shape[:2]

    # create the empty list to hold faces data.
    faces = []

    # check if face found or not
    if results.detections:
        # loop through detections
        for detection in results.detections:
            # get the score/confidence of the face
            score = detection.score

            # get face rect and convert into pixel coordinates
            face_rect = np.multiply(
                [
                    detection.location_data.relative_bounding_box.xmin,
                    detection.location_data.relative_bounding_box.ymin,
                    detection.location_data.relative_bounding_box.width,
                    detection.location_data.relative_bounding_box.height,
                ],
                [img_width, img_height, img_width, img_height],
            ).astype(int)

            # create the dict of data
            face_dict = {
                "box": face_rect,
                "score": score[0] * 100,
            }

            faces.append(face_dict)
    return faces


# focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# Load the input image
# you have to add an image that have a face  : test.jpg   !!!!!!!!!!!!!!!!!!1
input_image = cv.imread('test.jpg')

# Define the fonts
fonts = cv.FONT_HERSHEY_PLAIN

# Reference image (just for focal length calculation)
ref_image = cv.imread("./Ref_image.png")

# camera object
cap = cv.VideoCapture(0)
calc_fps = FPS()

# variables
# distance from camera to object (face) measured
KNOWN_DISTANCE = 76.2  # centimeters
# width of face in the real world or Object Plane
KNOWN_FACE_WIDTH = 14.3  # centimeters

# configure the Face detection model parameters
with mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6,
) as face_detector:
    face_data = detect_face(ref_image)

    # get the width of the face in the reference image
    rx, ry, rw, rh = face_data[0]["box"]
    # calculate the focal length from the reference image
    focal_point = focal_length_finder(KNOWN_DISTANCE, KNOWN_FACE_WIDTH, rw)

    # detecting face in frame
    faces = detect_face(input_image)

    # check if face found or not
    if faces is not None:
        for face in faces:
            score = face["score"]
            box = face["box"]
            x, y, w, h = box
            distance = distance_finder(focal_point, KNOWN_FACE_WIDTH, w)
            utils.text_with_background(
                input_image,
                f"Distance: {distance:.1f} cm",
                (x, y - 10),
                fonts,
                color=(0, 255, 255),
            )
            utils.rect_corners(input_image, box, (0, 255, 255), th=3)

            # color the face ----

            # Display the image with annotated distances
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            # Convert the image to RGB format
            rgb_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
            #  Process the image with MediaPipe FaceMesh
            results = face_mesh.process(rgb_image)
            # If no face landmarks are detected, exit
            if not results.multi_face_landmarks:
                print("No face landmarks detected in the image.")
                # exit()
            else:
                # Draw the facial landmarks on the image
                mp.solutions.drawing_utils.draw_landmarks(
                    input_image, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_CONTOURS, mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

            # end color the face ----

    cv.imshow("Image", input_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
