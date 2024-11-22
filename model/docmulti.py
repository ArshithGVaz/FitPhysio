from threading import Thread

import cv2
import imutils
from ultralytics import YOLO


def thread_safe_predict(path):
    """Predict on an image using a new YOLO model instance in a thread-safe manner; takes video path as input."""

    model = YOLO("yolo11m-pose.pt")
    cap = cv2.VideoCapture(path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame, conf=0.3, imgsz=160, max_det=1)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            annotated_frame = imutils.resize(annotated_frame, width=960)
            cv2.imshow(path, annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()


# Starting threads that each have their own model instance
Thread(target=thread_safe_predict, args=("jump2.mp4",)).start()
Thread(target=thread_safe_predict, args=("v2.mp4",)).start()

cv2.destroyAllWindows()
