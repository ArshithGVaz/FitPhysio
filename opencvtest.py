import cv2
import imutils
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
video = cv2.VideoCapture("v3.mp4")
webcam = cv2.VideoCapture(0)

# Loop through the video frames
while video.isOpened() and webcam.isOpened():
    ret_video, frame_video = video.read()
    ret_webcam, frame_webcam = webcam.read()

    if ret_video and ret_webcam:
        fn = video.get(cv2.CAP_PROP_POS_FRAMES)
        if fn % 5 == 0:
            results = model(frame_webcam, conf=0.3, imgsz=160, max_det=1)

            frame_webcam = results[0].plot()

        annotated_frame = imutils.resize(frame_webcam, width=960)

        cv2.imshow("Video Feed", frame_video)
        cv2.imshow("Webcam Feed", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release()
webcam.release()
cv2.destroyAllWindows()
