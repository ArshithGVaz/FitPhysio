# from threading import Thread

import math
import pickle

import cv2
import imutils
from ultralytics import YOLO

file = open("scale.pkl", "rb")
delta = pickle.load(file)

diff_list = []


def detect_pose(path1, path2):
    model = YOLO("yolo11m-pose.pt")
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    while cap1.isOpened() and cap2.isOpened():
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()

        if success1 and success2:
            fn = cap1.get(cv2.CAP_PROP_POS_FRAMES)
            if fn % 5 == 0:
                results1 = model(frame1, conf=0.3, imgsz=160, max_det=1)
                results2 = model(frame2, conf=0.3, imgsz=160, max_det=1)

                frame1 = results1[0].plot()
                frame2 = results2[0].plot()

                for r1, r2 in zip(results1, results2):
                    if r2.keypoints:
                        points1 = r1.keypoints.xy.numpy()
                        points2 = r2.keypoints.xy.numpy()

                        # Make point 5 as 0 for both arrays
                        points1[0][:, 0] -= points1[0][5, 0]
                        points1[0][:, 1] -= points1[0][5, 1]

                        points2[0][:, 0] -= points2[0][5, 0]
                        points2[0][:, 1] -= points2[0][5, 1]

                        points2[0] += delta

                        diff = 0
                        for i in range(0, 17):
                            # print(math.dist(points1[0][i], points2[0][i]))
                            diff += math.dist(points1[0][i], points2[0][i])

                        diff_list.append(diff)

            # Display the annotated frame
            annotated_frame1 = imutils.resize(frame1, width=960)
            annotated_frame2 = imutils.resize(frame2, width=960)

            cv2.imshow(path1, annotated_frame1)
            cv2.imshow(path2, annotated_frame2)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


detect_pose("v3.mp4", "jump2.mp4")

print(diff_list)
