import time
from threading import Thread

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

similarity_values = []


def normalize_keypoints(keypoints, anchor_idx1, anchor_idx2):
    # Extract reference keypoints
    x1, y1 = keypoints[anchor_idx1]
    x2, y2 = keypoints[anchor_idx2]

    # Calculate the scaling factor (distance between anchor keypoints)
    scale = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate the anchor point (e.g., midpoint)
    anchor_x, anchor_y = (x1 + x2) / 2, (y1 + y2) / 2

    # Normalize all keypoints
    normalized_keypoints = []
    for x, y in keypoints:
        norm_x = (x - anchor_x) / scale
        norm_y = (y - anchor_y) / scale
        normalized_keypoints.append((norm_x, norm_y))

    return np.array(normalized_keypoints)


def compute_cosine_similarity(pose1, pose2):
    # Flatten the keypoints into a single vector
    pose1_vector = pose1.flatten()
    pose2_vector = pose2.flatten()

    # Compute dot product and magnitudes
    dot_product = np.dot(pose1_vector, pose2_vector)
    magnitude1 = np.linalg.norm(pose1_vector)
    magnitude2 = np.linalg.norm(pose2_vector)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Treat zero-magnitude vectors as orthogonal

    # Compute cosine similarity
    return dot_product / (magnitude1 * magnitude2)


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
                        points1 = normalize_keypoints(points1[0][5:11, :], anchor_idx1=0, anchor_idx2=1)
                        points2 = normalize_keypoints(points2[0][5:11, :], anchor_idx1=0, anchor_idx2=1)

                        similarity = compute_cosine_similarity(points1, points2)

                        similarity_values.append(similarity)

            # Display the annotated frame
            annotated_frame1 = imutils.resize(frame1, width=960)
            annotated_frame2 = imutils.resize(frame2, width=960)

            cv2.imshow("Trainer", annotated_frame1)
            cv2.imshow("User", annotated_frame2)

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


# Start the detect_pose function in a separate thread
thread = Thread(
    target=detect_pose,
    args=("v3.mp4", "jump2.mp4"),
    # daemon=True,  # Ensures thread exits when the main program ends
).start()

# Set up the plot on the main thread
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
x_data, y_data = [], []  # Empty lists for x and y data
(line,) = ax.plot(x_data, y_data, "-o")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Similarity")
ax.set_title("Real-Time Similarity Plot")
ax.set_ylim(0, 1)  # Fix y-axis range between 0 and 1

start_time = time.time()

# Main loop to update the plot
try:
    while True:
        # Get elapsed time
        elapsed_time = time.time() - start_time

        # Check if there are new similarity values to plot
        while similarity_values:
            y_data.append(similarity_values.pop(0))  # Append similarity value
            x_data.append(elapsed_time)  # Append elapsed time for x-axis

        # Update the plot
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Adjust the view automatically

        # Redraw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)  # Adjust for smoother or slower updates

except KeyboardInterrupt:
    print("Plotting stopped by user.")

# Clean up and close windows
plt.ioff()
plt.show()
