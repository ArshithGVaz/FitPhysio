import threading
import time

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Define video sources
SOURCES = ["jump2.mp4", "v2.mp4"]  # local video, 0 for webcam

# Shared data structure for keypoint data
keypoint_data = [None, None]

# Store previous keypoint values to track changes
previous_keypoints = [None, None]


def run_tracker_in_thread(filename, index):
    model = YOLO("yolo11m-pose.pt")
    results = model(filename, stream=True, conf=0.3, imgsz=160, max_det=1, vid_stride=5, save=True)
    for r in results:
        if r.keypoints:
            # Extract and print xy coordinates for keypoints 5, 6, 7, 8, 9, and 10
            keypoint_data[index] = r.keypoints.xyn[0][5]  # Assuming a single person (index 0)
            print(keypoint_data)
        else:
            print("No keypoints detected in this frame.")


# Create and start tracker threads using a for loop
tracker_threads = []
for i, video_file in enumerate(SOURCES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(video_file, i), daemon=True)
    tracker_threads.append(thread)
    thread.start()

# Set up the plot on the main thread
plt.ion()
fig, ax = plt.subplots()
y_data = []
x_data = []
(line,) = ax.plot(x_data, y_data, "-o")
ax.set_xlabel("Time")
ax.set_ylabel("Keypoint Difference (Euclidean Distance)")

start_time = time.time()

# Plot updating loop in the main thread
try:
    while any(thread.is_alive() for thread in tracker_threads):
        if keypoint_data[0] is not None and keypoint_data[1] is not None:
            # Check if keypoints have changed
            if (previous_keypoints[0] is None or not torch.equal(keypoint_data[0], previous_keypoints[0])) and (
                previous_keypoints[1] is None or not torch.equal(keypoint_data[1], previous_keypoints[1])
            ):
                # Calculate Euclidean distance between the two keypoints
                point1 = keypoint_data[0]
                point2 = keypoint_data[1]
                diff = torch.dist(point1, point2).item()  # Euclidean distance
                elapsed_time = time.time() - start_time

                # Append data for plotting
                x_data.append(elapsed_time)
                y_data.append(diff)
                line.set_xdata(x_data)
                line.set_ydata(y_data)
                ax.relim()
                ax.autoscale_view()

                # Redraw plot
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Update previous keypoints
                previous_keypoints[0] = keypoint_data[0].clone()
                previous_keypoints[1] = keypoint_data[1].clone()

        time.sleep(0.1)  # Adjust for smoother or slower updates

except KeyboardInterrupt:
    print("Plotting stopped by user.")

# Wait for tracker threads to complete
for thread in tracker_threads:
    thread.join()

# Clean up and close windows
plt.ioff()
plt.show()
cv2.destroyAllWindows()
