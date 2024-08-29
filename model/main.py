import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

class VideoFile(BaseModel):
    file: UploadFile

landmark_pairs = [
    [7, 13], [8, 14], [11, 13], [12, 14], [7, 15], [8, 16],
    [11, 15], [12, 16], [11, 23], [12, 24], [25, 27], [27, 29],
    [25, 29], [26, 28], [28, 30], [26, 30]
]

def process_frame(cap, pose):
    ret, frame = cap.read()
    if not ret:
        return None, None, None, None
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    
    return frame, frame_rgb, landmarks, results

def calculate_euclidean_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2 + (landmark1.z - landmark2.z)**2)

def calculate_angular_difference(landmark1, landmark2, landmark3, landmark4):
    vector1 = np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])
    vector2 = np.array([landmark4.x - landmark3.x, landmark4.y - landmark3.y, landmark4.z - landmark3.z])
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_accuracy(differences):
    distances = [d[0] for d in differences]
    max_distance = max(distances) if distances else 1
    accuracy = 100 - (sum(distances) / len(distances) / max_distance * 100) if distances else 0
    return accuracy

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    cap1 = cv.VideoCapture('v1.mp4')
    file_location = f"temp_video.mp4"
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    cap2 = cv.VideoCapture(file_location)
    if not cap1.isOpened() or not cap2.isOpened():
        return {"error": "Could not open one or both video files."}
    
    fps1 = cap1.get(cv.CAP_PROP_FPS)
    fps2 = cap2.get(cv.CAP_PROP_FPS)
    playback_speed = 0.75
    delay1 = int((1 / fps1) * 1000 / playback_speed)
    delay2 = int((1 / fps2) * 1000 / playback_speed)
    
    max_y_value1 = -1
    max_y_frame_time1 = -1
    max_y_value2 = -1
    max_y_frame_time2 = -1
    synced = False
    frame_count = 0
    
    euclidean_distances = []
    angular_differences = []
    accuracies = []

    # Video writer for the processed video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('processed_video.mp4', fourcc, fps2, (int(cap2.get(3)), int(cap2.get(4))))

    try:
        while cap1.isOpened() and cap2.isOpened():
            frame1, frame_rgb1, landmarks1, results1 = process_frame(cap1, pose)
            frame2, frame_rgb2, landmarks2, results2 = process_frame(cap2, pose)
            
            if frame1 is not None and frame2 is not None:
                if not synced:
                    if landmarks1 and landmarks2:
                        wrist1 = landmarks1[16]
                        shoulder1 = landmarks1[12]
                        wrist2 = landmarks2[16]
                        shoulder2 = landmarks2[12]
                        
                        if wrist1.x > shoulder1.x:
                            if wrist1.y > max_y_value1:
                                max_y_value1 = wrist1.y
                                max_y_frame_time1 = cap1.get(cv.CAP_PROP_POS_MSEC)
                            
                        if wrist2.x > shoulder2.x:
                            if wrist2.y > max_y_value2:
                                max_y_value2 = wrist2.y
                                max_y_frame_time2 = cap2.get(cv.CAP_PROP_POS_MSEC)
                        
                        if max_y_frame_time1 != -1 and max_y_frame_time2 != -1:
                            synced = True
                            cap1.set(cv.CAP_PROP_POS_MSEC, max_y_frame_time1)
                            cap2.set(cv.CAP_PROP_POS_MSEC, max_y_frame_time2)
                else:
                    differences = []
                    if landmarks1 and landmarks2:
                        for pair in landmark_pairs:
                            idx1, idx2 = pair
                            try:
                                if landmarks1[idx1] and landmarks2[idx1] and landmarks1[idx2] and landmarks2[idx2]:
                                    landmark1_1 = landmarks1[idx1]
                                    landmark1_2 = landmarks1[idx2]
                                    landmark2_1 = landmarks2[idx1]
                                    landmark2_2 = landmarks2[idx2]
                                    
                                    euclidean_distance = calculate_euclidean_distance(landmark1_1, landmark2_1)
                                    angular_difference = calculate_angular_difference(landmark1_1, landmark1_2, landmark2_1, landmark2_2)
                                    
                                    differences.append((euclidean_distance, angular_difference))
                            except Exception as e:
                                continue
                        
                        if differences:
                            accuracy = calculate_accuracy(differences)
                            accuracies.append(accuracy)
                        
                        euclidean_distances.append([d[0] for d in differences])
                        angular_differences.append([d[1] for d in differences])
                        
                        if results1.pose_landmarks:
                            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        if results2.pose_landmarks:
                            mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        out.write(frame2)
                        cv.imshow('Pose Detection v1', frame1)
                        cv.imshow('Pose Detection v2', frame2)

                    cv.waitKey(max(delay1, delay2))

            else:
                break
            
            frame_count += 1

    finally:
        cv.destroyAllWindows()
        cap1.release()
        cap2.release()
        out.release()

    time_steps = range(len(euclidean_distances))
    plt.figure(figsize=(12, 6))
    for i, pair in enumerate(landmark_pairs):
        distances = [frame[i] for frame in euclidean_distances]
        plt.plot(time_steps, distances, label=f'Pair {pair}')
    plt.title('Euclidean Distances Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.savefig('euclidean_distances.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, pair in enumerate(landmark_pairs):
        angles = [frame[i] for frame in angular_differences]
        plt.plot(time_steps, angles, label=f'Pair {pair}')
    plt.title('Angular Differences Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angular Difference (degrees)')
    plt.legend()
    plt.savefig('angular_differences.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, accuracies, label='Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.close()

    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    return {
        "average_accuracy": average_accuracy,
        "processed_video": "processed_video.mp4",
        "accuracy_graph": "accuracy.png"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
