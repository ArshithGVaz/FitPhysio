import pickle
import time
import cv2
import numpy as np
from threading import Thread
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from ultralytics import YOLO

similarity_values = []

with open("data.pkl", "rb") as f:
    points1 = pickle.load(f)

class VideoCapture(BoxLayout):
    def __init__(self, path1, path2, **kwargs):
        super(VideoCapture, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.image1 = Image()
        self.image2 = Image()
        self.add_widget(self.image1)
        self.add_widget(self.image2)

        self.model = YOLO("yolo11n-pose.pt")
        self.cap1 = cv2.VideoCapture(path1)
        self.cap2 = cv2.VideoCapture(path2)
        
        self.update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

    def update(self, dt):
        success1, frame1 = self.cap1.read()
        success2, frame2 = self.cap2.read()

        if success1 and success2:
            results1 = self.model(frame1, conf=0.3, imgsz=160, max_det=1)
            results2 = self.model(frame2, conf=0.3, imgsz=160, max_det=1)

            frame1 = results1[0].plot() if results1 else frame1
            frame2 = results2[0].plot() if results2 else frame2

            # Convert BGR to RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Convert to texture for Kivy
            self.image1.texture = self.convert_frame_to_texture(frame1)
            self.image2.texture = self.convert_frame_to_texture(frame2)

    def convert_frame_to_texture(self, frame):
        # Convert frame to Kivy texture
        texture = Image(texture=frame).texture
        return texture

    def on_stop(self):
        self.cap1.release()
        self.cap2.release()
        Clock.unschedule(self.update)

class VideoApp(App):
    def build(self):
        return VideoCapture("v3.mp4", "jump2.mp4")

if __name__ == '__main__':
    VideoApp().run()
