import pickle

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.video import Video

# from multi_precomp import compute_cosine_similarity, normalize_keypoints


class MotionMatchApp(App):
    def build(self):
        # Main layout
        main_layout = BoxLayout(orientation="vertical")

        # Navigation Bar
        nav_bar = BoxLayout(size_hint_y=0.1, padding=10)
        heading = Label(text="Motion Match", font_size=24, bold=True, size_hint_x=0.8)
        nav_bar.add_widget(heading)
        main_layout.add_widget(nav_bar)

        # Body Layout (Two Boxes and Results Box)
        body_layout = BoxLayout(orientation="horizontal", padding=10, spacing=10)

        # Left Box with Video (matching dimensions with the right box)
        left_box = BoxLayout(size_hint=(0.5, 1), padding=10)
        video = Video(source="./videos/jump1.mp4", state="play", options={"eos": "loop"})  # Replace 'jump1.mp4' with your video file
        left_box.add_widget(video)

        # Right Camera Box
        right_box = BoxLayout(size_hint=(0.5, 1), padding=10)
        self.camera_feed = Image(size_hint=(1, 1))  # Kivy Image widget to display OpenCV feed
        right_box.add_widget(self.camera_feed)

        # Add Left and Right Boxes to the Body Layout
        body_layout.add_widget(left_box)
        body_layout.add_widget(right_box)

        # Results Box Below the Two Boxes
        results_box = BoxLayout(size_hint_y=0.2, padding=10)
        results_label = Label(text="Results will be displayed here", font_size=25)
        results_box.add_widget(results_label)

        # Add body layout and results box to the main layout
        main_layout.add_widget(body_layout)
        main_layout.add_widget(results_box)

        # Start OpenCV camera
        self.capture = cv2.VideoCapture(0)  # Open the default camera
        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            return

        # Schedule the update method for the camera feed
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS update

        return main_layout

    def update(self, dt):
        # Capture frame from OpenCV
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame")
            return

        # Convert frame from BGR to RGB (Kivy expects RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to texture for Kivy
        texture = self.create_texture_from_frame(frame)

        # Update the camera feed with the new texture
        self.camera_feed.texture = texture

    def create_texture_from_frame(self, frame):
        # Get the height and width of the frame
        height, width, _ = frame.shape

        # Create texture for the frame
        texture = Image(size=(width, height)).texture
        texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        return texture

    def on_stop(self):
        # Release the camera when the app is stopped
        if self.capture.isOpened():
            self.capture.release()


# Run the application
MotionMatchApp().run()
