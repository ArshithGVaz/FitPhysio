from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.video import Video


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
        video = Video(source="./videos/jump1.mp4", state="play", options={"eos": "loop"})  # Replace 'v2.mp4' with your video file
        left_box.add_widget(video)

        # Right Camera Box
        right_box = BoxLayout(size_hint=(0.5, 1), padding=10)
        camera = Camera(resolution=(640, 480), play=True)  # Enable camera
        right_box.add_widget(camera)

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

        return main_layout


# Run the application
MotionMatchApp().run()
