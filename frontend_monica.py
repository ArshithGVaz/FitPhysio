import pickle
import sys

import cv2
import imutils
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from ultralytics import YOLO

from multi_precomp import compute_cosine_similarity, normalize_keypoints

similarity_values = []

with open("./export/data.pkl", "rb") as f:
    points1 = pickle.load(f)


class VideoThread(QThread):
    frame_update_left = pyqtSignal(QImage)  # Signal for left box
    frame_update_right = pyqtSignal(QImage)  # Signal for right box

    def run(self):
        self.running = True
        i = 0
        model = YOLO("yolo11n-pose.pt")

        # Capture video from two cameras (adjust indices if necessary)
        cap_left = cv2.VideoCapture("./videos/jump1.mp4")  # Left camera
        cap_right = cv2.VideoCapture(0)  # Right camera

        while self.running:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if ret_left and ret_right:
                # Process left camera feed (if needed)
                rgb_frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                rgb_frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

                results_right = model(rgb_frame_right, conf=0.3, imgsz=320, max_det=1)
                rgb_frame_right = results_right[0].plot()

                rgb_frame_left = imutils.resize(rgb_frame_left, width=960)

                h_left, w_left, ch_left = rgb_frame_left.shape
                h_right, w_right, ch_right = rgb_frame_right.shape

                bytes_per_line_left = ch_left * w_left
                bytes_per_line_right = ch_right * w_right

                qt_image_left = QImage(rgb_frame_left.data, w_left, h_left, bytes_per_line_left, QImage.Format_RGB888)
                qt_image_right = QImage(rgb_frame_right.data, w_right, h_right, bytes_per_line_right, QImage.Format_RGB888)

                self.frame_update_left.emit(qt_image_left)  # Emit the signal for left feed
                self.frame_update_right.emit(qt_image_right)  # Emit the signal for right feed

        # Release the video captures when the thread is stopped
        cap_left.release()
        cap_right.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Motion Match GUI")
        self.setGeometry(100, 100, 800, 600)

        # Main Layout
        main_layout = QVBoxLayout()

        # Navigation Bar
        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(0, 0, 0, 0)
        nav_bar.setSpacing(10)

        nav_home = QPushButton("Home")
        nav_about = QPushButton("About")
        nav_contact = QPushButton("Contact")

        for btn in [nav_home, nav_about, nav_contact]:
            btn.setObjectName("navButton")
            nav_bar.addWidget(btn)

        nav_container = QWidget()
        nav_container.setLayout(nav_bar)
        nav_container.setObjectName("navBar")

        main_layout.addWidget(nav_container)

        # Heading
        heading = QLabel("Motion Match")
        heading.setAlignment(Qt.AlignCenter)
        heading.setObjectName("heading")
        main_layout.addWidget(heading)

        # Content Area
        content_layout = QHBoxLayout()

        # Left Box for Video (Camera Feed 1)
        self.left_box = QLabel("Left Camera Feed")
        self.left_box.setAlignment(Qt.AlignCenter)
        self.left_box.setObjectName("leftBox")

        content_layout.addWidget(self.left_box, 1)

        # Right Box for Video (Camera Feed 2)
        self.right_box = QLabel("Right Camera Feed")
        self.right_box.setAlignment(Qt.AlignCenter)
        self.right_box.setObjectName("rightBox")

        content_layout.addWidget(self.right_box, 1)

        content_container = QWidget()
        content_container.setLayout(content_layout)

        main_layout.addWidget(content_container)

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Start Video Thread to handle two camera feeds
        self.thread = VideoThread()
        self.thread.frame_update_left.connect(self.update_left_frame)
        self.thread.frame_update_right.connect(self.update_right_frame)
        self.thread.start()

        # Apply Styles
        self.setStyleSheet(self.load_styles())

    def load_styles(self):
        return """
        #navBar {
            background-color: #333;
            padding: 8px;
        }

        #navButton {
            color: white;
            background-color: #444;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }

        #navButton:hover {
            background-color: #555;
        }

        #heading {
            font-size: 24px;
            font-weight: bold;
            color: #222;
            margin: 20px 0;
        }

        #leftBox {
            background-color: #000;
            color: white;
            font-size: 16px;
            border: 2px solid #444;
        }

        #rightBox {
            background-color: #000;
            color: white;
            font-size: 16px;
            border: 2px solid #444;
        }
        """

    def update_left_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.left_box.setPixmap(pixmap)

    def update_right_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.right_box.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.stop()  # Stop the video thread
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
