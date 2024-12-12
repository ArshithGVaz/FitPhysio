import pickle
import sys

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from ultralytics import YOLO

from multi_precomp import compute_cosine_similarity, normalize_keypoints

similarity_values = []

with open("./export/data.pkl", "rb") as f:
    points1 = pickle.load(f)


class VideoThread(QThread):
    frame_update = pyqtSignal(QImage)

    def run(self):
        self.running = True
        i = 0
        model = YOLO("yolo11n-pose.pt")
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if ret:
                print(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # Convert the frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results2 = model(rgb_frame, conf=0.3, imgsz=320, max_det=1)

                rgb_frame = results2[0].plot()

                for r1 in results2:
                    if r1.keypoints:
                        points2 = r1.keypoints.xy.numpy()
                        points2 = normalize_keypoints(points2[0], anchor_idx1=5, anchor_idx2=6)
                        similarity = compute_cosine_similarity(points1[i], points2)
                        similarity_values.append(similarity)
                        i += 1

                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_update.emit(qt_image)

        cap.release()

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
        left_box = QFrame()
        left_box.setObjectName("leftBox")

        self.right_box = QLabel("Camera Feed")
        self.right_box.setAlignment(Qt.AlignCenter)
        self.right_box.setObjectName("rightBox")

        left_box.setFrameShape(QFrame.StyledPanel)

        content_layout.addWidget(left_box, 1)
        content_layout.addWidget(self.right_box, 1)

        content_container = QWidget()
        content_container.setLayout(content_layout)

        main_layout.addWidget(content_container)

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Start Video Thread
        self.thread = VideoThread()
        self.thread.frame_update.connect(self.update_frame)
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
            background-color: #f0f0f0;
        }

        #rightBox {
            background-color: #000;
            color: white;
            font-size: 16px;
            border: 2px solid #444;
        }
        """

    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.right_box.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
