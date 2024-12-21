import pickle
import time

import cv2
import imutils
import streamlit as st
from streamlit_image_select import image_select
from ultralytics import YOLO

from multi_precomp import compute_cosine_similarity, normalize_keypoints

st.set_page_config(
    page_title="Match Your Trainer",
    layout="wide",
    page_icon=":weight_lifter:",
    initial_sidebar_state="collapsed",
)
st.html("""<style>
        .st-emotion-cache-1ibsh2c   /* Align the whole document to center */
        {
            text-align:center;
            # background-color: blue; 
        }
        .st-emotion-cache-1kw2d9g   /* Remove the fullscreen button near videos */
        {
            display: none;
        }
        .st-emotion-cache-xhkv9f    /* Centers the 2 video boxes within their columns */
        {
            margin-left: auto;
            margin-right: auto;
        }
        img {   /* Round the corners of the videos */
            border-radius: 8px;
        }
        </style>""")

# Main container for dynamic content
placeholder = st.empty()

if "similarity_list" not in st.session_state:
    st.session_state.similarity_list = []


def stop():
    st.session_state.completed = True


@st.dialog("Similarity Chart")
def show_chart():
    if not st.session_state.similarity_list:
        st.warning("No data to display")

    else:
        st.write("Here's how well you perfomed:")
        st.line_chart(st.session_state.similarity_list)

    if st.button("Close"):
        st.session_state.similarity_list = []
        st.session_state.completed = False
        st.rerun()


def match_trainer(video_path, data_path):
    placeholder.markdown("")  # Clear the current content
    time.sleep(0.1)

    with placeholder.container(border=True):
        col1, col2 = st.columns((1, 1), vertical_alignment="center")
        trainer_col = col1.empty()
        user_col = col2.empty()

        with open(data_path, "rb") as f:
            points1 = pickle.load(f)

        index = 0
        no_person = 0
        no_camera = 0
        model = YOLO("yolo11n-pose.pt")

        cap1 = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(0)

        st.markdown("Similarity")
        score = st.empty()
        st.markdown("")

        st.button("Stop", on_click=stop)

        while cap1.isOpened() or cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if ret1 and ret2:
                fn = cap1.get(cv2.CAP_PROP_POS_FRAMES)

                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                if fn % 5 == 0:  # Process every 5th frame
                    results2 = model(frame2, conf=0.3, imgsz=320, max_det=1)
                    frame2 = results2[0].plot()

                    for r in results2:
                        points2 = r.keypoints.xy.numpy()

                        if points2.size:  # Check if keypoints are detected
                            points2 = normalize_keypoints(points2[0], anchor_idx1=5, anchor_idx2=6)
                            similarity = compute_cosine_similarity(points1[index], points2)
                            if similarity < 0:
                                similarity = 0

                            similarity = round(similarity * 100, 2)
                            st.session_state.similarity_list.append(similarity)

                            score.subheader(str(similarity) + " %", anchor=False)
                            index += 1

                        else:
                            st.session_state.similarity_list.append(None)
                            score.subheader("0 %", anchor=False)
                            no_person += 1
                            if no_person == 30:  # If no person is detected for 30 frames i.e 5 seconds
                                st.toast("No person detected", icon="⚠️")
                                no_person = 0

                frame1 = imutils.resize(frame1, width=960)
                frame2 = imutils.resize(frame2, height=374)

                # Update placeholders with the video frames
                with trainer_col.container():
                    st.markdown("Trainer")
                    st.image(frame1, channels="RGB")

                with user_col.container():
                    st.markdown("User")
                    st.image(frame2, channels="RGB")

            elif not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                index = 0

            else:
                if no_camera == 5:
                    st.rerun()
                else:
                    st.toast("Camera is not detected", icon="⚠️")
                    no_camera += 1
                    time.sleep(5)


def home():
    with placeholder.container(border=True):
        st.title("Match Your Trainer", anchor=False)
        st.markdown("")
        st.markdown("Select an exercise to get started")

        img = image_select(
            "",
            [
                "./photos/icon_jumping_jack.jpg",
                "./photos/icon_bicepcurl.jpg",
                "./photos/icon_jumping_jack.jpg",
                "./photos/icon_jumping_jack.jpg",
            ],
            captions=[
                "Jumping Jacks",
                "Bicep Curl",
                "Jumping Jacks",
                "Jumping Jacks",
            ],
            use_container_width=False,
            return_value="index",
        )

        st.markdown("")

        # Button action
        if st.button("Start"):  # change this thing
            if img == 0:  # If the first image is selected
                match_trainer("./videos/jump1.mp4", "./export/data_jumping_jacks.pkl")
            elif img == 1:  # NOT WORKING, ONLY 1 ARGUMENT PASSED, because no data for bicep curl
                match_trainer("./videos/jump2.mp4")


if "completed" not in st.session_state or not st.session_state.completed:
    st.session_state.completed = False
    home()

if st.session_state.completed:
    show_chart()
