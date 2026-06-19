import os
import cv2
import numpy as np
import streamlit as st
import dlib
import urllib.request
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# 1. Setup Predictor Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

@st.cache_resource
def load_dlib():
    detector = dlib.get_frontal_face_detector()
    # Auto-download model weights if missing
    if not os.path.exists(PREDICTOR_PATH):
        url = "https://github.com/spmallick/PyImageConf2018/raw/master/shape_predictor_68_face_landmarks.dat"
        urllib.request.urlretrieve(url, PREDICTOR_PATH)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return detector, predictor

detector, predictor = load_dlib()

# 2. EAR Calculation Logic
def calculate_ear(eye_pts):
    # Euclidean distance between landmarks
    v1 = np.linalg.norm(np.array([eye_pts[1].x, eye_pts[1].y]) - np.array([eye_pts[5].x, eye_pts[5].y]))
    v2 = np.linalg.norm(np.array([eye_pts[2].x, eye_pts[2].y]) - np.array([eye_pts[4].x, eye_pts[4].y]))
    h = np.linalg.norm(np.array([eye_pts[0].x, eye_pts[0].y]) - np.array([eye_pts[3].x, eye_pts[3].y]))
    return (v1 + v2) / (2.0 * h)

# 3. Video Processing Engine
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

            # Logic: If EAR < 0.23, count frames
            if ear < 0.23:
                self.drowsy_frames += 1
                color = (0, 0, 255) # Red box
                if self.drowsy_frames >= 15:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                self.drowsy_frames = 0
                color = (0, 255, 0) # Green box

            # Draw boxes around eyes
            for eye in [left_eye, right_eye]:
                pts = np.array([(pt.x, pt.y) for pt in eye])
                x, y, w, h = cv2.boundingRect(pts)
                cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), color, 2)
                
        return VideoFrame.from_ndarray(img, format="bgr24")

# 4. Streamlit App Layout
st.title("🚗 Geometric Drowsiness Detection")
st.write("Detecting sleepiness using Eye Aspect Ratio (EAR) calculations.")
# Replace your existing webrtc_streamer block with this:
rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    }