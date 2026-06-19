import os
import cv2
import numpy as np
import streamlit as st
import dlib
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
# Windows native sound library 
import winsound  
# Set landmark path relative to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
@st.cache_resource
def load_dlib():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return detector, predictor
detector, predictor = load_dlib()
st.title("Geometric Drowsiness Detection System")
st.write("This application tracks your eye aspect ratio natively to detect sleepiness.")
def calculate_ear(eye_pts):
    # Compute the distances between the two sets of vertical eye landmarks
    p2_minus_p6 = np.linalg.norm(np.array([eye_pts[1].x, eye_pts[1].y]) - np.array([eye_pts[5].x, eye_pts[5].y]))
    p3_minus_p5 = np.linalg.norm(np.array([eye_pts[2].x, eye_pts[2].y]) - np.array([eye_pts[4].x, eye_pts[4].y]))
    # Compute the distance between the horizontal eye landmarks
    p1_minus_p4 = np.linalg.norm(np.array([eye_pts[0].x, eye_pts[0].y]) - np.array([eye_pts[3].x, eye_pts[3].y]))
    # Eye Aspect Ratio formula
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        any_eye_closed = False
        for face in faces:
            # Draw face bounding frame
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 120, 255), 2)
            shape = predictor(gray, face)
            left_eye_pts = [shape.part(i) for i in range(36, 42)]
            right_eye_pts = [shape.part(i) for i in range(42, 48)]
            for eye in [left_eye_pts, right_eye_pts]:
                x_coords = [p.x for p in eye]
                y_coords = [p.y for p in eye]
                # Dynamic visual box sizing around the landmarks
                x1, x2 = max(0, min(x_coords) - 6), min(w, max(x_coords) + 6)
                y1, y2 = max(0, min(y_coords) - 8), min(h, max(y_coords) + 8)
                # Calculate the exact geometric opening ratio of this eye
                ear = calculate_ear(eye)
                # LOWERED THRESHOLD: 0.18 perfectly matches your relaxed open-eye baseline
                if ear < 0.18:  
                    any_eye_closed = True
                    color = (0, 0, 255) # Red box
                else:
                    color = (0, 255, 0) # Green box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"EAR: {ear:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if any_eye_closed:
            self.counter += 1
        else:
            self.counter = 0            
        # 0.5-second filter window (15 consecutive frames)
        if self.counter >= 15:
            cv2.putText(img, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            winsound.Beep(2200, 100)
        return VideoFrame.from_ndarray(img, format="bgr24")
webrtc_streamer(key="drowsiness-detection", mode=WebRtcMode.SENDRECV,video_processor_factory=DrowsinessProcessor,media_stream_constraints={"video": True, "audio": False},async_processing=True)