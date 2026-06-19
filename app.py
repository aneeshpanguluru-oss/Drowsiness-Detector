import os
import cv2
import numpy as np
import streamlit as st
import dlib
from av import VideoFrame
from streamlit_webrtc import组件 webrtc_streamer, VideoProcessorBase, WebRtcMode

# Cross-platform sound configuration for Windows vs Linux Cloud
try:
    import winsound
except ImportError:
    winsound = None  # Running on Linux Cloud (Streamlit)

# Set landmark path relative to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

@st.cache_resource
def load_dlib():
    detector = dlib.get_frontal_face_detector()
    # Check if weights file exists before trying to load it
    if os.path.exists(PREDICTOR_PATH):
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    else:
        predictor = None
    return detector, predictor

detector, predictor = load_dlib()

st.title("🚗 Geometric Drowsiness Detection System")
st.write("This application tracks your eye aspect ratio natively to detect sleepiness.")

def calculate_ear(eye_pts):
    # Compute the distances between the two sets of vertical eye landmarks
    p2_minus_p6 = np.linalg.norm(np.array([eye_pts[1].x, eye_pts[1].y]) - np.array([eye_pts[5].x, eye_pts[5].y]))
    p3_minus_p5 = np.linalg.norm(np.array([eye_pts[2].x, eye_pts[2].y]) - np.array([eye_pts[4].x, eye_pts[4].y]))
    # Compute the distance between the horizontal eye landmarks
    p1_minus_p4 = np.linalg.norm(np.array([eye_pts[0].x, eye_pts[0].y]) - np.array([eye_pts[3].x, eye_pts[3].y]))
    
    # Calculate the Eye Aspect Ratio (EAR)
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0
        self.is_drowsy = False

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if predictor is None:
            cv2.putText(img, "Error: shape_predictor file missing!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return VideoFrame.from_ndarray(img, format="bgr24")

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Extract left and right eye landmarks mapping indexes
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            # Average Eye Aspect Ratio
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eye bounding lines for reference visual
            for n in range(36, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            # Check threshold logic
            if ear < 0.25:
                self.drowsy_frames += 1
                if self.drowsy_frames >= 20:
                    self.is_drowsy = True
                    cv2.putText(img, "⚠️ DROWSINESS ALERT! ⚠️", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                self.drowsy_frames = 0
                self.is_drowsy = False

            # Display real-time structural geometry mathematical ratio on stream
            cv2.putText(img, f"EAR: {ear:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# Initialize real-time components tracking streaming interface via WebRTC
ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Trigger global interface alert audio rendering safely outside the video threading worker
if ctx.video_processor and ctx.video_processor.is_drowsy:
    st.error("⚠️ EMERGENCY WARNING: Driver drowsiness detected! Please pull over safely.")
    if winsound:
        # Motherboard system speaker alert tone for native Windows local environment running
        winsound.Beep(1200, 400)
    else:
        # Cloud safe media runtime playback component execution stream injection
        st.audio("https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg", autoplay=True)