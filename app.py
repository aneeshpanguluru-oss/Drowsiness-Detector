import os
import cv2
import numpy as np
import streamlit as st
import dlib
import urllib.request
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Set landmark path relative to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

@st.cache_resource
def load_dlib():
    detector = dlib.get_frontal_face_detector()
    
    # Auto-download model weights if missing from server container
    if not os.path.exists(PREDICTOR_PATH):
        with st.spinner("Downloading facial landmark predictor model weights (~97MB)... Please wait."):
            try:
                url = "https://github.com/spmallick/PyImageConf2018/raw/master/shape_predictor_68_face_landmarks.dat"
                urllib.request.urlretrieve(url, PREDICTOR_PATH)
            except Exception as e:
                st.error(f"Failed to auto-download model weights: {e}")
                return detector, None

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return detector, predictor

detector, predictor = load_dlib()

st.title("🚗 Geometric Drowsiness Detection System")
st.write("This application tracks your eye aspect ratio natively to detect sleepiness.")

# Standard Euclidean distance formula calculation using normal square method
def calculate_ear(eye_pts):
    # Vertical distances using standard square difference and square root (Euclidean Distance)
    v1 = np.sqrt((eye_pts[1].x - eye_pts[5].x)**2 + (eye_pts[1].y - eye_pts[5].y)**2)
    v2 = np.sqrt((eye_pts[2].x - eye_pts[4].x)**2 + (eye_pts[2].y - eye_pts[4].y)**2)
    
    # Horizontal distance
    h = np.sqrt((eye_pts[0].x - eye_pts[3].x)**2 + (eye_pts[0].y - eye_pts[3].y)**2)
    
    # Calculate Eye Aspect Ratio (EAR)
    ear = (v1 + v2) / (2.0 * h)
    return ear

def draw_eye_box(img, eye_pts, color=(0, 255, 0), thickness=2):
    x_coords = [pt.x for pt in eye_pts]
    y_coords = [pt.y for pt in eye_pts]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # ADJUST THESE VALUE PADDING PARAMETERS RIGHT HERE:
    padding_width = 15   # Increase this to make the boxes wider on the sides
    padding_height = 6   # Keep this lower so the box doesn't go too high into your eyebrows
    
    # Map the adjusted coordinates into the rectangle drawer
    cv2.rectangle(img, 
                  (x_min - padding_width, y_min - padding_height), 
                  (x_max + padding_width, y_max + padding_height), 
                  color, 
                  thickness)
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if predictor is None:
            cv2.putText(img, "Error: Weights missing!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return VideoFrame.from_ndarray(img, format="bgr24")

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Map eye coordinate points index ranges
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            ear = (left_ear + right_ear) / 2.0
            
            # Check threshold logic to decide bounding box colors
            if ear < 0.23:
                self.drowsy_frames += 1
                box_color = (0, 0, 255)  # Red boxes when drowsy
                
                if self.drowsy_frames >= 15:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                self.drowsy_frames = 0
                box_color = (0, 255, 0)  # Green boxes when awake

            # Draw the square bounding boxes over the eyes instead of point dots
            draw_eye_box(img, left_eye, color=box_color, thickness=2)
            draw_eye_box(img, right_eye, color=box_color, thickness=2)

            # Display real-time numeric EAR tracking ratio
            cv2.putText(img, f"EAR: {ear:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit real-time network signaling container initialization mapping
# Streamlit real-time network signaling container initialization mapping
webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    # Updated firewall-immune network configuration mapping
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]}
        ]
    }
)