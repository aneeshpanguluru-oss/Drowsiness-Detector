import os
import cv2
import numpy as np
import streamlit as st
import dlib
import tensorflow as tf
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL_PATH = os.path.join(BASE_DIR, "best_drowsiness_model.h5")

@st.cache_resource
def load_resources():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH) if os.path.exists(PREDICTOR_PATH) else None
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = None
    return detector, predictor, model

detector, predictor, mobilenet_model = load_resources()

# 2. Video Processing Engine processing eyes individually
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if mobilenet_model is None:
            cv2.putText(img, "Error: best_drowsiness_model.h5 missing!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return VideoFrame.from_ndarray(img, format="bgr24")

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Isolate individual eye landmark mappings
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            eye_probabilities = []
            
            # Process left and right eyes completely separately
            for eye_pts in [left_eye_pts, right_eye_pts]:
                x, y, w, h = cv2.boundingRect(eye_pts)
                
                # Dynamic square padding around each individual eye
                pad = 12
                y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
                
                eye_crop = img[y1:y2, x1:x2]
                
                if eye_crop.size > 0:
                    # Convert BGR camera frame to RGB channel order
                    eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                    
                    # Target input dimensions matching your Sequential training layer
                    eye_resized = cv2.resize(eye_rgb, (224, 224))
                    
                    # Normalize pixels to [0, 1] matching your train_dataset rescaling configuration
                    eye_normalized = eye_resized / 255.0
                    eye_batch = np.expand_dims(eye_normalized, axis=0)
                    
                    # Run classification inference
                    pred = mobilenet_model.predict(eye_batch, verbose=0)[0][0]
                    eye_probabilities.append(pred)
                    
                    # Draw visual reference boxes over each eye
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            if eye_probabilities:
                # Average the raw output of both eyes (1 = Open, 0 = Closed)
                raw_open_prediction = np.mean(eye_probabilities)
                
                # Invert the value mathematically to get actual Closed Probability
                closed_prediction = 1.0 - raw_open_prediction
                
                # Check consecutive frames against your threshold parameter
                if closed_prediction > 0.5:  
                    self.drowsy_frames += 1
                    if self.drowsy_frames >= 5:
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    self.drowsy_frames = 0
                
        return VideoFrame.from_ndarray(img, format="bgr24")

# 3. Streamlit Interface Configuration
st.title("Geometric Drowsiness Detection")
st.write("Detecting sleepiness using MobileNetV2 Deep Learning predictions.")

webrtc_streamer(
    key="drowsiness-mobilenet",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuraztion={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)