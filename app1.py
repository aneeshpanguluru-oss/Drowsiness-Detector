import os
import cv2
import numpy as np
import streamlit as st
import dlib
import tensorflow as tf  # Used strictly for the lightweight TFLite Interpreter
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL_PATH = os.path.join(BASE_DIR, "best_drowsiness_model.tflite")

@st.cache_resource
def load_resources():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH) if os.path.exists(PREDICTOR_PATH) else None
    
    # Load the lightweight TFLite runtime instead of heavy Keras models
    interpreter = None
    input_details = None
    output_details = None
    
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
    return detector, predictor, interpreter, input_details, output_details

detector, predictor, interpreter, input_details, output_details = load_resources()

# 2. Video Processing Engine
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_frames = 0
        self.frame_counter = 0
        self.last_prediction = 0.0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1
        
        if interpreter is None:
            cv2.putText(img, "Error: TFLite model missing!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return VideoFrame.from_ndarray(img, format="bgr24")

        # Process every 2nd frame to optimize cloud performance smoothly
        if self.frame_counter % 2 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                landmarks = predictor(gray, face)
                
                left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                eye_probabilities = []
                
                for eye_pts in [left_eye_pts, right_eye_pts]:
                    x, y, w, h = cv2.boundingRect(eye_pts)
                    pad = 12
                    y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
                    x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
                    
                    eye_crop = img[y1:y2, x1:x2]
                    
                    if eye_crop.size > 0:
                        eye_rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
                        eye_resized = cv2.resize(eye_rgb, (224, 224))
                        eye_normalized = (eye_resized / 255.0).astype(np.float32)
                        eye_batch = np.expand_dims(eye_normalized, axis=0)
                        
                        # TFLite Tensor Inference
                        interpreter.set_tensor(input_details[0]['index'], eye_batch)
                        interpreter.invoke()
                        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
                        eye_probabilities.append(pred)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                if eye_probabilities:
                    raw_open_prediction = np.mean(eye_probabilities)
                    self.last_prediction = 1.0 - raw_open_prediction

        # Snappy drowsiness alerting threshold logic
        if self.last_prediction > 0.5:  
            self.drowsy_frames += 1
            if self.drowsy_frames >= 4:  
                cv2.putText(img, "DROWSINESS ALERT!", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            self.drowsy_frames = 0
                
        return VideoFrame.from_ndarray(img, format="bgr24")

# 3. Layout Interface
st.title("Geometric Drowsiness Detection")
st.write("Detecting sleepiness using MobileNetV2 Deep Learning predictions.")

webrtc_streamer(
    key="drowsiness-mobilenet-tflite",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)