# Real-Time Geometric Drowsiness Detection System 🚗💤

During the research and prototyping phase, 5 distinct deep learning architectures were benchmarked to find the optimal balance between classification accuracy and inference latency. While MobileNetV2 delivered the best real-time metrics among the neural networks, a pure geometric tracking approach was ultimately deployed to ensure absolute immunity to shadows and extreme lighting variations.

This high-performance real-time drowsiness detection web application is built with **Streamlit** and **WebRTC**. The system tracks facial landmarks dynamically to compute the **Eye Aspect Ratio (EAR)**, bypassing the need for heavy runtime deep learning models in production.

---

## 🚀 Features

- **Pure Geometric Tracking:** Uses dlib's 68-point facial landmark predictor to measure eyelid distance mathematically.
- **Shadow & Light Immunity:** Outperforms pixel-intensity models by tracking physical eye coordinates rather than raw image gradients.
- **Low-Latency Streaming:** Powered by `streamlit-webrtc` for real-time asynchronous browser-side video processing.
- **Instant Native Alarm:** Triggers a high-frequency Windows kernel buzzer (`winsound`) the exact millisecond a micro-sleep threshold is crossed.
- **Zero-Lag Recovery:** The alarm cuts off instantly the moment the user opens their eyes, resetting the frame counter to zero.

---

## 📐 Core Methodology: Eye Aspect Ratio (EAR)

The application tracks facial coordinates to map the eye regions. For each eye, six landmarks are extracted to calculate the Eye Aspect Ratio (EAR) using the following geometric relationship:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
```

- **Baseline EAR:** A value below `0.18` indicates a closed eye.
- **Trigger Window:** If the EAR remains below the baseline for more than **0.5 seconds** (~15 consecutive frames at 30 FPS), a drowsiness alert is instantly triggered.

---

## 📂 Project Directory Structure

```text
Drowsiness-Detector/
├── app.py                      # Core Streamlit WebRTC application script
├── requirements.txt            # Python package dependencies for deployment
├── .gitignore                  # Excludes local environments and document clutter
├── README.md                   # Project documentation and setup guide
└── notebook/
    └── codeforproject.ipynb    # Development Jupyter Notebook scratchpad
```

---

## ⚙️ System Threshold Parameters

Through testing and optimization, the configuration thresholds have been tuned to prevent false alarms from natural blinking while maximizing responsiveness:

| Parameter | Value | Notes |
|---|---|---|
| Eye Padding (width) | 5 px | Snug to ignore outer nose/face shadows |
| Eye Padding (height) | 10 px | Tall vertical bounds for accurate capture |
| EAR Baseline Threshold | `< 0.18` | Accommodates relaxed, natural open-eye profiles |
| Trigger Window | 0.5 s | 15 consecutive frames at ~30 FPS before alarm |

---

## 🛠️ Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AneeshPanguluru/Drowsiness-Detector.git
cd Drowsiness-Detector
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Download the Facial Landmark Weights

The pre-trained facial model (`shape_predictor_68_face_landmarks.dat`) is roughly **100 MB** and is excluded from Git tracking to maintain a lightweight repository.

1. Download the weights from the official source:
   [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat)
2. Move the downloaded file directly into your root `Drowsiness-Detector/` directory.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📦 Dependencies

The core libraries listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `streamlit-webrtc` | Real-time browser video streaming |
| `dlib-bin` | 68-point facial landmark predictor |
| `opencv-contrib-python` | Image processing and frame capture |
| `numpy` | Numerical computation for EAR geometry |
| `av` | Audio/video codec support for WebRTC |
