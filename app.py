import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessor
from deepface import DeepFace
import av
import cv2
import time

st.title("Real-time Emotion Analysis (WebRTC)")

class EmotionProcessor(VideoProcessor):
    def __init__(self):
        self.last_analysis_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        if current_time - self.last_analysis_time >= 5:  # Analyze every 5 seconds
            try:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                cv2.putText(img, dominant_emotion, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                self.last_analysis_time = current_time
            except ValueError:
                pass 
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Create and start the WebRTC component
webrtc_ctx = webrtc_streamer(key="example",
                         video_processor_factory=EmotionProcessor,
                         media_stream_constraints={"video": True, "audio": False},
                         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
