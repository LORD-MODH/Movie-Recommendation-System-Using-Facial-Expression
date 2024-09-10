import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.title("Webcam Live Feed with Facial Expression Analysis")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = DeepFace

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            result = self.model.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            cv2.putText(img, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except ValueError:
            cv2.putText(img, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False} 
)

