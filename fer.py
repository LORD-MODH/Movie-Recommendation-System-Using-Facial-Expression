import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

st.title("Webcam Live Feed with Facial Expression Analysis")

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.model = DeepFace

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            result = self.model.analyze(img_rgb, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            cv2.putText(img, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except ValueError as e:
            cv2.putText(img, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return img

webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
