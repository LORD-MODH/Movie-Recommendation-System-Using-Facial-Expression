import streamlit as st
from streamlit_webrtc import webrtc_streamer
from deepface import DeepFace
import av
import cv2

st.title("Real-time Emotion Detection with Confidence Levels")


def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        emotions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        dominant_emotion = emotions[0]['dominant_emotion']
        emotion_confidences = emotions[0]['emotion']
        face_confidence = emotions[0]['face_confidence']

        cv2.putText(img, f"Dominant: {dominant_emotion} ({emotion_confidences[dominant_emotion]:.2f}%)",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        y_offset = 100
        for emotion, confidence in emotion_confidences.items():
            cv2.putText(img, f"{emotion}: {confidence:.2f}%", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30

    except Exception as e:
        print("Error in processing frame:", str(e))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="emotion-analysis", video_frame_callback=process_frame)
