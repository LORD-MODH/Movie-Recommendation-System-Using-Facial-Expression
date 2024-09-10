import streamlit as st
from streamlit_webrtc import webrtc_streamer
from deepface import DeepFace
import av
import time

st.title("Real-time Emotion Analysis (WebRTC)")

def process_frame(frame):
    last_analysis_time = 0
    current_time = time.time()
    if current_time - last_analysis_time >= 5:
         try:
            img = frame.to_ndarray(format="bgr24")

            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            cv2.putText(img, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            last_analysis_time = current_time
        except ValueError:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    else:
        return frame  

webrtc_streamer(
    key="example", 
    video_frame_callback=process_frame,  
)
