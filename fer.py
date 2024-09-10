import cv2
import streamlit as st
from deepface import DeepFace

def facial_analysis():
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    if run:
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Failed to capture image from webcam. Please check if the webcam is accessible.")
                break  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                st.write(result['emotion'])
            except ValueError:
                st.write("No face detected")
            FRAME_WINDOW.image(frame)
        camera.release()
        st.write("Webcam stopped.")
    else:
        st.write("Click 'Run Webcam' to start.")
