import cv2
import streamlit as st
from deepface import DeepFace

def facial_analysis():
  st.title("Webcam Live Feed with Facial Expression Analysis")
  run = st.checkbox('Run the Model')
  FRAME_WINDOW = st.image([])
  if run:
      camera = cv2.VideoCapture(0)
      while run:
          _, frame = camera.read()
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          try:
              result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
              st.write(result)
          except ValueError as e:
              st.write("No face detected")
          FRAME_WINDOW.image(frame)
      camera.release()
      st.write('Stopped')
  else:
      st.write("Click 'Run' to start the webcam.")
