import cv2
import numpy as np

def preprocess_frame(frame, target_size=(224, 224)):
    # Resize and normalize frame
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def preprocess_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    return np.array(frames)