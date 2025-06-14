import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path='emotion_model.h5'):
        # Load pre-trained emotion classification model
        self.model = load_model(model_path)
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Emotion labels
        self.emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']
        # Image size expected by the model
        self.img_size = 48

    def preprocess_image(self, img):
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        img = cv2.resize(img, (self.img_size, self.img_size))
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        # Add batch and channel dimensions
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
        return img

    def predict_emotion(self, img):
        # Preprocess image
        processed_img = self.preprocess_image(img)
        # Predict emotion
        prediction = self.model.predict(processed_img, verbose=0)
        # Get emotion with highest probability
        emotion_idx = np.argmax(prediction[0])
        return self.emotions[emotion_idx], prediction[0][emotion_idx]

    def process_frame(self, frame):
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y+h, x:x+w]
            # Predict emotion
            emotion, confidence = self.predict_emotion(face)
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add emotion label and confidence
            label = f'{emotion}: {confidence:.2f}'
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame