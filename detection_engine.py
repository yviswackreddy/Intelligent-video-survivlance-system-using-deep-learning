import tensorflow as tf
import numpy as np

class DetectionEngine:
    def __init__(self, model_paths):
        self.models = {}
        for task, path in model_paths.items():
            self.models[task] = tf.keras.models.load_model(path)
    
    def detect(self, frame):
        detections = {}
        for task, model in self.models.items():
            pred = model.predict(frame, verbose=0)[0][0]
            detections[task] = float(pred)
        return detections