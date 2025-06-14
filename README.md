# Intelligent Video Surveillance System

## Overview
This project implements a deep learning-based video surveillance system for detecting fire, mask absence, weapons, and fights in real-time video streams. It uses TensorFlow/Keras for model training, OpenCV for video processing, and Flask for a web interface.

## Project Structure
- models/: Contains scripts for training models and saved model files.
- utils/: Utility scripts for preprocessing and alert systems.
- surveillance/: Core video processing and detection logic.
- web_app/: Flask web application for user interface.
- data/: Placeholder for datasets and sample videos.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare datasets in `data/datasets/` with subfolders `fire/`, `mask/`, `weapon/`, `fight/`, each containing positive and negative samples.
3. Train models:
   ```bash
   python models/train_models.py
   ```
4. Run the web application:
   ```bash
   python web_app/app.py
   ```
5. Access the interface at `http://localhost:5000`.

## Notes
- Models are trained using a CNN architecture with data augmentation.
- The alert system requires an SMTP server configuration.
- Video streaming uses OpenCV's default webcam (index 0) or specify a video file.
