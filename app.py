from flask import Flask, render_template, Response
import cv2
from emotion_detector import EmotionDetector

app = Flask(__name__)

# Initialize emotion detector
detector = EmotionDetector()

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use webcam
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame and get annotated frame with emotions
            frame = detector.process_frame(frame)
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)