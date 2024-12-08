from flask import Flask, Response, render_template, request, redirect, url_for
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolov8n.pt')

cap = None

def generate_frames():
    global cap
    while True:
        if cap is None:
            break
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    if cap is not None:
        cap.release()
        cap = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        source = request.form["source"]
        global cap
        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            elif source.startswith("http") or source.endswith(".mp4"):
                cap = cv2.VideoCapture(source)
            else:
                raise ValueError("Invalid video source. Use 0, RTSP URL, or MP4 path.")
            if not cap.isOpened():
                raise ValueError("Failed to open video source.")
            return redirect(url_for('detection'))
        except Exception as e:
            return f"<h1>Error: {e}</h1><a href='/'>Go Back</a>"
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/stop', methods=['POST'])
def stop():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
