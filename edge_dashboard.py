import cv2
import time
import threading
import requests
from flask import Flask, Response, jsonify
from ultralytics import YOLO


# ==========================================================
# Camera Controller
# ==========================================================
class IPCameraController:

    def __init__(self, ip, port=8080):
        self.base = f"http://{ip}:{port}"
        self.current_res = "Unknown"

    def set_resolution(self, res):
        url = f"{self.base}/settings/video_size?set={res}"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                self.current_res = res
        except:
            pass


# ==========================================================
# Edge Processing Engine
# ==========================================================
class EdgeProcessor:

    def __init__(self, stream_url, cam_controller):

        self.cap = cv2.VideoCapture(stream_url)
        self.model = YOLO("yolov8n.pt")

        self.frame = None
        self.score = 0
        self.fps = 0
        self.bandwidth = 0

        self.cam = cam_controller

    def compute_score(self, boxes):
        return min(len(boxes) / 8.0, 1.0)

    def adaptive_resolution(self):

        if self.score > 0.7:
            self.cam.set_resolution("1920x1080")

        elif self.score > 0.3:
            self.cam.set_resolution("1280x720")

        else:
            self.cam.set_resolution("640x480")

    def run(self):

        frame_count = 0
        start = time.time()

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame, verbose=False)[0]
            self.score = self.compute_score(results.boxes)
            self.frame = results.plot()

            frame_count += 1

            if time.time() - start >= 1:
                self.fps = frame_count
                frame_count = 0
                start = time.time()

            self.bandwidth = frame.nbytes * self.fps / 1e6
            self.adaptive_resolution()


# ==========================================================
# Flask Dashboard
# ==========================================================
app = Flask(__name__)

CAM_IP = "192.0.0.4"
STREAM_URL = "http://192.0.0.4:8080/video"

controller = IPCameraController(CAM_IP)
edge = EdgeProcessor(STREAM_URL, controller)


def generate_stream():
    while True:
        if edge.frame is None:
            continue

        _, jpg = cv2.imencode('.jpg', edge.frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpg.tobytes() + b'\r\n')


@app.route("/")
def index():
    return """
    <html>
    <body style="background:#111;color:white;font-family:sans-serif">
    <h2>Edge Surveillance Dashboard</h2>

    <img src="/video" width="720"/>

    <div id="meta"></div>

    <script>
    setInterval(()=>{
        fetch('/meta')
        .then(r=>r.json())
        .then(d=>{
            document.getElementById("meta").innerHTML =
            "Score: " + d.score + "<br>" +
            "FPS: " + d.fps + "<br>" +
            "Bandwidth MB/s: " + d.bandwidth + "<br>" +
            "Resolution: " + d.resolution + "<br>" +
            "Stream URL: " + d.url;
        })
    },1000)
    </script>

    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/meta")
def meta():
    return jsonify({
        "score": round(edge.score, 3),
        "fps": edge.fps,
        "bandwidth": round(edge.bandwidth, 2),
        "resolution": controller.current_res,
        "url": STREAM_URL
    })


# ==========================================================
# Start Threads
# ==========================================================
if __name__ == "__main__":

    threading.Thread(target=edge.run, daemon=True).start()

    app.run(host="0.0.0.0", port=5001)
