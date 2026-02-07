import cv2
import time
import requests
from ultralytics import YOLO
from datetime import datetime


# =====================================================
# Camera Controller (Safe)
# =====================================================
class IPCameraController:

    def __init__(self, ip, port=8080):
        self.base = f"http://{ip}:{port}"
        self.last_change = 0
        self.current_res = "Unknown"
        self.cooldown = 3

    def set_resolution(self, res):

        now = time.time()

        if res == self.current_res:
            return

        if now - self.last_change < self.cooldown:
            return

        url = f"{self.base}/settings/video_size?set={res}"

        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                self.current_res = res
                self.last_change = now
                print("Resolution changed ->", res)
        except:
            pass


# =====================================================
# Smart Frame Processor
# =====================================================
class SmartEdgeProcessor:

    def __init__(self, stream_url, cam_ip):

        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            raise RuntimeError("Stream failed")

        self.model = YOLO("yolov8n.pt")
        self.cam = IPCameraController(cam_ip)

        self.score = 0
        self.bandwidth_mbps = 0
        self.frame_bytes = 0
        self.obj_count = 0

    def compute_score(self, boxes):
        self.obj_count = len(boxes)
        return min(self.obj_count / 8.0, 1.0)

    def adaptive_resolution(self):

        if self.score > 0.7:
            target = "1920x1080"
        elif self.score > 0.3:
            target = "1280x720"
        else:
            target = "640x480"

        self.cam.set_resolution(target)

    def run(self):

        prev = time.time()
        fps = 0
        count = 0

        while True:

            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame, verbose=False)[0]
            self.score = self.compute_score(results.boxes)
            annotated = results.plot()

            # FPS
            count += 1
            now = time.time()
            if now - prev >= 1:
                fps = count
                count = 0
                prev = now

                # Bandwidth estimation
                self.frame_bytes = frame.nbytes
                bytes_per_sec = self.frame_bytes * fps
                self.bandwidth_mbps = (bytes_per_sec * 8) / 1e6

            self.adaptive_resolution()

            h, w, _ = frame.shape

            # ================= Overlay =================

            meta = [
                f"Score: {self.score:.2f}",
                f"Objects: {self.obj_count}",
                f"FPS: {fps}",
                f"Bandwidth: {self.bandwidth_mbps:.2f} Mbps",
                f"Frame: {w}x{h}",
                f"CameraRes: {self.cam.current_res}",
                f"FrameSize: {self.frame_bytes/1e6:.2f} MB",
                datetime.now().strftime("%H:%M:%S")
            ]

            y = 30
            for line in meta:
                cv2.putText(
                    annotated,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )
                y += 30

            cv2.imshow("Edge Live", annotated)

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":

    STREAM = "http://192.0.0.4:8080/video"
    CAM_IP = "192.0.0.4"

    node = SmartEdgeProcessor(STREAM, CAM_IP)
    node.run()
