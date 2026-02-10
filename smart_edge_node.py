import cv2
import time
from datetime import datetime

from camera_controller import IPCameraController
from frame_extractor import FrameExtractor
from importance_score import ImportanceScorer


# =====================================================
# Visual Edge Node (Composition Layer)
# =====================================================
class VisualEdgeNode:

    def __init__(self, stream_url, cam_ip):

        self.stream_url = stream_url

        self.camera = IPCameraController(cam_ip)
        self.extractor = FrameExtractor(stream_url)
        self.scorer = ImportanceScorer("yolov8n-face.pt")
        self.current_quality = 20
        # live capture for display
        self.cap = cv2.VideoCapture(stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Stream failed")

        self.score = 0
        self.obj_count = 0
        self.bandwidth = 0
        self.frame_bytes = 0
        self.current_res = None
        


    # --------------------------------------------------
    def adaptive_resolution(self):

        # -----------------------
        # Resolution logic
        # -----------------------
        if self.score >= 0.5:
            target_res = (1920, 1080)
        else:
            target_res = (1280, 720)

        if target_res != self.current_res:
            old_res = self.current_res
            self.camera.set_resolution(target_res[0], target_res[1])
            self.current_res = target_res
            print(f"[ADAPT] Resolution changed: {old_res} -> {target_res} | score={self.score:.2f}")

        # -----------------------
        # Quality logic
        # -----------------------
        if self.score >= 0.8:
            target_quality = 50
        elif self.score >= 0.6:
            target_quality = 40
        elif self.score >= 0.4:
            target_quality = 30
        else:
            target_quality = 20

        # Hard clamp (because entropy)
        target_quality = max(20, min(50, target_quality))

        if target_quality != self.current_quality:
            old_q = self.current_quality
            self.camera.set_quality(target_quality)
            self.current_quality = target_quality
            print(f"[ADAPT] Quality changed: {old_q} -> {target_quality} | score={self.score:.2f}")

    # --------------------------------------------------
    def run(self):

        # start background frame saving
        import threading
        threading.Thread(target=self.extractor.run, daemon=True).start()

        prev = time.time()
        fps = 0
        count = 0

        # cv2.namedWindow("Edge Visual Node", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Edge Visual Node", 1000, 700)


        while True:
            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()
            if not ret:
                continue

            # score from saved frame
            self.score = self.scorer.get_score()
            if not self.score: continue
            self.obj_count = int(self.score * 10)

            # fps
            count += 1
            now = time.time()
            if now - prev >= 1:
                fps = count
                count = 0
                prev = now

                self.frame_bytes = frame.nbytes
                self.bandwidth = (self.frame_bytes * fps * 8) / 1e6

            self.adaptive_resolution()

            # ============ Overlay ============
            h, w, _ = frame.shape

            meta = [
                f"Score: {self.score:.2f}",
                f"Objects(est): {self.obj_count}",
                f"FPS: {fps}",
                f"Bandwidth: {self.bandwidth:.2f} Mbps",
                f"Frame: {w}x{h}",
                f"CameraRes: {self.current_res}",
                datetime.now().strftime("%H:%M:%S")
            ]

            y = 30
            for line in meta:
                cv2.putText(frame, line, (20,y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)
                y += 30

        #     cv2.imshow("Edge Visual Node", frame)

        #     if cv2.waitKey(1) == 27:
        #         break

        # self.cap.release()
        # cv2.destroyAllWindows()


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":

    STREAM = "http://10.63.44.97:8080/video"
    CAM_IP = "10.63.44.97"

    node = VisualEdgeNode(STREAM, CAM_IP)
    node.run()
