import cv2
import time


class FrameExtractor:
    def __init__(self, stream_url: str, save_path="frames/latest.jpg"):
        self.stream_url = stream_url
        self.save_path = save_path
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open stream")

    def grab_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        cv2.imwrite(self.save_path, frame)
        return True

    def run(self, delay=0.05):
        self.start()
        while True:
            self.grab_frame()
            time.sleep(delay)

    def stop(self):
        if self.cap:
            self.cap.release()


# Example
if __name__ == "__main__":
    url = "rtsp://192.0.0.4:8080/h264.sdp"
    extractor = FrameExtractor(url)
    extractor.run()
