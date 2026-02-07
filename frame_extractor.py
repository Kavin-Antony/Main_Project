import cv2
import time


class FrameExtractor:

    def __init__(self, stream_url: str, save_path="frames/latest.jpg"):
        self.stream_url = stream_url
        self.save_path = save_path
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open stream")

    def grab_frame(self):

        flushed = 0

        # Keep grabbing until buffer empty
        while flushed < 10 and self.cap.grab():
            flushed += 1

        ret, frame = self.cap.read()
        if not ret:
            return False

        cv2.imwrite(self.save_path, frame)
        return True

    def run(self, delay=0.3):
        self.start()

        while True:
            self.grab_frame()
            time.sleep(delay)

    def stop(self):
        if self.cap:
            self.cap.release()
