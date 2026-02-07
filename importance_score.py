from ultralytics import YOLO
import cv2
import os


class ImportanceScorer:
    def __init__(self, model_path: str, frame_path="frames/latest.jpg"):
        self.model = YOLO(model_path)
        self.frame_path = frame_path

    def compute_score(self, detections):
        count = len(detections)
        score = min(count / 10.0, 1.0)
        return score

    def get_score(self):
        if not os.path.exists(self.frame_path):
            return 0.0

        img = cv2.imread(self.frame_path)
        if img is None:
            return 0.0

        results = self.model(img, verbose=False)[0]
        score = self.compute_score(results.boxes)

        return score


# Example standalone usage
if __name__ == "__main__":
    scorer = ImportanceScorer("yolov8n.pt")
    print("Importance Score:", scorer.get_score())
