from typing import Tuple
import numpy as np
from ultralytics import YOLO


class YOLOMODEL:
    def __init__(self, weights_path: str):
        self.model = YOLO(model=weights_path, task='detect')
        self.model_pose = YOLO(model='./weights/yolov8n-pose.pt')

    def predict(self, image) -> Tuple:
        results = self.model.predict(image, device='cpu', stream=True)
        results_pose = self.model_pose.predict(image, device='cpu', conf=0.3,  imgsz=600, stream=True, boxes=False)
        keypoints = []
        for r in results_pose:
            keypoint = r.keypoints.xy.numpy()
            for i in range(keypoint.shape[0]):
                keypoints.append(keypoint)

        bboxes, labels, scores = [], [], []
        for r in results:
            boxes = r.boxes.xywh.numpy()
            cls = r.boxes.cls.numpy()
            scores_ = r.boxes.conf.numpy()
            label_dict = r.names
            for i in range(boxes.shape[0]):
                x, y, w, h = boxes[i]
                xmin, ymin = x - w / 2, y - h / 2
                xmax, ymax = x + w / 2, y + h / 2
                label = label_dict[cls[i]]
                score = scores_[i]
                bbox = (xmin, ymin, xmax, ymax)
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)
        return bboxes, labels, scores, keypoints
