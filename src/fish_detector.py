import cv2
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class FishDetector:
    """
    YOLOv8 detection + Hungarian tracking, with static filtering and overlap suppression.
    """

    def __init__(self, model_path='yolov8n.pt', max_disappeared=50, max_distance=75,
                 static_speed_threshold=1.0, static_patience=15,
                 allowed_classes=None, min_area=60, max_area_ratio=0.25, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.next_object_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.static_speed_threshold = static_speed_threshold
        self.static_patience = static_patience

        self.allowed_classes = allowed_classes or []
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.iou_threshold = iou_threshold

    def detect_and_track(self, frame):
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot(labels=False)

        raw_boxes, confs = [], []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            if self.allowed_classes and class_id not in self.allowed_classes:
                continue
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            if area < self.min_area or (area / frame_area) > self.max_area_ratio:
                continue
            raw_boxes.append((x1, y1, x2, y2))
            confs.append(conf)

        detected_boxes = self._suppress_overlaps(raw_boxes, confs, self.iou_threshold)
        centroids = np.array([((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in detected_boxes])

        if len(centroids) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]['disappeared'] += 1
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    self._deregister(object_id)
            tracked_objects = self._get_tracked_objects()
            self._draw_objects(annotated_frame, tracked_objects)
            return tracked_objects, annotated_frame

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self._register(centroids[i], detected_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            previous_centroids = np.array([obj['centroid'] for obj in self.objects.values()])

            D = cdist(previous_centroids, centroids)
            rows, cols = linear_sum_assignment(D)

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                new_centroid = centroids[col]
                old_centroid = self.objects[object_id]['centroid']
                speed_vector = (new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
                speed = np.linalg.norm(speed_vector)

                if speed < self.static_speed_threshold:
                    self.objects[object_id]['static_frames'] += 1
                else:
                    self.objects[object_id]['static_frames'] = 0

                self.objects[object_id]['speed_vector'] = speed_vector
                self.objects[object_id]['previous_centroid'] = old_centroid
                self.objects[object_id]['centroid'] = new_centroid
                self.objects[object_id]['box'] = detected_boxes[col]
                x1, y1, x2, y2 = detected_boxes[col]
                self.objects[object_id]['area'] = (x2 - x1) * (y2 - y1)
                self.objects[object_id]['disappeared'] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.objects[object_id]['disappeared'] += 1
                self.objects[object_id]['speed_vector'] = (0, 0)
                if self.objects[object_id]['disappeared'] > self.max_disappeared:
                    self._deregister(object_id)

            for col in unused_cols:
                self._register(centroids[col], detected_boxes[col])

        tracked_objects = self._get_tracked_objects()
        self._draw_objects(annotated_frame, tracked_objects)
        return tracked_objects, annotated_frame

    def _suppress_overlaps(self, boxes, confs, iou_thr):
        if not boxes:
            return []
        idxs = np.argsort(confs)[::-1]
        keep = []
        used = set()

        def iou(b1, b2):
            xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        for i in idxs:
            if i in used:
                continue
            b1 = boxes[i]
            keep.append(b1)
            for j in idxs:
                if j in used or j == i:
                    continue
                if iou(b1, boxes[j]) >= iou_thr:
                    used.add(j)
        return keep

    def _get_tracked_objects(self):
        tracked_objects = []
        for (object_id, obj) in self.objects.items():
            is_static = obj['static_frames'] > self.static_patience
            tracked_objects.append({
                'id': object_id,
                'centroid': obj['centroid'],
                'box': obj['box'],
                'area': obj['area'],
                'speed_vector': obj['speed_vector'],
                'disappeared': obj['disappeared'],
                'is_static': is_static
            })
        return tracked_objects

    def _draw_objects(self, frame, tracked_objects):
        max_draw_disappeared = 10  # Only show red IDs for 10 frames
        for obj in tracked_objects:
            if obj['disappeared'] > max_draw_disappeared:
                continue

            centroid = obj['centroid']
            color = (128, 128, 128) if obj['is_static'] else ((0, 0, 255) if obj['disappeared'] > 0 else (0, 255, 0))
            text = f"ID {obj['id']}"
            cv2.putText(frame, text, (int(centroid[0] - 10), int(centroid[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _register(self, centroid, box):
        x1, y1, x2, y2 = box
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'previous_centroid': centroid,
            'box': box,
            'area': (x2 - x1) * (y2 - y1),
            'speed_vector': (0, 0),
            'disappeared': 0,
            'static_frames': 0
        }
        self.next_object_id += 1

    def _deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]