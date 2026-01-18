#!/usr/bin/env python3
"""
PyTorch-based object tracker for Ultimate frisbee players using YOLOv8/YOLOv10.
Replaces TensorFlow YOLOv4 implementation with modern PyTorch + Ultralytics.
"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from absl import app, flags
from absl.flags import FLAGS

# PyTorch + Ultralytics imports
from ultralytics import YOLO

# Deep sort imports (framework-agnostic)
from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# Command line flags
flags.DEFINE_string("model", "yolov8", "YOLO model family: yolov8, yolov10, yolov26")
flags.DEFINE_string(
    "variant",
    "n",
    "Model variant: n (nano), s (small), m (medium), l (large), x (extra)",
)
flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string("output_df", None, "path to output CSV for tracking data")
flags.DEFINE_string(
    "output_format", "XVID", "codec used in VideoWriter when saving video to file"
)
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")
flags.DEFINE_boolean("dont_show", False, "dont show video output")
flags.DEFINE_boolean("info", False, "show detailed info of tracked objects")
flags.DEFINE_boolean("count", False, "count objects being tracked on screen")
flags.DEFINE_boolean(
    "limits",
    False,
    "Limit to detections inside the field. TODO: Make interactive to define bounds.",
)
flags.DEFINE_string("device", "auto", "Device: auto, cpu, cuda, mps")

# Field transformation coordinates (same as original)
RECT_COORD = [[683, 519], [1259, 169], [844, 120], [110, 250]]
TARGET_COORD = [[0, 0], [280, 0], [280, 150], [0, 150]]
TRANSFORM_MAT = cv2.getPerspectiveTransform(
    np.array(RECT_COORD, dtype=np.float32), np.array(TARGET_COORD, dtype=np.float32)
)

# Class names for COCO dataset (YOLO default)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class PyTorchObjectTracker:
    """PyTorch-based object tracker using YOLOv8/YOLOv10 and DeepSort."""

    def __init__(self, model_family: str, variant: str, device: str = "auto"):
        """Initialize the tracker with specified model."""
        self.model_family = model_family.lower()
        self.variant = variant.lower()
        self.device = self._get_device(device)

        # Load YOLO model
        model_name = f"{self.model_family}{self.variant}.pt"
        print(f"Loading {model_name} on {self.device}...")
        self.yolo_model = YOLO(model_name)

        # Initialize DeepSort
        self._init_deepsort()

        # Color map for visualization
        self.cmap = plt.get_cmap("tab20b")
        self.colors = [self.cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Allowed classes for Ultimate frisbee
        self.allowed_classes = ["person", "frisbee"]

    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _init_deepsort(self):
        """Initialize DeepSort tracker."""
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # Create encoder (using the same mock approach as realtime_tracker)
        def mock_encoder(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
            """Mock encoder that generates random features for testing."""
            return np.random.rand(len(boxes), 128).astype(np.float32)

        self.encoder = mock_encoder
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_age=60)
        self.nms_max_overlap = nms_max_overlap

        # Expose tracks for external access
        self.tracks = self.tracker.tracks

    def detect_objects(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO detection on frame.

        Returns:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            scores: Confidence scores
            classes: Class indices
        """
        # Run YOLO inference
        results = self.yolo_model(frame, conf=FLAGS.score, iou=FLAGS.iou, verbose=False)

        # Extract detections
        result = results[0]
        if result.boxes is None:
            return np.array([]), np.array([]), np.array([])

        # Convert to numpy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        return boxes, scores, classes

    def filter_detections(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Filter detections based on allowed classes and field limits.

        Returns:
            filtered_boxes: Filtered bounding boxes
            filtered_scores: Filtered confidence scores
            class_names: Class names for filtered detections
        """
        if len(boxes) == 0:
            return boxes, scores, []

        # Convert class indices to names
        class_names = [COCO_CLASSES[cls] for cls in classes]

        # Filter by allowed classes
        allowed_mask = np.array([name in self.allowed_classes for name in class_names])

        # Apply field limits if enabled
        if FLAGS.limits:
            field_mask = np.ones(len(boxes), dtype=bool)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                y_max_transformed = cv2.perspectiveTransform(
                    np.array([[x1, y2]], dtype="float32")[None, :, :],
                    TRANSFORM_MAT,
                ).flatten()[-1]
                if y_max_transformed > 150:
                    field_mask[i] = False
            allowed_mask = allowed_mask & field_mask

        # Apply filters
        filtered_boxes = boxes[allowed_mask]
        filtered_scores = scores[allowed_mask]
        filtered_class_names = [
            class_names[i] for i in range(len(class_names)) if allowed_mask[i]
        ]

        return filtered_boxes, filtered_scores, filtered_class_names

    def update_tracker(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_names: list[str],
    ):
        """Update DeepSort tracker with new detections."""
        # Create detections for DeepSort
        features = self.encoder(frame, boxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(
                boxes, scores, class_names, features, strict=True
            )
        ]

        # Run non-maxima suppression
        if len(detections) > 0:
            boxes_nms = np.array([d.tlwh for d in detections])
            scores_nms = np.array([d.confidence for d in detections])
            classes_nms = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes_nms, classes_nms, self.nms_max_overlap, scores_nms
            )
            detections = [detections[i] for i in indices]

        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # Update tracks reference
        self.tracks = self.tracker.tracks

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracked objects on frame."""
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Get color for this track
            color = self.colors[int(track.track_id) % len(self.colors)]
            color = [i * 255 for i in color]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )

            # Draw label background
            label = f"{class_name[0]}-{track.track_id}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, 2
            )[0]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1] - 30)),
                (int(bbox[0]) + label_size[0] + 10, int(bbox[1])),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (int(bbox[0]) + 5, int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.75,
                (255, 255, 255),
                2,
            )

            # Print info if enabled
            if FLAGS.info:
                print(
                    f"Tracker ID: {track.track_id}, Class: {class_name}, "
                    f"BBox Coords (xmin, ymin, xmax, ymax): "
                    f"({int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])})"
                )

        return frame


def main(_argv):
    """Main tracking function."""
    print(f"🚀 Starting PyTorch Object Tracker with {FLAGS.model}{FLAGS.variant}")

    # Initialize tracker
    tracker = PyTorchObjectTracker(FLAGS.model, FLAGS.variant, FLAGS.device)

    # Initialize video capture
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except (ValueError, TypeError):
        vid = cv2.VideoCapture(FLAGS.video)

    # Setup video writer if output specified
    out = None
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # Setup DataFrame for tracking data
    out_df = pd.DataFrame()

    frame_num = 0

    # Main processing loop
    while True:
        return_value, frame = vid.read()
        if not return_value:
            print("Video has ended or failed, try a different video format!")
            break

        frame_num += 1
        start_time = time.time()

        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detections
        boxes, scores, classes = tracker.detect_objects(frame_rgb)

        # Filter detections
        filtered_boxes, filtered_scores, class_names = tracker.filter_detections(
            boxes, scores, classes
        )

        # Update tracker
        tracker.update_tracker(frame_rgb, filtered_boxes, filtered_scores, class_names)

        # Draw tracks on original BGR frame
        result_frame = tracker.draw_tracks(frame)

        # Draw object count if enabled
        if FLAGS.count:
            count = len(
                [
                    t
                    for t in tracker.tracks
                    if t.is_confirmed() and t.time_since_update <= 1
                ]
            )
            cv2.putText(
                result_frame,
                f"Objects being tracked: {count}",
                (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0),
                2,
            )
            print(f"Objects being tracked: {count}")

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"Frame #: {frame_num} -- FPS: {round(fps, 1)}")

        # Show frame
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result_frame)

        # Save frame if output specified
        if out is not None:
            out.write(result_frame)

        # Save tracking data if specified
        if FLAGS.output_df:
            tracking_data = {}
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    track_id = track.track_id
                    # Calculate center point
                    center_x = int(0.5 * (bbox[0] + bbox[2]))
                    center_y = int(bbox[3])

                    tracking_data[f"{class_name}-{track_id}x"] = center_x
                    tracking_data[f"{class_name}-{track_id}y"] = center_y

            out_df = pd.concat(
                [out_df, pd.Series(tracking_data, name=frame_num).to_frame().T]
            )
            out_df.to_csv(FLAGS.output_df, index=True)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    vid.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
