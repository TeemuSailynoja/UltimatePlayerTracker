"""
YOLOv10 Inference Wrapper

Provides inference functionality for YOLOv10 models with DeepSORT integration.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any, Union
import time
import logging

logger = logging.getLogger(__name__)


class YOLOv10Inference:
    """
    YOLOv10 inference wrapper for object detection.

    This class provides a clean interface for running inference with YOLOv10 models,
    handling preprocessing, inference, and postprocessing automatically.
    """

    def __init__(self, model, config):
        """
        Initialize YOLOv10 inference.

        Args:
            model: Loaded YOLOv10 model
            config: YOLOv10Config instance
        """
        self.model = model
        self.config = config
        self.inference_count = 0
        self.total_inference_time = 0.0

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB (YOLO expects RGB)
        if image.shape[-1] == 3:  # BGR image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def infer_single(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image: Input image in BGR format

        Returns:
            List of detection results
        """
        # Preprocess
        processed_image = self.preprocess_image(image)

        # Run inference
        start_time = time.time()

        try:
            results = self.model(
                processed_image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_det,
                verbose=self.config.verbose,
            )

            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time

            if self.config.verbose:
                logger.debug(f"Inference time: {inference_time:.3f}s")

            return self._process_results(results[0])

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def infer_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images

        Returns:
            List of detection results for each image
        """
        if len(images) == 0:
            return []

        # Preprocess batch
        processed_images = [self.preprocess_image(img) for img in images]

        # Run batch inference
        start_time = time.time()

        try:
            results = self.model(
                processed_images,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_det,
                verbose=self.config.verbose,
            )

            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time

            if self.config.verbose:
                logger.debug(
                    f"Batch inference time: {inference_time:.3f}s for {len(images)} images"
                )

            return [self._process_results(result) for result in results]

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise

    def _process_results(self, result) -> List[Dict[str, Any]]:
        """
        Process YOLOv10 results into standard format.

        Args:
            result: YOLOv10 result object

        Returns:
            List of detection dictionaries
        """
        detections = []

        if result.boxes is not None:
            boxes = result.boxes

            # Convert to CPU numpy arrays
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
            conf = boxes.conf.cpu().numpy()  # Confidence scores
            cls = boxes.cls.cpu().numpy()  # Class indices

            for i in range(len(xyxy)):
                detection = {
                    "bbox": xyxy[i].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(conf[i]),
                    "class_id": int(cls[i]),
                    "class_name": self._get_class_name(int(cls[i])),
                }

                # Filter by target classes if specified
                if (
                    self.config.target_classes is None
                    or int(cls[i]) in self.config.target_classes
                ):
                    detections.append(detection)

        return detections

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        if self.config.class_names and class_id in self.config.class_names:
            return self.config.class_names[class_id]
        return f"class_{class_id}"

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get inference performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if self.inference_count == 0:
            return {"avg_inference_time": 0.0, "fps": 0.0}

        avg_time = self.total_inference_time / self.inference_count
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "total_inferences": self.inference_count,
            "total_time": self.total_inference_time,
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0

    def detect_video_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect objects in a video frame and return annotated frame.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Tuple of (detections, annotated_frame)
        """
        # Run inference
        detections = self.infer_single(frame)

        # Create annotated frame if needed
        annotated_frame = frame.copy()
        if self.config.verbose or self.config.save_results:
            annotated_frame = self._annotate_frame(annotated_frame, detections)

        return detections, annotated_frame

    def _annotate_frame(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Annotate frame with detection results.

        Args:
            frame: Input frame (BGR format)
            detections: List of detection results

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class_name"]

            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background for label
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            # Text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return annotated
