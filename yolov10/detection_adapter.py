"""
YOLOv10 Detection Adapter

Converts YOLOv10 output to DeepSORT compatible format.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DetectionAdapter:
    """
    Adapter class to convert YOLOv10 detections to DeepSORT compatible format.

    This class maintains compatibility with the existing DeepSORT tracking pipeline
    by converting YOLOv10 output to the expected format from the old filter_boxes() function.
    """

    def __init__(
        self,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.25,
    ):
        """
        Initialize DetectionAdapter.

        Args:
            target_classes: List of target class names (e.g., ['person', 'sports ball'])
            confidence_threshold: Minimum confidence for detections
        """
        self.target_classes = target_classes or ["person", "sports ball"]
        self.confidence_threshold = confidence_threshold

        # COCO class mappings (subset for Ultimate frisbee)
        self.class_mapping = {
            "person": 0,
            "sports ball": 37,  # COCO sports ball class
        }

        # Reverse mapping for filtering
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}

    def convert_to_deepsort_format(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Convert YOLOv10 detections to DeepSORT compatible format.

        Args:
            detections: List of YOLOv10 detection dictionaries

        Returns:
            Tuple of (bboxes, scores, classes, num_objects) matching filter_boxes() signature
            - bboxes: numpy array of [x, y, w, h] format
            - scores: numpy array of confidence scores
            - classes: numpy array of class indices
            - num_objects: number of detected objects
        """
        if not detections:
            return np.array([]), np.array([]), np.array([]), 0

        # Filter by target classes and confidence
        filtered_detections = []
        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]

            # Check if this is a target class and meets confidence threshold
            if (
                class_name in self.target_classes
                and confidence >= self.confidence_threshold
            ):
                filtered_detections.append(det)

        if not filtered_detections:
            return np.array([]), np.array([]), np.array([]), 0

        # Extract information
        bboxes_xyxy = []
        scores = []
        classes = []

        for det in filtered_detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2] in xyxy format
            conf = det["confidence"]
            class_id = det["class_id"]

            # Convert xyxy to xywh (DeepSORT expects [x, y, w, h])
            x1, y1, x2, y2 = bbox
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            bboxes_xyxy.append([x, y, w, h])
            scores.append(conf)
            classes.append(class_id)

        # Convert to numpy arrays
        bboxes_array = np.array(bboxes_xyxy, dtype=np.float32)
        scores_array = np.array(scores, dtype=np.float32)
        classes_array = np.array(classes, dtype=np.int32)
        num_objects = len(filtered_detections)

        return bboxes_array, scores_array, classes_array, num_objects

    def filter_by_class(
        self, detections: List[Dict[str, Any]], allowed_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter detections to only include specified classes.

        Args:
            detections: List of YOLOv10 detections
            allowed_classes: List of allowed class names

        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det["class_name"] in allowed_classes]

    def filter_ultimate_frisbee_objects(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter detections for Ultimate frisbee specific objects.

        Args:
            detections: List of YOLOv10 detections

        Returns:
            Filtered list containing only players and frisbee
        """
        ultimate_classes = ["person", "sports ball"]
        return self.filter_by_class(detections, ultimate_classes)

    def convert_coordinates(
        self,
        bboxes_xyxy: np.ndarray,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert coordinates from model input space to original image space.

        Args:
            bboxes_xyxy: Bounding boxes in model input coordinates
            input_size: Model input size (width, height)
            original_size: Original image size (width, height)

        Returns:
            Bounding boxes in original image coordinates
        """
        if len(bboxes_xyxy) == 0:
            return bboxes_xyxy

        input_w, input_h = input_size
        orig_w, orig_h = original_size

        # Calculate scale factors
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        # Scale coordinates
        scaled_bboxes = bboxes_xyxy.copy()
        scaled_bboxes[:, 0] *= scale_x  # x
        scaled_bboxes[:, 1] *= scale_y  # y
        scaled_bboxes[:, 2] *= scale_x  # w
        scaled_bboxes[:, 3] *= scale_y  # h

        return scaled_bboxes

    def validate_detections(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Validate detection format and content.

        Args:
            detections: List of detections to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(detections, list):
            logger.error("Detections must be a list")
            return False

        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                logger.error(f"Detection {i} is not a dictionary")
                return False

            required_keys = ["bbox", "confidence", "class_id", "class_name"]
            for key in required_keys:
                if key not in det:
                    logger.error(f"Detection {i} missing required key: {key}")
                    return False

            # Validate bbox format
            bbox = det["bbox"]
            if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
                logger.error(f"Detection {i} has invalid bbox format: {bbox}")
                return False

        return True

    def get_class_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get statistics of detected classes.

        Args:
            detections: List of detections

        Returns:
            Dictionary with class counts
        """
        class_counts = {}
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

    def format_for_deepsort(
        self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Format detections specifically for DeepSORT pipeline.

        Args:
            detections: List of YOLOv10 detections
            image_shape: Original image shape (height, width, channels)

        Returns:
            Tuple in DeepSORT format (bboxes, scores, classes, num_objects)
        """
        if not self.validate_detections(detections):
            logger.error("Invalid detections provided")
            return np.array([]), np.array([]), np.array([]), 0

        # Filter for Ultimate frisbee objects
        filtered_detections = self.filter_ultimate_frisbee_objects(detections)

        # Convert to DeepSORT format
        return self.convert_to_deepsort_format(filtered_detections)

    def get_info(self) -> Dict[str, Any]:
        """
        Get adapter configuration information.

        Returns:
            Dictionary with adapter configuration
        """
        return {
            "target_classes": self.target_classes,
            "confidence_threshold": self.confidence_threshold,
            "class_mapping": self.class_mapping,
            "reverse_mapping": self.reverse_mapping,
        }
