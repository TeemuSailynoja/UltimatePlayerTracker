"""
YOLOv10 Configuration

Configuration management for YOLOv10 models and detection parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class YOLOv10Config:
    """
    Configuration class for YOLOv10 detection parameters.

    This class provides a centralized configuration system for YOLOv10 models,
    supporting different variants, hardware targets, and optimization settings.
    """

    # Model Configuration
    model_variant: str = "yolov10s"  # n, s, m, b, l, x
    model_path: Optional[str] = None  # Custom model path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Hardware Configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 1
    num_workers: int = 0

    # Image Configuration
    input_size: int = 640  # Square input size
    letterbox: bool = True  # Maintain aspect ratio

    # Performance Configuration
    half_precision: bool = True  # FP16 for GPU
    optimize_for_speed: bool = True  # Optimize for inference speed
    max_det: int = 300  # Maximum detections per image

    # Class Filtering (for Ultimate frisbee)
    target_classes: List[int] = None  # None = all classes
    class_names: Dict[int, str] = None

    # Export Configuration
    export_format: str = "onnx"  # onnx, torchscript, coreml, tflite
    export_optimize: bool = True

    # Tracking Integration
    deepsort_compatible: bool = True  # Ensure DeepSORT compatibility
    coordinate_format: str = "xyxy"  # xyxy, xywh

    # Logging and Debugging
    verbose: bool = False
    save_results: bool = False
    output_dir: str = "./outputs"

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.target_classes is None:
            # Default to person (0) and sports ball (37) for Ultimate frisbee
            self.target_classes = [0, 37]

        if self.class_names is None:
            # COCO class names (subset relevant for Ultimate frisbee)
            self.class_names = {
                0: "person",
                32: "sports ball",  # Updated for COCO
                37: "sports ball",  # Alternative index
            }

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = self._detect_device()

    def _detect_device(self) -> str:
        """Automatically detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLOv10Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args) -> "YOLOv10Config":
        """Create configuration from command line arguments."""
        config_dict = {}

        # Map common argument names
        arg_mapping = {
            "model": "model_variant",
            "weights": "model_path",
            "score": "confidence_threshold",
            "iou": "iou_threshold",
            "device": "device",
            "img_size": "input_size",
            "half": "half_precision",
        }

        for arg_name, config_name in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[config_name] = getattr(args, arg_name)

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_variant": self.model_variant,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "half_precision": self.half_precision,
            "target_classes": self.target_classes,
            "export_format": self.export_format,
            "verbose": self.verbose,
        }

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        # Validate model variant
        valid_variants = [
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ]
        if self.model_variant not in valid_variants:
            errors.append(f"Invalid model variant: {self.model_variant}")

        # Validate thresholds
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"Invalid confidence threshold: {self.confidence_threshold}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            errors.append(f"Invalid IOU threshold: {self.iou_threshold}")

        # Validate input size
        if self.input_size <= 0 or self.input_size > 1280:
            errors.append(f"Invalid input size: {self.input_size}")

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Invalid batch size: {self.batch_size}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model variant information."""
        model_specs = {
            "yolov10n": {"params": "2.3M", "flops": "6.7B", "map": "39.5"},
            "yolov10s": {"params": "7.2M", "flops": "21.6B", "map": "46.7"},
            "yolov10m": {"params": "15.4M", "flops": "59.1B", "map": "51.3"},
            "yolov10b": {"params": "24.4M", "flops": "92.0B", "map": "52.7"},
            "yolov10l": {"params": "29.5M", "flops": "120.3B", "map": "53.3"},
            "yolov10x": {"params": "56.9M", "flops": "160.4B", "map": "54.4"},
        }

        return model_specs.get(self.model_variant, {})

    def optimize_for_hardware(self) -> "YOLOv10Config":
        """Optimize configuration based on detected hardware."""
        if self.device == "cpu":
            # CPU optimizations
            self.half_precision = False
            self.optimize_for_speed = True
            self.batch_size = 1
            # Use smaller model for CPU
            if self.model_variant in ["yolov10l", "yolov10x"]:
                self.model_variant = "yolov10m"

        elif self.device == "cuda":
            # GPU optimizations
            self.half_precision = True
            self.optimize_for_speed = True
            # Can use larger models on GPU
            if self.model_variant == "yolov10n":
                self.model_variant = "yolov10s"

        return self

    def copy(self, **kwargs) -> "YOLOv10Config":
        """Create a copy of the configuration with optional updates."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return YOLOv10Config.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = YOLOv10Config()
