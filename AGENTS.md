# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in the UltimatePlayerTracker repository.

## Project Overview

UltimatePlayerTracker is a Python-based object tracking system for Ultimate frisbee players using YOLOv4 for object detection and DeepSort for tracking. The project uses TensorFlow 2.3.0 and follows a computer vision pipeline architecture.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Install dependencies (Poetry is the preferred package manager)
pip install poetry
poetry install

# Or install with pip directly
pip install -r requirements.txt  # For CPU
pip install -r requirements-gpu.txt  # For GPU
```

### Code Quality Tools
The project uses the following development dependencies (defined in pyproject.toml):
- **black**: Code formatting (v21.9b0)
- **flake8**: Linting (v4.0.1) 
- **pycodestyle**: Style checking (v2.8.0)
- **mypy**: Type checking (v0.910)

```bash
# Format code
poetry run black .

# Run linting
poetry run flake8 .

# Run type checking
poetry run mypy .

# Run all quality checks together
poetry run black . && poetry run flake8 . && poetry run mypy .
```

### Running the Application
```bash
# Convert YOLOv4 weights to TensorFlow model
python save_model.py --model yolov4

# Run object tracker on video
python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov4

# Run with webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4

# Run with YOLOv4-tiny (faster but less accurate)
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

### Testing
This project currently does not have formal unit tests. Testing is done by running the main application with sample videos. When adding tests, place them in a `tests/` directory and use pytest.

## Code Style Guidelines

### Import Organization
- Standard library imports first (os, sys, time, etc.)
- Third-party imports next (numpy, tensorflow, cv2, PIL, etc.)
- Local imports last (core.*, deep_sort.*, tools.*)
- Use absolute imports for local modules
- Group related imports together

```python
# Standard library
import os
import time

# Third-party
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Local imports
import core.utils as utils
from core.yolov4 import filter_boxes
from deep_sort.tracker import Tracker
```

### Formatting Conventions
- Use Black for automatic formatting (line length 88)
- 4-space indentation (no tabs)
- Maximum line length: 88 characters
- Use double quotes for strings and docstrings
- Leave 2 blank lines before top-level functions/classes

### Type Hints
- Use type hints for function parameters and return values
- Import typing constructs as needed (Optional, List, Tuple, etc.)
- Use numpy array types where appropriate: `np.ndarray`

### Naming Conventions
- **Variables**: snake_case (e.g., `max_cosine_distance`, `frame_size`)
- **Functions**: snake_case (e.g., `filter_boxes`, `load_weights`)
- **Classes**: PascalCase (e.g., `Tracker`, `Detection`, `KalmanFilter`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `FLAGS`, `TRANSFORM_MAT`)
- **Module names**: snake_case (e.g., `object_tracker.py`, `kalman_filter.py`)

### Error Handling
- Use try-except blocks for file I/O and video operations
- Handle TensorFlow GPU configuration gracefully
- Provide meaningful error messages
- Use specific exception types when possible

```python
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)
```

### Documentation
- Use docstrings for all classes and public functions
- Follow NumPy docstring format for parameters and returns
- Include parameter types and descriptions
- Add brief usage examples for complex functions

### TensorFlow Specific Guidelines
- Use TensorFlow 2.x eager execution patterns
- Configure GPU memory growth to prevent allocation issues
- Use tf.constant for fixed values, tf.Variable for trainable parameters
- Prefer tf.keras layers when building models

### Computer Vision Pipeline Patterns
- Follow the detection → tracking → visualization pipeline
- Use consistent coordinate systems (normalize when needed)
- Handle frame-by-frame processing in main loops
- Store intermediate results for debugging

### Configuration Management
- Use the cfg object from core.config for all configuration
- Define constants at module level when appropriate
- Use command-line flags (absl.flags) for runtime parameters

### File Organization
- `core/`: YOLO model implementation and utilities
- `deep_sort/`: Tracking algorithm components
- `tools/`: Helper scripts and utilities
- `data/`: Input data, weights, and class files
- `checkpoints/`: Saved TensorFlow models

### Performance Considerations
- Minimize memory allocations in video processing loops
- Use numpy operations instead of Python loops when possible
- Pre-allocate arrays for repeated operations
- Consider GPU memory usage for TensorFlow operations

### Testing and Validation
- Test with sample videos before deploying
- Validate model paths and file existence
- Check video capture properties before processing
- Log frame processing metrics (FPS) for performance monitoring

## Framework-Specific Notes

### TensorFlow 2.3.0
- This project uses an older TensorFlow version for compatibility
- Avoid using newer TF features not available in 2.3.0
- Use tf.compat.v1 for TF 1.x compatibility when needed

### OpenCV
- Use BGR format for OpenCV operations, convert to RGB when needed
- Handle video codec compatibility for output formats
- Use cv2.VideoWriter for output video generation

### DeepSort Integration
- Follow the existing DeepSort class interfaces
- Maintain compatibility with the original DeepSort paper implementation
- Use cosine distance for appearance-based matching