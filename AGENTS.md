# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in the UltimatePlayerTracker repository.

## Project Overview

UltimatePlayerTracker is a Python-based object tracking system for Ultimate frisbee players using modern YOLO models (YOLOv8/YOLOv10/YOLOv26) for object detection and DeepSort for tracking. The project uses PyTorch with Ultralytics and follows a computer vision pipeline architecture.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Install Pixi package manager (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies with Pixi (RECOMMENDED)
pixi install

# Legacy pip install (deprecated - TensorFlow no longer supported)
# pip install -r requirements.txt  # For CPU
# pip install -r requirements-gpu.txt  # For GPU
```

**IMPORTANT**: The project has been fully migrated from TensorFlow to PyTorch. Use Pixi for environment management. TensorFlow dependencies are deprecated and should not be used.

### Code Quality Tools
The project uses the following development dependencies (defined in pyproject.toml):
- **ruff**: Combined linting and formatting (10-100x faster than black/flake8)
- **mypy**: Type checking (v0.910)

```bash
# Format code
pixi run ruff format .

# Run linting
pixi run ruff check .

# Fix auto-fixable issues
pixi run ruff check --fix .

# Run type checking
pixi run mypy .

# Run all quality checks together
pixi run check-all
```

### Running the Application
```bash
# Run object tracker on video (default YOLOv8-nano)
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi

# Run with specific model
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov8
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov10
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolo26

# Run with webcam (set video flag to 0)
pixi run python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov8

# Run with different model variants
pixi run python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/test.avi --model yolov8 --variant n
pixi run python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/test.avi --model yolov10 --variant s
```

### Running Benchmarks
```bash
# Run YOLOv8 benchmark (recommended for testing)
pixi run python benchmark/yolo8_benchmark.py

# Run modern YOLO comparison (YOLOv8 vs YOLOv10 vs YOLO26)
pixi run python benchmark/modern_yolo_comparison.py

# Run specific model families
pixi run python benchmark/modern_yolo_comparison.py --models yolov8 yolov10

# Run YOLO26-specific benchmarks
pixi run python benchmark/yolo26_performance_benchmark.py

# Check current status
pixi run python benchmark/show_status.py
```

### Testing
This project currently does not have formal unit tests. Testing is done by running the main application with sample videos. When adding tests, place them in a `tests/` directory and use pytest.

## Code Style Guidelines

### Import Organization
- Standard library imports first (os, sys, time, etc.)
- Third-party imports next (numpy, torch, cv2, PIL, etc.)
- Local imports last (core.*, deep_sort.*, tools.*, yolov26.*)
- Use absolute imports for local modules
- Group related imports together

```python
# Standard library
import os
import time

# Third-party
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO

# Local imports
import core.utils as utils
from deep_sort.tracker import Tracker
from yolov26.model_loader import ModelLoader
from yolov26.inference import YOLO26Inference
```

### Formatting Conventions
- Use Ruff for automatic formatting (line length 88)
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
- Handle PyTorch/CUDA device configuration gracefully
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

### PyTorch/Ultralytics Specific Guidelines
- Use PyTorch 2.x patterns with modern ultralytics API
- Configure GPU memory efficiently with CUDA when available
- Use torch tensors for model operations
- Leverage ultralytics built-in optimizations and callbacks
- Prefer model.export() for deployment optimizations

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
- `checkpoints/`: Saved models (legacy TensorFlow, now PyTorch)

### Performance Considerations
- Minimize memory allocations in video processing loops
- Use numpy operations instead of Python loops when possible
- Pre-allocate arrays for repeated operations
- Consider GPU memory usage for PyTorch operations
- Leverage ultralytics built-in optimizations and callbacks

### Testing and Validation
- Test with sample videos before deploying
- Validate model paths and file existence
- Check video capture properties before processing
- Log frame processing metrics (FPS) for performance monitoring

## Framework-Specific Notes

### PyTorch 2.x
- Uses modern PyTorch 2.x with latest optimizations
- Supports dynamic computation graphs and eager execution
- Excellent CUDA acceleration support
- Compatible with latest ultralytics features

### Ultralytics 8.x
- Provides unified interface for YOLOv8/YOLOv10/YOLOv26
- Built-in data augmentation and training utilities
- Automatic model optimization and export capabilities
- Active development and regular updates

### OpenCV
- Use BGR format for OpenCV operations, convert to RGB when needed
- Handle video codec compatibility for output formats
- Use cv2.VideoWriter for output video generation

### DeepSort Integration
- Follow the existing DeepSort class interfaces
- Maintain compatibility with the original DeepSort paper implementation
- Use cosine distance for appearance-based matching