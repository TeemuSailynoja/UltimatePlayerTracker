This subdirectory contains the implementation of YOLOv4 used for object detection and feature extraction in the context of DeepSORT-based object tracking.

## Purpose

YOLOv4 serves as the detection backbone for the DeepSORT framework. It provides bounding boxes and deep feature embeddings that enable accurate and efficient multi-object tracking.

## Structure

- **`backbone.py`**: Implements the feature extraction backbone of YOLOv4.
- **`common.py`**: Houses shared utility functions and reusable components.
- **`config.py`**: Contains configuration settings such as model parameters and dataset paths.
- **`dataset.py`**: Manages dataset loading and preprocessing for training and evaluation.
- **`utils.py`**: Provides helper functions for tasks like plotting, logging, and non-max suppression.
- **`yolov4.py`**: The main implementation file for the YOLOv4 model, including architecture, training, and inference.

## Role in the Project

This directory focuses on detection and feature extraction for the broader DeepSORT project. After YOLOv4 generates detections and embeddings, the tracking logic in DeepSORT assigns unique IDs to detected objects, maintaining consistency across frames.

## Configuration and Usage

Refer to `config.py` to set model parameters, input/output paths, and other settings. Integrate this module with the tracking pipeline in the main project directory.
