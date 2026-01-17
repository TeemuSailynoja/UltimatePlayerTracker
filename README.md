# UltimatePlayerTracker

Tracking of players on the field for Ultimate. 
This project is based on a more general workflow of object tracking implemented with YOLOv10, DeepSort, and PyTorch. YOLOv10 is an algorithm that uses deep convolutional neural networks for object detections. These detections are then fed to into DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

## Work in progress
This project is currently at a proof of consept stage. The algorithm is more tuned for tracking pedestrians and would benefit from finetuning to the players actively making fakes. Also, we could achieve a much better performance of staying on playes, if we leverage the fact thast there are always sa set number of players on field for any given point.

Some first steps for the project could be
  - Obtaining automatic video processing by recognising where a point ends and another starts.
  - Automated labelling for points; offence, defence, break, etc.
  - Statistics that also contain information on the movements of off-disc players -- most current apps for statistics tracking focus on the movement of the  disc, and completion raters of throwers.

## Getting Started
First, install project dependencies using Pixi (recommended) or the legacy methods.

### Pixi (Recommended - Modern)

```bash
# Install Pixi package manager (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies
pixi install

# Activate the environment
pixi shell
```

### Conda (Legacy)

```bash
# For YOLOv4 legacy support
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# For YOLOv4 GPU legacy support
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip (Legacy)
```bash
# For YOLOv4 legacy CPU support
pip install -r requirements.txt

# For YOLOv4 legacy GPU support
pip install -r requirements-gpu.txt

# For YOLOv10 modern support
pip install torch torchvision ultralytics opencv-python
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## YOLOv10 Model Setup

Our object tracker uses YOLOv10 for making the player (and in the future disc) detections. YOLOv10 provides state-of-the-art performance with multiple model variants optimized for different use cases.

### Available YOLOv10 Variants
- **YOLOv10n**: Nano - Fastest, suitable for real-time applications
- **YOLOv10s**: Small - Balanced speed and accuracy (default)
- **YOLOv10m**: Medium - Higher accuracy
- **YOLOv10b**: Base - Even better accuracy
- **YOLOv10l**: Large - High accuracy
- **YOLOv10x**: Extra Large - Best accuracy

### Automatic Model Download
The first time you run the tracker with YOLOv10, the model will be automatically downloaded. No manual download required.

### Manual Model Download (Optional)
If you prefer to download models manually:
```bash
# Download YOLOv10s (recommended)
wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10s.pt

# Place in project root or data/ folder
```

### Legacy YOLOv4 Support
YOLOv4 support is maintained for comparison and legacy purposes. See the "Running the Tracker with YOLOv4 (Legacy)" section below if needed.

## Running the Tracker with YOLOv10

### Quick Start
```bash
# Run with default YOLOv10s model
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov10

# Run with webcam (set video flag to 0)
pixi run python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov10
```

### Advanced Usage
```bash
# Use specific YOLOv10 variant
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10n --output ./outputs/fast.avi

# GPU acceleration
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10m --device cuda

# Custom confidence and IOU thresholds
pixi run python object_tracker.py --video ./data/video/demo.mp4 --model yolov10s --confidence 0.3 --iou 0.5
```

### Model Management
```bash
# Download and export specific variant
pixi run python save_model.py --model yolov10s --export_format onnx

# Download all variants
pixi run python save_model.py --download_all

# Benchmark performance
pixi run python save_model.py --benchmark
```

## Running the Tracker with YOLOv4 (Legacy)
Legacy YOLOv4 support is maintained for comparison purposes.
```bash
# Convert darknet weights to tensorflow model
pixi run python save_model.py --model yolov4

# Run yolov4 deep sort object tracker on video
pixi run python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov4
```
The ``--output`` flag saves the resulting video of the object tracker to the specified location.

## Performance Comparison

### YOLOv10 Variants Performance
- **YOLOv10n**: 60-120 FPS (fastest)
- **YOLOv10s**: 40-80 FPS (balanced)
- **YOLOv10m**: 25-50 FPS (accurate)
- **YOLOv10l**: 15-30 FPS (very accurate)
- **YOLOv10x**: 10-20 FPS (most accurate)

### YOLOv10 vs YOLOv4
- **Speed**: 3-4x faster inference
- **Accuracy**: 5-10% better mAP
- **Memory**: 50% reduction
- **Framework**: Modern PyTorch vs legacy TensorFlow

### Command Line Args Reference

```bash
save_model.py:
  --model: yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x (default: yolov10s)
  --export_format: onnx, torchscript, coreml, tflite (default: onnx)
  --download_all: download all model variants
  --benchmark: run performance benchmarking
     
object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/demo.mp4')
  --output: path to output video
    (default: None)
  --model: yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x, yolov4 (default: yolov10)
  --device: cpu, cuda, mps (auto-detected, default: auto)
  --confidence: confidence threshold
    (default: 0.25)
  --iou: iou threshold
    (default: 0.45)
  --output_df: path to output file containing player locations
    (default: None)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)

Legacy YOLOv4 args:
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: tf, trt, tflite (default: tf)
  --size: resize images to (default: 416)
  --tiny: use yolov4-tiny (default: false)
```

## Resulting Video
The resulting video will be saved to where you set the ``--output`` command line flag path to. You can also change the format of the video saved with the ``--output_format`` flag, by default it is set to AVI codec which is XVID.

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --output_df: path to output file containing the player locations from confirmed tracks in the video.
    (default: None)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --limits: only track detections inside the given limits. Still work in progress.
    (default: False)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

## Acknowledgements 

I want to thank the following open-source projects and contributors:

**Original Foundation:**
* [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) - TheAiGuy for the original YOLOv4 + DeepSort implementation
* [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) - hunglc007 for TensorFlow YOLOv4 implementation
* [Deep SORT Repository](https://github.com/nwojke/deep_sort) - nwojke for the original DeepSORT algorithm

**YOLOv10 Integration:**
* [YOLOv10](https://github.com/THU-MIG/yolov10) - THU-MIG lab for the state-of-the-art YOLOv10 implementation
* [Ultralytics](https://github.com/ultralytics/ultralytics) - For the YOLO ecosystem and PyTorch implementation

This project combines these excellent works to create a specialized Ultimate frisbee player tracking system.
