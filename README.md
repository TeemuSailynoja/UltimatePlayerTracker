# UltimatePlayerTracker

Tracking of players on the field for Ultimate. 
This project is based on a more general workflow of object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is an algorithm that uses deep convolutional neural networks for object detections. These detections are then fed to into DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

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
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip (Legacy)
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 for making the player (and in the future disc) detections. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.

Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )

Place the yolov4.weights file into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, the weights are already included in the 'data' folder.

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.
```bash
# Convert darknet weights to tensorflow model
pixi run save-model

# Run yolov4 deep sort object tracker on video
pixi run run-tracker

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
pixi run python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4

# Or use traditional Python calls
python object_tracker.py --video ./data/video/demo.mp4 --output ./outputs/demo.avi --model yolov4
```
The ``--output`` flag saves the resulting video of the object tracker to the specified location.

## Running the Tracker with YOLOv4-Tiny
The following commands allow you to run the yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!
```
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
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

### Aknowledgements 

   I want to thank TheAiGuy for his repository introducing the deepsort algorithm for object tracking using Yolov4. I also thank hunglc007 and nwojke whose work that repository is based upon:
  * [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
