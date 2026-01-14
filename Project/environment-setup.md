# Development Environment Setup Guide

This guide helps set up a clean development environment for migrating UltimatePlayerTracker from YOLOv4 to modern YOLO (YOLOv10/YOLOv8).

## Current Environment (To Be Replaced)

The existing project uses outdated dependencies:
- **TensorFlow 2.3.0** (July 2020)
- **OpenCV 4.1.1.26** (outdated)
- **Python 3.8-3.10** compatibility
- **Custom YOLOv4 implementation**

## New Environment Requirements

### Target Dependencies
```bash
# New modern stack (to be installed)
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.3
matplotlib>=3.4.3
```

### Removed Dependencies
```bash
# Legacy stack (to be removed)
tensorflow==2.3.0
opencv-python==4.1.*
```

## Step-by-Step Environment Setup

### 1. Create Virtual Environment

```bash
# Using conda (recommended for GPU support)
conda create -n yolov10-migration python=3.10
conda activate yolov10-migration

# Or using venv
python -m venv venv_yolov10
source venv_yolov10/bin/activate  # Linux/Mac
# venv_yolov10\Scripts\activate  # Windows
```

### 2. Install New Dependencies

#### Option A: Using Poetry (Recommended)
```bash
# Navigate to project root
cd /path/to/UltimatePlayerTracker

# Update pyproject.toml (see Task 2.1)
poetry install
poetry shell  # Activate virtual environment
```

#### Option B: Using Pip
```bash
# Install PyTorch (CUDA 11.8 - adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics and other dependencies
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install tqdm>=4.62.3
pip install matplotlib>=3.4.3
```

### 3. Verify Installation

#### Test PyTorch Installation
```python
# test_pytorch.py
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

#### Test Ultralytics Installation
```python
# test_ultralytics.py
from ultralytics import YOLO

# Test model loading (will download default model)
model = YOLO('yolov10n.pt')
print(f"YOLOv10 model loaded successfully")
print(f"Model info: {model.info()}")

# Test inference on dummy image
import numpy as np
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model(dummy_image)
print(f"Inference test completed")
print(f"Detected {len(results[0].boxes)} objects")
```

#### Test OpenCV Installation
```python
# test_opencv.py
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# Test basic operations
dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
resized = cv2.resize(dummy_image, (640, 640))
print(f"OpenCV resize test passed: {resized.shape}")
```

### 4. GPU Setup (if using NVIDIA GPU)

#### CUDA Toolkit Verification
```bash
# Check CUDA version
nvidia-smi

# Check if CUDA version matches PyTorch
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

#### CuDNN Verification
```python
# test_cudnn.py
import torch

if torch.backends.cudnn.is_available():
    print(f"CuDNN version: {torch.backends.cudnn.version()}")
    print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")
else:
    print("CuDNN not available")
```

### 5. Hardware-Specific Optimizations

#### For NVIDIA GPU Users
```bash
# Install TensorRT (optional, for deployment optimization)
# Download from NVIDIA developer portal
pip install tensorrt

# Test TensorRT availability
python -c "import tensorrt; print('TensorRT available')"
```

#### For Apple Silicon Users
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Test MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### For CPU-Only Users
```bash
# Install ONNX Runtime for CPU optimization
pip install onnxruntime

# Test ONNX Runtime
python -c "import onnxruntime; print('ONNX Runtime available')"
```

## Environment Variables

### Set for GPU Usage
```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Windows
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Set for Memory Optimization
```bash
# Limit PyTorch memory usage (adjust based on your GPU memory)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable memory pooling for better performance
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
```

## Development IDE Setup

### VS Code Extensions
- **Python** - Microsoft
- **PyTorch** - PyTorch team
- **Jupyter** - Microsoft
- **Remote Development** - Microsoft (if using remote GPU)

### PyCharm Setup
1. File → Settings → Project → Python Interpreter
2. Add new interpreter (point to virtual environment)
3. Install PyTorch plugin if desired

## Validation Checklist

### Basic Functionality Tests
- [ ] Python 3.8+ installed and working
- [ ] PyTorch 2.0+ loads without errors
- [ ] CUDA (if GPU) available and functional
- [ ] Ultralytics YOLO loads models successfully
- [ ] OpenCV operations work correctly
- [ ] Basic inference pipeline runs without errors

### Performance Validation
- [ ] GPU memory allocation works correctly
- [ ] CPU inference performance is acceptable
- [ ] Multi-threading (if applicable) works

### Integration Tests
- [ ] Can load sample video frames
- [ ] Can run basic object detection
- [ ] Can process detection results
- [ ] No import errors or missing dependencies

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch
```bash
# Symptoms: CUDA out of memory, import errors
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV Import Errors
```bash
# Symptoms: cv2 import fails
# Solution: Install opencv-python instead of opencv-contrib-python
pip uninstall opencv-contrib-python
pip install opencv-python>=4.8.0
```

#### 3. Ultralytics Model Download Issues
```python
# Symptoms: Model download fails or times out
# Solution: Download models manually or set cache directory
import os
os.environ['ULTRALYTICS_CACHE'] = '/path/to/cache/directory'
```

#### 4. Memory Issues on Limited Hardware
```python
# Solutions for low-memory systems:
# 1. Use smaller model variants
model = YOLO('yolov10n.pt')  # Use 'n' variant

# 2. Reduce batch size
results = model(image, batch=1)

# 3. Use CPU fallback if needed
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 4000000000:
    device = 'cpu'
else:
    device = 'cuda'
```

## Next Steps

After completing environment setup:

1. **Run all validation tests** above
2. **Document any issues** in Project/environment-issues.md
3. **Begin Phase 2** of the migration plan
4. **Update progress tracker** with environment setup completion

## Support Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Project Issues**: Create GitHub issues for blocking problems

---

*Last Updated: January 15, 2026*