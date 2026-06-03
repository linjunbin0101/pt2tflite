# pt2tflite

Convert Ultralytics YOLO PyTorch models (.pt) to TensorFlow Lite (.tflite).

Conversion pipeline: **PyTorch → ONNX → TensorFlow SavedModel → TFLite**

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | Linux (recommended), macOS, Windows |
| Python | 3.10 ~ 3.12 |
| Disk Space | ≥ 2GB (venv + dependencies) |

## Environment Setup

### 1. Create Virtual Environment

```bash
cd /home/lins/Desktop/pt2tflite
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install \
  torch \
  onnx onnxruntime \
  ultralytics \
  tensorflow \
  onnx2tf \
  onnx_graphsurgeon \
  sng4onnx \
  ai-edge-litert \
  "tf_keras<=2.19.0"
```

If downloads are slow, use a mirror:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch \
  onnx onnxruntime \
  ultralytics \
  tensorflow \
  onnx2tf \
  onnx_graphsurgeon \
  sng4onnx \
  ai-edge-litert \
  "tf_keras<=2.19.0"
```

Dependency overview:

| Package | Purpose |
|---------|---------|
| `torch` | Load .pt model, export ONNX |
| `onnx`, `onnxruntime` | ONNX processing & validation |
| `ultralytics` | YOLO model loading & export |
| `tensorflow` | TF SavedModel → TFLite conversion |
| `onnx2tf` | ONNX → TensorFlow SavedModel conversion |
| `tf_keras<=2.19.0` | Keras compatibility layer for ultralytics |
| `ai-edge-litert` | LiteRT runtime for onnx2tf |
| `onnx_graphsurgeon`, `sng4onnx` | Graph optimization tools for onnx2tf |

### 3. Verify Installation

```bash
source .venv/bin/activate
python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import tensorflow as tf; print('tf:', tf.__version__)"
python3 -c "from ultralytics import YOLO; print('ultralytics: OK')"
python3 -c "import onnx2tf; print('onnx2tf: OK')"
```

### 4. NumPy 2.x Compatibility (if needed)

If you encounter `ValueError: This file contains pickled (object) data`, patch onnx2tf's `np.load` calls:

Change `np.load(...)` to `np.load(..., allow_pickle=True)` in:

- `.venv/lib/python3.12/site-packages/onnx2tf/utils/common_functions.py` — lines 3828, 4020, 4284
- `.venv/lib/python3.12/site-packages/onnx2tf/onnx2tf.py` — line 1790

## Usage

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib NUMPY_ALLOW_PICKLE=1 python3 pt2tflite.py
```

The script automatically finds all `.pt` / `.pth` files in the project root, converts them one by one to `.tflite`, and cleans up intermediate files.

## Example Output

```
Found 1 PyTorch model file(s):
  - best.pt (42.7 MB)

Exporting TFLite (PyTorch → ONNX → TF SavedModel → TFLite)...
...
✅ Conversion complete: best.pt -> best.tflite
   Model size: 42.7 MB
```

## Output Files

| File | Description |
|------|-------------|
| `best.pt` | Original PyTorch model |
| `best.tflite` | Converted TFLite model (float32) |

Intermediate files (auto-cleaned):

| File/Directory | Description |
|----------------|-------------|
| `best.onnx` | Intermediate ONNX format |
| `best_saved_model/` | Intermediate TF SavedModel |
| `calibration_image_sample_data_*.npy*` | onnx2tf calibration data |

## Model Info

- **Type**: Ultralytics YOLO object detection model
- **Input**: 1 × 3 × 640 × 640 (BCHW)
- **Output**: 1 × 5 × 8400 (bboxes + confidence + classes)
- **TFLite input format**: NHWC (1 × 640 × 640 × 3)

## Notes

1. The first run downloads calibration data (`calibration_image_sample_data_*.npy.zip`, ~1.1MB) from GitHub
2. `NUMPY_ALLOW_PICKLE=1` enables pickle loading for NumPy 2.x compatibility
3. `MPLCONFIGDIR=/tmp/matplotlib` works around unwritable matplotlib cache directory (common in containers)
4. Conversion time depends on model size and CPU; a 42MB YOLO model takes about **40 seconds**