# pt2tflite

Convert Ultralytics YOLO PyTorch models (.pt) to TensorFlow Lite (.tflite).

Conversion pipeline: **PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite**

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | Linux, macOS, Windows |
| Python | 3.10 ~ 3.12 |
| Disk Space | >= 2GB (venv + dependencies) |

## Environment Setup

### 1. Create Virtual Environment

```bash
cd /home/lins/Desktop/pt2tflite
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
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

For GUI, also install PyQt6:

```bash
pip install PyQt6
```

Dependency overview:

| Package | Purpose |
|---------|---------|
| `torch` | Load .pt model, export ONNX |
| `onnx`, `onnxruntime` | ONNX processing & validation |
| `ultralytics` | YOLO model loading & export |
| `tensorflow` | TF SavedModel -> TFLite conversion |
| `onnx2tf` | ONNX -> TensorFlow SavedModel conversion |
| `tf_keras<=2.19.0` | Keras compatibility layer for ultralytics |
| `ai-edge-litert` | LiteRT runtime for onnx2tf |
| `onnx_graphsurgeon`, `sng4onnx` | Graph optimization tools for onnx2tf |
| `PyQt6` | GUI application (optional) |

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

- `.venv/lib/python3.12/site-packages/onnx2tf/utils/common_functions.py` - lines 3828, 4020, 4284
- `.venv/lib/python3.12/site-packages/onnx2tf/onnx2tf.py` - line 1790

## Usage (CLI)

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib NUMPY_ALLOW_PICKLE=1 python3 pt2tflite.py
```

The script automatically finds all `.pt` / `.pth` files in the project root and converts each one.

### Example Output

```
Found 1 PyTorch model file(s):
  - best.pt (42.7 MB)

============================================================
Converting: best.pt
  Source: /home/lins/Desktop/pt2tflite/best.pt
============================================================

Exporting TFLite (PyTorch -> ONNX -> TF SavedModel -> TFLite)...
...
Conversion complete: best.pt -> best.tflite
  Source: /home/lins/Desktop/pt2tflite/best.pt
  Output: /home/lins/Desktop/pt2tflite/best.tflite
  Size: 42.7 MB
  Elapsed: 40.2s
```

## GUI Application

A PyQt6-based graphical interface provides a more user-friendly experience:

```bash
source .venv/bin/activate
python3 pt2tflite_gui.py
```

### Features

| Feature | Description |
|---------|-------------|
| File Selection | Add .pt/.pth files via dialog, clear list |
| Input Size | Adjustable (default 640, step 32) |
| ONNX Cleanup | Option to keep or delete intermediate .onnx |
| Real-time Log | Captures all conversion output (ANSI-stripped) |
| Download Progress | Shows tqdm-style progress bars for calibration data |
| Busy Animation | Progress bar animates during long operations |
| Elapsed Timer | Live `Elapsed: M:SS` counter during conversion |
| File Paths | Shows full source and output paths |
| Clear Log | One-click log cleanup |
| Cancel | Stop running conversion |

### Screenshot Layout

```
+-----------------------------+----------------------------------+
| Model Files                 | Conversion Log                   |
| [Add .pt/.pth Files]        | +----------------------------+   |
| Selected files:             | | [Clear Log]                |   |
| best.pt (42.7 MB)           | | ========================== |   |
| [Clear List]                | | Converting: best.pt        |   |
|                             | |   Source: /home/.../best.pt|   |
| Settings                    | | ...                        |   |
| Input size: [640]           | | Conversion complete: ...   |   |
| [x] Delete intermediate .onnx| |   Source: /home/.../best.pt|   |
|                             | |   Output: /home/.../best...|   |
| [Start Conversion]          | |   Size: 42.7 MB            |   |
| [Cancel]                    | |   Elapsed: 40.2s           |   |
| [========== Progress =====]  | +----------------------------+   |
| Elapsed: 0:40               |                                  |
+-----------------------------+----------------------------------+
```

## Output Files

| File | Description |
|------|-------------|
| `best.pt` | Original PyTorch model |
| `best.tflite` | Converted TFLite model (float32) |

Intermediate files (auto-cleaned after conversion):

| File/Directory | Description |
|----------------|-------------|
| `best.onnx` | Intermediate ONNX format (kept if option unchecked) |
| `best_saved_model/` | Intermediate TF SavedModel |

Calibration data (`calibration_image_sample_data_*.npy.zip`, ~1.1 MB) is **cached** - downloaded once and reused across all conversions.

## Model Info

- **Type**: Ultralytics YOLO object detection model
- **Input**: 1 x 3 x 640 x 640 (BCHW)
- **Output**: 1 x 5 x 8400 (bboxes + confidence + classes)
- **TFLite input format**: NHWC (1 x 640 x 640 x 3)

## Notes

1. The first run downloads calibration data (`calibration_image_sample_data_*.npy.zip`, ~1.1MB) from GitHub - subsequent runs reuse it
2. `NUMPY_ALLOW_PICKLE=1` enables pickle loading for NumPy 2.x compatibility
3. `MPLCONFIGDIR=/tmp/matplotlib` works around unwritable matplotlib cache directory (common in containers)
4. Conversion time depends on model size and CPU; a 42MB YOLO model takes about **40 seconds**
5. The slowest step is ONNX -> TF SavedModel conversion (onnx2tf), which does not output intermediate progress
6. On Windows, use `set MPLOCONFIGDIR=%TEMP%\matplotlib` and `set NUMPY_ALLOW_PICKLE=1` instead