#!/usr/bin/env python3
"""
Convert .pt PyTorch models to .tflite format.

Scans the project root for all .pt/.pth files and converts each one.
Usage: python3 pt2tflite.py
"""

import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def find_pt_files():
    """Find all .pt and .pth files in the root directory."""
    pt_files = []
    for ext in ('*.pt', '*.pth'):
        pt_files.extend(ROOT_DIR.glob(ext))
    return pt_files


def convert_pt_to_tflite(pt_path: Path):
    """Convert a single .pt file to .tflite."""
    print(f"\n{'='*60}")
    print(f"Converting: {pt_path.name}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    # Use ultralytics built-in TFLite export
    # Pipeline: PyTorch -> ONNX -> TF SavedModel -> TFLite
    print("\nExporting TFLite (PyTorch -> ONNX -> TF SavedModel -> TFLite)...")
    output_path = model.export(format='tflite', imgsz=640)

    # Copy tflite from saved_model subdir to root
    tflite_in_dir = Path(output_path)
    if tflite_in_dir.exists():
        target_path = pt_path.with_suffix('.tflite')
        shutil.copy2(tflite_in_dir, target_path)
        print(f"\n✅ Conversion complete: {pt_path.name} -> {target_path.name}")
        print(f"   Model size: {target_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n⚠️ TFLite file not found: {output_path}")

    # Cleanup intermediate files
    saved_model_dir = pt_path.parent / f"{pt_path.stem}_saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    onnx_file = pt_path.with_suffix('.onnx')
    if onnx_file.exists():
        onnx_file.unlink()
    for calib_file in ROOT_DIR.glob("calibration_image_sample_data_*.npy*"):
        calib_file.unlink(missing_ok=True)

    return target_path if tflite_in_dir.exists() else None


def main():
    pt_files = find_pt_files()
    if not pt_files:
        print("No .pt or .pth files found")
        sys.exit(1)

    print(f"Found {len(pt_files)} PyTorch model file(s):")
    for f in pt_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    for pt_file in pt_files:
        try:
            convert_pt_to_tflite(pt_file)
        except Exception as e:
            print(f"\n❌ Conversion failed for {pt_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()