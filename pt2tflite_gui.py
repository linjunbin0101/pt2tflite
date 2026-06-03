#!/usr/bin/env python3
"""
pt2tflite GUI - Convert PyTorch .pt models to TFLite with a Qt interface.
"""

import io
import os
import re
import sys
import time
import logging
import shutil
import traceback
from pathlib import Path

# Strip ANSI escape codes before displaying in GUI
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")

def _clean(text):
    return _ANSI_RE.sub("", text)

from PyQt6.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox, QCheckBox,
    QSpinBox, QFormLayout, QGroupBox, QSplitter,
)


class SignalStream(io.StringIO):
    """Redirect stdout to Qt signal for real-time log display."""
    def __init__(self, signal_emit, original_stdout):
        super().__init__()
        self.signal_emit = signal_emit
        self.original_stdout = original_stdout
        self._line_buffer = ""

    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        if not text:
            return
        self._line_buffer += text
        # Handle carriage returns (tqdm/download bars use \r to update in-place)
        if "\r" in self._line_buffer:
            lines = self._line_buffer.split("\r")
            for line in lines[:-1]:
                stripped = line.strip()
                if stripped:
                    self.signal_emit(_clean(stripped))
            self._line_buffer = lines[-1]
        elif "\n" in self._line_buffer:
            lines = self._line_buffer.split("\n")
            for line in lines[:-1]:
                stripped = line.strip()
                if stripped:
                    self.signal_emit(_clean(stripped))
            self._line_buffer = lines[-1]

    def flush(self):
        self.original_stdout.flush()

    def close(self):
        stripped = self._line_buffer.strip()
        if stripped:
            self.signal_emit(_clean(stripped))


class ConvertWorker(QThread):
    """Runs conversion in a background thread to keep UI responsive."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    busy_signal = pyqtSignal(bool)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pt_paths, imgsz=640, clean_onnx=True):
        super().__init__()
        self.pt_paths = pt_paths
        self.imgsz = imgsz
        self.clean_onnx = clean_onnx
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self.pt_paths)
        start_time = time.time()
        for idx, pt_path in enumerate(self.pt_paths):
            if self._cancel:
                self.finished_signal.emit(False, "Cancelled by user")
                return

            self.log_signal.emit(f"\n{'='*60}")
            self.log_signal.emit(f"Converting: {pt_path.name}")
            self.log_signal.emit(f"  Source: {pt_path}")
            self.log_signal.emit(f"{'='*60}")

            try:
                from ultralytics import YOLO
                from ultralytics.utils import LOGGER as ultralytics_logger

                model = YOLO(str(pt_path))
                self.log_signal.emit("Exporting TFLite (PyTorch -> ONNX -> TF SavedModel -> TFLite)...")

                # Set env vars for compatibility
                os.environ.setdefault("NUMPY_ALLOW_PICKLE", "1")
                os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

                # --- Intercept all output at BOTH levels ---

                # 1. Replace logging handlers (ultralytics + root logger)
                original_ultra_handlers = ultralytics_logger.handlers[:]
                original_root_handlers = logging.getLogger().handlers[:]
                ultralytics_logger.handlers.clear()
                logging.getLogger().handlers.clear()

                class QtLogHandler(logging.Handler):
                    def __init__(self, signal_fn):
                        super().__init__()
                        self.signal_fn = signal_fn
                    def emit(self, record):
                        msg = self.format(record)
                        if msg:
                            self.signal_fn(_clean(msg))

                qt_handler = QtLogHandler(self.log_signal.emit)
                qt_handler.setFormatter(logging.Formatter("%(message)s"))
                ultralytics_logger.addHandler(qt_handler)

                # 2. Redirect stdout (for print() calls)
                original_stdout = sys.stdout
                sys.stdout = SignalStream(self.log_signal.emit, original_stdout)

                self.busy_signal.emit(True)
                try:
                    output_path = model.export(format="tflite", imgsz=self.imgsz)
                finally:
                    self.busy_signal.emit(False)
                    # Restore stdout
                    sys.stdout = original_stdout
                    # Restore logging handlers
                    ultralytics_logger.handlers = original_ultra_handlers
                    logging.getLogger().handlers = original_root_handlers

                tflite_in_dir = Path(output_path)
                if tflite_in_dir.exists():
                    target_path = pt_path.with_suffix('.tflite')
                    shutil.copy2(tflite_in_dir, target_path)
                    size_mb = target_path.stat().st_size / 1024 / 1024
                    elapsed = time.time() - start_time
                    self.log_signal.emit(f"\nConversion complete: {pt_path.name} -> {target_path.name}")
                    self.log_signal.emit(f"  Source: {pt_path}")
                    self.log_signal.emit(f"  Output: {target_path}")
                    self.log_signal.emit(f"  Size: {size_mb:.1f} MB")
                    self.log_signal.emit(f"  Elapsed: {elapsed:.1f}s")
                else:
                    self.log_signal.emit(f"TFLite file not found: {output_path}")

                # Cleanup
                saved_model_dir = pt_path.parent / f"{pt_path.stem}_saved_model"
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                if self.clean_onnx:
                    onnx_file = pt_path.with_suffix('.onnx')
                    if onnx_file.exists():
                        onnx_file.unlink()
            except Exception as e:
                self.log_signal.emit(f"\nConversion failed: {e}")
                traceback.print_exc()
                self.finished_signal.emit(False, str(e))
                return

            self.progress_signal.emit(int((idx + 1) / total * 100))

        self.finished_signal.emit(True, "All conversions completed!")


class Pt2TfliteApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pt2tflite - YOLO to TFLite Converter")
        self.setMinimumSize(800, 600)

        self.worker = None
        self.selected_files = []
        self._start_time = 0
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ======== Left Panel ========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)

        # File selection
        file_group = QGroupBox("Model Files")
        file_layout = QVBoxLayout(file_group)
        add_btn = QPushButton("Add .pt / .pth Files")
        add_btn.clicked.connect(self._add_files)
        file_layout.addWidget(add_btn)

        self.file_list = QListWidget()
        file_layout.addWidget(QLabel("Selected files:"))
        file_layout.addWidget(self.file_list)

        clear_btn = QPushButton("Clear List")
        clear_btn.clicked.connect(self._clear_files)
        file_layout.addWidget(clear_btn)

        left_layout.addWidget(file_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        settings_layout.addRow("Input size:", self.imgsz_spin)

        self.clean_onnx_cb = QCheckBox("Delete intermediate .onnx")
        self.clean_onnx_cb.setChecked(True)
        settings_layout.addRow(self.clean_onnx_cb)

        left_layout.addWidget(settings_group)

        # Control buttons
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self._start_conversion)
        left_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_conversion)
        left_layout.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)

        self.elapsed_label = QLabel("Elapsed: --:--")
        left_layout.addWidget(self.elapsed_label)

        left_layout.addStretch()

        # ======== Right Panel ========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        log_group = QGroupBox("Conversion Log")
        log_layout = QVBoxLayout(log_group)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Conversion Log"))
        log_header.addStretch()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self._clear_log)
        log_header.addWidget(clear_log_btn)
        log_layout.addLayout(log_header)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Courier New")
        log_layout.addWidget(self.log_output)
        right_layout.addWidget(log_group)

        # ======== Splitter ========
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 450])
        main_layout.addWidget(splitter)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PyTorch model files", "",
            "PyTorch Models (*.pt *.pth);;All Files (*)"
        )
        for f in files:
            path = Path(f)
            if str(path) not in self.selected_files:
                self.selected_files.append(str(path))
                item = QListWidgetItem(f"{path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
                item.setToolTip(str(path))
                self.file_list.addItem(item)

    def _clear_files(self):
        self.file_list.clear()
        self.selected_files.clear()

    def _set_busy(self, busy):
        """Toggle progress bar between indeterminate (busy) and normal mode."""
        if busy:
            self.progress_bar.setRange(0, 0)  # indeterminate: bouncing bar
            self.progress_bar.setFormat("Converting...")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setFormat("%p%")

    def _start_conversion(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please add at least one .pt file.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Conversion is already running.")
            return

        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.elapsed_label.setText("Elapsed: 0:00")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._start_time = time.time()
        self._elapsed_timer.start(1000)

        pt_paths = [Path(f) for f in self.selected_files]
        self.worker = ConvertWorker(
            pt_paths,
            imgsz=self.imgsz_spin.value(),
            clean_onnx=self.clean_onnx_cb.isChecked(),
        )
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.busy_signal.connect(self._set_busy)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _cancel_conversion(self):
        if self.worker:
            self.worker.cancel()
            self._append_log("\nCancelling...")

    def _on_finished(self, success, message):
        self._elapsed_timer.stop()
        self._update_elapsed()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._append_log(f"\n{message}")
        if success:
            self.progress_bar.setValue(100)
        else:
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")

    def _clear_log(self):
        self.log_output.clear()

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        mins, secs = elapsed // 60, elapsed % 60
        self.elapsed_label.setText(f"Elapsed: {mins}:{secs:02d}")

    def _append_log(self, text):
        self.log_output.append(text)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("pt2tflite")
    window = Pt2TfliteApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()