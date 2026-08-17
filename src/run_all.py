"""Run the complete VisGuard-Ur scoped prototype in order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(module: str, *args: str) -> None:
    command = [sys.executable, str(SRC / module), *args]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("generate_dataset.py")
    run("ocr_pipeline.py")
    run("train_detector.py")
    run("evaluate.py")
    print("\nVisGuard-Ur run complete. See results/ and dataset/.")
