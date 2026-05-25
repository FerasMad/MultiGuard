"""Forensic Image Detector — binary AI-generated vs real classifier on GenImage.

Doctor's brief: docs/doctor-briefs/Forensic_Image_Detector_En.pdf

Two approaches implemented under this package:
- Approach 1 (RGB + Fourier Mask): fine-tune chandlerbing65nm/FakeImageDetection
- Approach 2 (DCT Frequency Domain): torchvision ResNet50 + 1-ch conv1 + two-phase training

Output artifacts: forensic_rgb_model.pth, forensic_dct_model.pth, dct_stats.json
"""
from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

# Forensic phase root (so submodules can find data/, outputs/, configs/ relative to it)
FORENSIC_ROOT = Path(__file__).resolve().parents[2]  # phases/forensic/
