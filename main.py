#!/usr/bin/env python3
"""
InsiteSlideFromPDF - PDF/Image to PowerPoint Converter

A Windows desktop application that converts PDF documents and images
to PowerPoint presentations with color object detection.

Features:
- PDF text and image extraction using PyMuPDF
- Image OCR using Tesseract
- Color detection using OpenCV
- PowerPoint generation using python-pptx
- User-friendly tkinter interface
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.main_window import MainWindow


def main():
    """Main entry point for the application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
