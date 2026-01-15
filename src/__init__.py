"""
InsiteSlideFromPDF - PDF/Image to PowerPoint Converter

Main source package containing all application modules.
"""

from .pdf_processor import PDFProcessor, extract_pdf_content
from .image_processor import ImageProcessor, process_image_file
from .color_detector import ColorDetector, detect_colored_objects
from .pptx_generator import PPTXGenerator, create_presentation_from_images
from .converter import Converter, convert_file
from .settings import get_settings, save_settings

__all__ = [
    'PDFProcessor',
    'extract_pdf_content',
    'ImageProcessor',
    'process_image_file',
    'ColorDetector',
    'detect_colored_objects',
    'PPTXGenerator',
    'create_presentation_from_images',
    'Converter',
    'convert_file',
    'get_settings',
    'save_settings',
]

__version__ = '1.0.0'
