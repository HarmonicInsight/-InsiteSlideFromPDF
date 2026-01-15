"""
Converter Module
Coordinates the conversion from PDF/Images to PowerPoint.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Union
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

from .pdf_processor import PDFProcessor, PageContent
from .image_processor import ImageProcessor, OCRResult, DetectedShape
from .color_detector import ColorDetector, ColoredObject, ColorType
from .pptx_generator import (
    PPTXGenerator, SlideContent, TextElement, ShapeElement, ImageElement, ShapeType
)
from .settings import get_settings, AppSettings


@dataclass
class ConversionProgress:
    """Progress information for conversion."""
    current_step: int
    total_steps: int
    step_name: str
    percentage: float


ProgressCallback = Callable[[ConversionProgress], None]


class Converter:
    """Main converter class for PDF/Image to PowerPoint conversion."""

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    SUPPORTED_PDF_EXTENSION = '.pdf'

    def __init__(self, settings: Optional[AppSettings] = None):
        """
        Initialize converter.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self.settings = settings if settings else get_settings()
        self.image_processor = ImageProcessor(self.settings.tesseract_path or None)
        self.color_detector = ColorDetector()

        # Set color ranges from settings
        from .settings import get_settings_manager
        color_ranges = get_settings_manager().get_color_ranges()
        if color_ranges:
            self.color_detector.set_color_ranges(color_ranges)

    def is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == self.SUPPORTED_PDF_EXTENSION

    def is_image(self, file_path: str) -> bool:
        """Check if file is a supported image."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS

    def convert(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> bool:
        """
        Convert input file to PowerPoint.

        Args:
            input_path: Path to input file (PDF or image).
            output_path: Path to output PPTX file.
            progress_callback: Optional progress callback.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.is_pdf(input_path):
                return self._convert_pdf(input_path, output_path, progress_callback)
            elif self.is_image(input_path):
                return self._convert_image(input_path, output_path, progress_callback)
            else:
                print(f"Unsupported file type: {input_path}")
                return False
        except Exception as e:
            print(f"Conversion error: {e}")
            return False

    def _report_progress(
        self,
        callback: Optional[ProgressCallback],
        current: int,
        total: int,
        step_name: str
    ):
        """Report progress to callback."""
        if callback:
            callback(ConversionProgress(
                current_step=current,
                total_steps=total,
                step_name=step_name,
                percentage=(current / total) * 100 if total > 0 else 0
            ))

    def _convert_pdf(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback]
    ) -> bool:
        """Convert PDF to PowerPoint."""
        processor = PDFProcessor(input_path)

        if not processor.open():
            return False

        try:
            generator = PPTXGenerator()
            page_count = processor.get_page_count()
            total_steps = page_count * 3  # Extract, process, generate per page

            for page_num in range(page_count):
                step_base = page_num * 3

                # Step 1: Extract content
                self._report_progress(
                    progress_callback,
                    step_base + 1, total_steps,
                    f"Extracting page {page_num + 1}/{page_count}"
                )

                page_content = processor.extract_page_content(page_num)
                if not page_content:
                    continue

                # Render page as image for color detection
                page_image = processor.render_page_as_image(page_num, self.settings.processing.dpi)

                # Step 2: Process content
                self._report_progress(
                    progress_callback,
                    step_base + 2, total_steps,
                    f"Processing page {page_num + 1}/{page_count}"
                )

                # Detect colored objects if enabled
                colored_objects = []
                if self.settings.processing.detect_colors and page_image:
                    cv_image = self.image_processor.pil_to_cv2(page_image)
                    colored_objects = self.color_detector.detect_colors(
                        cv_image,
                        min_area=self.settings.processing.min_color_area
                    )

                # Step 3: Generate slide
                self._report_progress(
                    progress_callback,
                    step_base + 3, total_steps,
                    f"Generating slide {page_num + 1}/{page_count}"
                )

                self._create_slide_from_page(
                    generator,
                    page_content,
                    page_image,
                    colored_objects
                )

            # Save presentation
            generator.save(output_path)
            return True

        finally:
            processor.close()

    def _convert_image(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback]
    ) -> bool:
        """Convert image to PowerPoint."""
        total_steps = 4

        # Step 1: Load image
        self._report_progress(progress_callback, 1, total_steps, "Loading image")

        image = self.image_processor.load_image(input_path)
        if image is None:
            return False

        pil_image = self.image_processor.cv2_to_pil(image)
        height, width = image.shape[:2]

        # Step 2: Extract text with OCR
        self._report_progress(progress_callback, 2, total_steps, "Extracting text (OCR)")

        ocr_results = []
        if self.settings.processing.ocr_language:
            ocr_results = self.image_processor.extract_text_ocr(
                image,
                lang=self.settings.processing.ocr_language
            )

        # Step 3: Detect colors and shapes
        self._report_progress(progress_callback, 3, total_steps, "Detecting objects")

        colored_objects = []
        if self.settings.processing.detect_colors:
            colored_objects = self.color_detector.detect_colors(
                image,
                min_area=self.settings.processing.min_color_area
            )

        shapes = []
        if self.settings.processing.detect_shapes:
            shapes = self.image_processor.detect_shapes(image)

        # Step 4: Generate PowerPoint
        self._report_progress(progress_callback, 4, total_steps, "Generating PowerPoint")

        generator = PPTXGenerator()
        self._create_slide_from_image(
            generator,
            pil_image,
            width, height,
            ocr_results,
            colored_objects,
            shapes
        )

        generator.save(output_path)
        return True

    def _create_slide_from_page(
        self,
        generator: PPTXGenerator,
        page_content: PageContent,
        page_image: Optional[Image.Image],
        colored_objects: List[ColoredObject]
    ):
        """Create a slide from PDF page content."""
        elements = []
        source_width = page_content.width
        source_height = page_content.height

        # Add page image as background if available
        if page_image:
            img_width, img_height = page_image.size
            elements.append(ImageElement(
                x=0, y=0,
                width=generator.SLIDE_WIDTH_INCHES,
                height=generator.SLIDE_HEIGHT_INCHES,
                image=page_image
            ))

        # Add colored objects as shapes
        for obj in colored_objects:
            x, y, w, h = obj.bbox
            # Convert from image coordinates to PDF coordinates
            if page_image:
                img_width, img_height = page_image.size
                scale_x = source_width / img_width
                scale_y = source_height / img_height
                x *= scale_x
                y *= scale_y
                w *= scale_x
                h *= scale_y

            x_in, y_in = generator.convert_position_to_inches(
                x, y, source_width, source_height
            )
            w_in, h_in = generator.convert_size_to_inches(
                w, h, source_width, source_height
            )

            # Determine shape type based on color
            shape_type = self._get_shape_type_for_color(obj.color_type)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.2),
                height=max(h_in, 0.2),
                shape_type=shape_type,
                fill_color=obj.rgb_color,
                line_color=obj.rgb_color,
                line_width=2
            ))

        content = SlideContent(elements=elements)
        generator.add_slide_with_content(content)

    def _create_slide_from_image(
        self,
        generator: PPTXGenerator,
        image: Image.Image,
        width: int,
        height: int,
        ocr_results: List[OCRResult],
        colored_objects: List[ColoredObject],
        shapes: List[DetectedShape]
    ):
        """Create a slide from image content."""
        elements = []

        # Add image as background
        elements.append(ImageElement(
            x=0, y=0,
            width=generator.SLIDE_WIDTH_INCHES,
            height=generator.SLIDE_HEIGHT_INCHES,
            image=image
        ))

        # Add colored objects as shapes (semi-transparent overlay)
        for obj in colored_objects:
            x, y, w, h = obj.bbox
            x_in, y_in = generator.convert_position_to_inches(x, y, width, height)
            w_in, h_in = generator.convert_size_to_inches(w, h, width, height)

            shape_type = self._get_shape_type_for_color(obj.color_type)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.2),
                height=max(h_in, 0.2),
                shape_type=shape_type,
                fill_color=None,  # Transparent fill
                line_color=obj.rgb_color,
                line_width=3
            ))

        content = SlideContent(elements=elements)
        generator.add_slide_with_content(content)

    def _get_shape_type_for_color(self, color_type: ColorType) -> ShapeType:
        """Get appropriate shape type for a color type."""
        # Map semantic color types to shapes
        shape_map = {
            ColorType.RED: ShapeType.DIAMOND,      # Warning
            ColorType.YELLOW: ShapeType.TRIANGLE,  # Caution
            ColorType.GREEN: ShapeType.OVAL,       # Success
            ColorType.BLUE: ShapeType.RECTANGLE,   # Info
            ColorType.ORANGE: ShapeType.PENTAGON,  # Warning (mild)
            ColorType.PURPLE: ShapeType.STAR,      # Special
        }
        return shape_map.get(color_type, ShapeType.RECTANGLE)


def convert_file(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None
) -> bool:
    """
    Convenience function to convert a file.

    Args:
        input_path: Path to input file.
        output_path: Path to output PPTX file.
        progress_callback: Optional progress callback.

    Returns:
        True if successful.
    """
    converter = Converter()
    return converter.convert(input_path, output_path, progress_callback)
