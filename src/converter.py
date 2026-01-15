"""
Converter Module
Coordinates the conversion from PDF/Images to PowerPoint.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

from .pdf_processor import PDFProcessor, PageContent, ExtractedText, ExtractedImage
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
            import traceback
            traceback.print_exc()
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
            total_steps = page_count * 3

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

                # Render page as high-quality image
                page_image = processor.render_page_as_image(page_num, self.settings.processing.dpi)

                # Step 2: Detect colored objects
                self._report_progress(
                    progress_callback,
                    step_base + 2, total_steps,
                    f"Processing page {page_num + 1}/{page_count}"
                )

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

        # 1. Add background image first (maintains original layout)
        if page_image:
            img_width, img_height = page_image.size
            elements.append(ImageElement(
                x=0, y=0,
                width=generator.SLIDE_WIDTH_INCHES,
                height=generator.SLIDE_HEIGHT_INCHES,
                image=page_image
            ))

            # Calculate scale for coordinate conversion
            img_scale_x = img_width / source_width
            img_scale_y = img_height / source_height
        else:
            img_scale_x = 1.0
            img_scale_y = 1.0

        # 2. Group texts into larger paragraph blocks
        text_paragraphs = self._group_texts_into_paragraphs(page_content.texts, source_height)

        # 3. Add text blocks as editable text boxes
        for paragraph in text_paragraphs:
            text_content = paragraph['text']
            bbox = paragraph['bbox']
            font_size = paragraph['font_size']

            x_in, y_in = generator.convert_position_to_inches(
                bbox[0], bbox[1], source_width, source_height
            )
            w_in, h_in = generator.convert_size_to_inches(
                bbox[2] - bbox[0], bbox[3] - bbox[1], source_width, source_height
            )

            # Ensure minimum size
            w_in = max(w_in, 1.0)
            h_in = max(h_in, 0.4)

            elements.append(TextElement(
                x=x_in, y=y_in,
                width=w_in, height=h_in,
                text=text_content,
                font_size=min(max(font_size, 8), 36),
                font_name=self.settings.font.default_font,
                font_color=(0, 0, 0)
            ))

        # 4. Add colored objects as shapes (on top of background)
        if page_image:
            img_width, img_height = page_image.size
            for obj in colored_objects:
                x, y, w, h = obj.bbox

                # Convert from image coordinates to slide coordinates
                x_in = (x / img_width) * generator.SLIDE_WIDTH_INCHES
                y_in = (y / img_height) * generator.SLIDE_HEIGHT_INCHES
                w_in = (w / img_width) * generator.SLIDE_WIDTH_INCHES
                h_in = (h / img_height) * generator.SLIDE_HEIGHT_INCHES

                # Skip very small objects
                if w_in < 0.3 or h_in < 0.3:
                    continue

                shape_type = self._detect_shape_type_from_contour(obj.contour)

                elements.append(ShapeElement(
                    x=x_in, y=y_in,
                    width=w_in,
                    height=h_in,
                    shape_type=shape_type,
                    fill_color=obj.rgb_color,
                    line_color=self._darken_color(obj.rgb_color),
                    line_width=1.5
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

        # 1. Add image as background
        elements.append(ImageElement(
            x=0, y=0,
            width=generator.SLIDE_WIDTH_INCHES,
            height=generator.SLIDE_HEIGHT_INCHES,
            image=image
        ))

        # 2. Group OCR results into paragraph blocks
        text_paragraphs = self._group_ocr_into_paragraphs(ocr_results, height)

        # 3. Add text blocks
        for paragraph in text_paragraphs:
            text_content = paragraph['text']
            bbox = paragraph['bbox']

            x_in = (bbox[0] / width) * generator.SLIDE_WIDTH_INCHES
            y_in = (bbox[1] / height) * generator.SLIDE_HEIGHT_INCHES
            w_in = ((bbox[2] - bbox[0]) / width) * generator.SLIDE_WIDTH_INCHES
            h_in = ((bbox[3] - bbox[1]) / height) * generator.SLIDE_HEIGHT_INCHES

            # Ensure minimum size
            w_in = max(w_in, 1.0)
            h_in = max(h_in, 0.4)

            elements.append(TextElement(
                x=x_in, y=y_in,
                width=w_in, height=h_in,
                text=text_content,
                font_size=self.settings.font.body_size,
                font_name=self.settings.font.default_font,
                font_color=(0, 0, 0)
            ))

        # 4. Add colored objects as shapes
        for obj in colored_objects:
            x, y, w, h = obj.bbox
            x_in = (x / width) * generator.SLIDE_WIDTH_INCHES
            y_in = (y / height) * generator.SLIDE_HEIGHT_INCHES
            w_in = (w / width) * generator.SLIDE_WIDTH_INCHES
            h_in = (h / height) * generator.SLIDE_HEIGHT_INCHES

            # Skip very small objects
            if w_in < 0.3 or h_in < 0.3:
                continue

            shape_type = self._detect_shape_type_from_contour(obj.contour)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=w_in,
                height=h_in,
                shape_type=shape_type,
                fill_color=obj.rgb_color,
                line_color=self._darken_color(obj.rgb_color),
                line_width=1.5
            ))

        # 5. Add detected shapes
        for shape in shapes:
            x, y, w, h = shape.bbox
            x_in = (x / width) * generator.SLIDE_WIDTH_INCHES
            y_in = (y / height) * generator.SLIDE_HEIGHT_INCHES
            w_in = (w / width) * generator.SLIDE_WIDTH_INCHES
            h_in = (h / height) * generator.SLIDE_HEIGHT_INCHES

            # Skip very small shapes
            if w_in < 0.3 or h_in < 0.3:
                continue

            shape_type = self._convert_detected_shape_type(shape.shape_type)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=w_in,
                height=h_in,
                shape_type=shape_type,
                fill_color=None,
                line_color=(0, 0, 0),
                line_width=1
            ))

        content = SlideContent(elements=elements)
        generator.add_slide_with_content(content)

    def _group_texts_into_paragraphs(
        self,
        texts: List[ExtractedText],
        page_height: float,
        line_spacing_threshold: float = 3.0,
        paragraph_gap_threshold: float = 15.0
    ) -> List[dict]:
        """
        Group text elements into paragraph blocks.

        Returns list of dicts with 'text', 'bbox', 'font_size'.
        """
        if not texts:
            return []

        # Sort by y position (top to bottom), then x position (left to right)
        sorted_texts = sorted(texts, key=lambda t: (round(t.bbox[1] / 5) * 5, t.bbox[0]))

        paragraphs = []
        current_paragraph_texts = []
        current_line_texts = []
        current_line_y = None

        for text in sorted_texts:
            text_y = text.bbox[1]

            if current_line_y is None:
                # First text
                current_line_y = text_y
                current_line_texts = [text]
            elif abs(text_y - current_line_y) <= line_spacing_threshold:
                # Same line
                current_line_texts.append(text)
            else:
                # New line - check if it's a new paragraph
                if current_line_texts:
                    # Add current line to paragraph
                    current_paragraph_texts.extend(current_line_texts)

                if text_y - current_line_y > paragraph_gap_threshold:
                    # New paragraph - save current and start new
                    if current_paragraph_texts:
                        paragraphs.append(self._create_paragraph_block(current_paragraph_texts))
                    current_paragraph_texts = []

                current_line_y = text_y
                current_line_texts = [text]

        # Don't forget the last line and paragraph
        if current_line_texts:
            current_paragraph_texts.extend(current_line_texts)
        if current_paragraph_texts:
            paragraphs.append(self._create_paragraph_block(current_paragraph_texts))

        return paragraphs

    def _create_paragraph_block(self, texts: List[ExtractedText]) -> dict:
        """Create a paragraph block from a list of texts."""
        # Sort texts left to right for proper reading order
        sorted_texts = sorted(texts, key=lambda t: (round(t.bbox[1] / 5) * 5, t.bbox[0]))

        # Build text content
        lines = []
        current_line = []
        current_y = None

        for text in sorted_texts:
            if current_y is None or abs(text.bbox[1] - current_y) <= 5:
                current_line.append(text.text)
                if current_y is None:
                    current_y = text.bbox[1]
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [text.text]
                current_y = text.bbox[1]

        if current_line:
            lines.append(" ".join(current_line))

        text_content = "\n".join(lines)

        # Calculate bounding box
        min_x = min(t.bbox[0] for t in texts)
        min_y = min(t.bbox[1] for t in texts)
        max_x = max(t.bbox[2] for t in texts)
        max_y = max(t.bbox[3] for t in texts)

        # Average font size
        avg_font_size = sum(t.font_size for t in texts) / len(texts)

        return {
            'text': text_content,
            'bbox': (min_x, min_y, max_x, max_y),
            'font_size': avg_font_size
        }

    def _group_ocr_into_paragraphs(
        self,
        ocr_results: List[OCRResult],
        image_height: int,
        line_threshold: int = 15,
        paragraph_gap: int = 30
    ) -> List[dict]:
        """Group OCR results into paragraph blocks."""
        if not ocr_results:
            return []

        # Filter low confidence results
        filtered = [r for r in ocr_results if r.confidence > 0.5 and r.text.strip()]
        if not filtered:
            return []

        # Sort by y then x
        sorted_results = sorted(filtered, key=lambda r: (r.bbox[1] // 10 * 10, r.bbox[0]))

        paragraphs = []
        current_paragraph = []
        current_line_y = None

        for result in sorted_results:
            y = result.bbox[1]

            if current_line_y is None:
                current_line_y = y
                current_paragraph = [result]
            elif abs(y - current_line_y) <= line_threshold:
                current_paragraph.append(result)
            elif y - current_line_y > paragraph_gap:
                # New paragraph
                if current_paragraph:
                    paragraphs.append(self._create_ocr_paragraph(current_paragraph))
                current_paragraph = [result]
                current_line_y = y
            else:
                # Same paragraph, new line
                current_paragraph.append(result)
                current_line_y = y

        if current_paragraph:
            paragraphs.append(self._create_ocr_paragraph(current_paragraph))

        return paragraphs

    def _create_ocr_paragraph(self, results: List[OCRResult]) -> dict:
        """Create paragraph from OCR results."""
        # Sort by y then x
        sorted_results = sorted(results, key=lambda r: (r.bbox[1] // 10 * 10, r.bbox[0]))

        # Build text
        lines = []
        current_line = []
        current_y = None

        for result in sorted_results:
            y = result.bbox[1]
            if current_y is None or abs(y - current_y) <= 15:
                current_line.append(result.text)
                if current_y is None:
                    current_y = y
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [result.text]
                current_y = y

        if current_line:
            lines.append(" ".join(current_line))

        text_content = "\n".join(lines)

        # Bounding box
        min_x = min(r.bbox[0] for r in results)
        min_y = min(r.bbox[1] for r in results)
        max_x = max(r.bbox[0] + r.bbox[2] for r in results)
        max_y = max(r.bbox[1] + r.bbox[3] for r in results)

        return {
            'text': text_content,
            'bbox': (min_x, min_y, max_x, max_y)
        }

    def _detect_shape_type_from_contour(self, contour: np.ndarray) -> ShapeType:
        """Detect shape type from contour."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        if vertices == 3:
            return ShapeType.TRIANGLE
        elif vertices == 4:
            return ShapeType.RECTANGLE
        elif vertices == 5:
            return ShapeType.PENTAGON
        elif vertices == 6:
            return ShapeType.HEXAGON
        else:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return ShapeType.OVAL
            return ShapeType.RECTANGLE

    def _convert_detected_shape_type(self, shape_type_str: str) -> ShapeType:
        """Convert detected shape type string to ShapeType enum."""
        mapping = {
            'rectangle': ShapeType.RECTANGLE,
            'square': ShapeType.RECTANGLE,
            'circle': ShapeType.OVAL,
            'triangle': ShapeType.TRIANGLE,
            'pentagon': ShapeType.PENTAGON,
            'hexagon': ShapeType.HEXAGON,
            'polygon': ShapeType.RECTANGLE,
        }
        return mapping.get(shape_type_str, ShapeType.RECTANGLE)

    def _darken_color(self, rgb: tuple, factor: float = 0.7) -> tuple:
        """Darken a color for border."""
        return (
            int(rgb[0] * factor),
            int(rgb[1] * factor),
            int(rgb[2] * factor)
        )


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
