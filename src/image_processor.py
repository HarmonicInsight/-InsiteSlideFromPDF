"""
Image Processing Module
Uses Tesseract OCR for text extraction and OpenCV for image analysis.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Represents OCR result for a text block."""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float


@dataclass
class DetectedShape:
    """Represents a detected shape in an image."""
    shape_type: str  # 'rectangle', 'circle', 'line', 'arrow', 'polygon'
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    contour: np.ndarray
    center: Tuple[int, int]
    area: float


class ImageProcessor:
    """Processes images for OCR and shape detection."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize image processor.

        Args:
            tesseract_cmd: Optional path to Tesseract executable.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.

        Args:
            image_path: Path to image file.

        Returns:
            OpenCV image (BGR format).
        """
        try:
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format.

        Args:
            pil_image: PIL Image object.

        Returns:
            OpenCV image (BGR format).
        """
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        numpy_image = np.array(pil_image)
        # Convert RGB to BGR
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL format.

        Args:
            cv2_image: OpenCV image (BGR format).

        Returns:
            PIL Image object.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def extract_text_ocr(self, image: np.ndarray, lang: str = 'jpn+eng') -> List[OCRResult]:
        """
        Extract text from image using OCR.

        Args:
            image: OpenCV image.
            lang: Language code for Tesseract.

        Returns:
            List of OCRResult objects.
        """
        results = []

        # Convert to RGB for pytesseract
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Get detailed OCR data
        try:
            data = pytesseract.image_to_data(
                pil_image,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if text and conf > 0:  # Filter empty and low confidence
                    results.append(OCRResult(
                        text=text,
                        bbox=(
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ),
                        confidence=conf / 100.0
                    ))
        except Exception as e:
            print(f"OCR error: {e}")

        return results

    def extract_full_text(self, image: np.ndarray, lang: str = 'jpn+eng') -> str:
        """
        Extract full text from image.

        Args:
            image: OpenCV image.
            lang: Language code for Tesseract.

        Returns:
            Extracted text string.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        try:
            text = pytesseract.image_to_string(pil_image, lang=lang)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def detect_shapes(
        self,
        image: np.ndarray,
        min_area: int = 100,
        approx_epsilon: float = 0.02
    ) -> List[DetectedShape]:
        """
        Detect shapes in an image.

        Args:
            image: OpenCV image.
            min_area: Minimum contour area to consider.
            approx_epsilon: Approximation accuracy factor.

        Returns:
            List of DetectedShape objects.
        """
        shapes = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Approximate the contour
            epsilon = approx_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Determine shape type
            shape_type = self._classify_shape(approx, contour)

            shapes.append(DetectedShape(
                shape_type=shape_type,
                bbox=(x, y, w, h),
                contour=contour,
                center=(cx, cy),
                area=area
            ))

        return shapes

    def _classify_shape(self, approx: np.ndarray, contour: np.ndarray) -> str:
        """
        Classify a shape based on its approximated contour.

        Args:
            approx: Approximated contour.
            contour: Original contour.

        Returns:
            Shape type string.
        """
        vertices = len(approx)

        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            # Check if rectangle or square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return 'square'
            else:
                return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        else:
            # Check if circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:
                    return 'circle'
            return 'polygon'

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: OpenCV image.

        Returns:
            Preprocessed image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        return denoised


def process_image_file(
    image_path: str,
    tesseract_cmd: Optional[str] = None,
    lang: str = 'jpn+eng'
) -> Tuple[List[OCRResult], List[DetectedShape]]:
    """
    Convenience function to process an image file.

    Args:
        image_path: Path to image file.
        tesseract_cmd: Optional path to Tesseract executable.
        lang: Language code for OCR.

    Returns:
        Tuple of (OCR results, detected shapes).
    """
    processor = ImageProcessor(tesseract_cmd)
    image = processor.load_image(image_path)

    if image is None:
        return [], []

    ocr_results = processor.extract_text_ocr(image, lang)
    shapes = processor.detect_shapes(image)

    return ocr_results, shapes
