"""
Color Detection Module
Uses OpenCV to detect and identify colored objects in images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ColorType(Enum):
    """Predefined color types with semantic meanings."""
    RED = "red"           # Warning, alert
    BLUE = "blue"         # Information
    YELLOW = "yellow"     # Caution, attention
    GREEN = "green"       # Success, positive
    ORANGE = "orange"     # Warning (mild)
    PURPLE = "purple"     # Special, highlight
    PINK = "pink"         # Highlight
    CYAN = "cyan"         # Info (light)
    WHITE = "white"       # Background
    BLACK = "black"       # Text, border
    GRAY = "gray"         # Neutral


@dataclass
class ColorRange:
    """Defines HSV color range for detection."""
    name: str
    color_type: ColorType
    lower_hsv: Tuple[int, int, int]
    upper_hsv: Tuple[int, int, int]
    rgb_display: Tuple[int, int, int]  # For PowerPoint rendering


@dataclass
class ColoredObject:
    """Represents a detected colored object."""
    color_type: ColorType
    color_name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    rgb_color: Tuple[int, int, int]


# Default color ranges (HSV format)
DEFAULT_COLOR_RANGES = [
    ColorRange(
        name="Red",
        color_type=ColorType.RED,
        lower_hsv=(0, 100, 100),
        upper_hsv=(10, 255, 255),
        rgb_display=(255, 0, 0)
    ),
    ColorRange(
        name="Red (High)",
        color_type=ColorType.RED,
        lower_hsv=(160, 100, 100),
        upper_hsv=(180, 255, 255),
        rgb_display=(255, 0, 0)
    ),
    ColorRange(
        name="Blue",
        color_type=ColorType.BLUE,
        lower_hsv=(100, 100, 100),
        upper_hsv=(130, 255, 255),
        rgb_display=(0, 0, 255)
    ),
    ColorRange(
        name="Yellow",
        color_type=ColorType.YELLOW,
        lower_hsv=(20, 100, 100),
        upper_hsv=(35, 255, 255),
        rgb_display=(255, 255, 0)
    ),
    ColorRange(
        name="Green",
        color_type=ColorType.GREEN,
        lower_hsv=(35, 100, 100),
        upper_hsv=(85, 255, 255),
        rgb_display=(0, 255, 0)
    ),
    ColorRange(
        name="Orange",
        color_type=ColorType.ORANGE,
        lower_hsv=(10, 100, 100),
        upper_hsv=(20, 255, 255),
        rgb_display=(255, 165, 0)
    ),
    ColorRange(
        name="Purple",
        color_type=ColorType.PURPLE,
        lower_hsv=(130, 100, 100),
        upper_hsv=(160, 255, 255),
        rgb_display=(128, 0, 128)
    ),
    ColorRange(
        name="Pink",
        color_type=ColorType.PINK,
        lower_hsv=(145, 50, 150),
        upper_hsv=(175, 255, 255),
        rgb_display=(255, 192, 203)
    ),
    ColorRange(
        name="Cyan",
        color_type=ColorType.CYAN,
        lower_hsv=(85, 100, 100),
        upper_hsv=(100, 255, 255),
        rgb_display=(0, 255, 255)
    ),
]


class ColorDetector:
    """Detects colored objects in images using HSV color space."""

    def __init__(self, color_ranges: Optional[List[ColorRange]] = None):
        """
        Initialize color detector.

        Args:
            color_ranges: Custom color ranges to use. If None, uses defaults.
        """
        self.color_ranges = color_ranges if color_ranges else DEFAULT_COLOR_RANGES

    def add_color_range(self, color_range: ColorRange):
        """Add a new color range for detection."""
        self.color_ranges.append(color_range)

    def remove_color_range(self, name: str):
        """Remove a color range by name."""
        self.color_ranges = [cr for cr in self.color_ranges if cr.name != name]

    def set_color_ranges(self, color_ranges: List[ColorRange]):
        """Set the color ranges to use."""
        self.color_ranges = color_ranges

    def detect_colors(
        self,
        image: np.ndarray,
        min_area: int = 100,
        color_types: Optional[List[ColorType]] = None
    ) -> List[ColoredObject]:
        """
        Detect colored objects in an image.

        Args:
            image: OpenCV image (BGR format).
            min_area: Minimum contour area to consider.
            color_types: Optional list of color types to detect. If None, detects all.

        Returns:
            List of ColoredObject.
        """
        objects = []

        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Filter color ranges if specified
        ranges_to_use = self.color_ranges
        if color_types:
            ranges_to_use = [cr for cr in self.color_ranges if cr.color_type in color_types]

        for color_range in ranges_to_use:
            detected = self._detect_single_color(
                image, hsv_image, color_range, min_area
            )
            objects.extend(detected)

        return objects

    def _detect_single_color(
        self,
        image: np.ndarray,
        hsv_image: np.ndarray,
        color_range: ColorRange,
        min_area: int
    ) -> List[ColoredObject]:
        """
        Detect objects of a single color.

        Args:
            image: Original BGR image.
            hsv_image: HSV converted image.
            color_range: Color range to detect.
            min_area: Minimum area threshold.

        Returns:
            List of detected objects.
        """
        objects = []

        # Create mask for the color range
        lower = np.array(color_range.lower_hsv)
        upper = np.array(color_range.upper_hsv)
        mask = cv2.inRange(hsv_image, lower, upper)

        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Get average color of the region
            mask_region = mask[y:y+h, x:x+w]
            if np.any(mask_region):
                region = image[y:y+h, x:x+w]
                avg_color = cv2.mean(region, mask=mask_region)[:3]
                rgb_color = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            else:
                rgb_color = color_range.rgb_display

            objects.append(ColoredObject(
                color_type=color_range.color_type,
                color_name=color_range.name,
                bbox=(x, y, w, h),
                contour=contour,
                center=(cx, cy),
                area=area,
                rgb_color=rgb_color
            ))

        return objects

    def create_color_mask(
        self,
        image: np.ndarray,
        color_types: Optional[List[ColorType]] = None
    ) -> np.ndarray:
        """
        Create a binary mask for specified colors.

        Args:
            image: OpenCV image (BGR format).
            color_types: Color types to include in mask.

        Returns:
            Binary mask image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        ranges_to_use = self.color_ranges
        if color_types:
            ranges_to_use = [cr for cr in self.color_ranges if cr.color_type in color_types]

        for color_range in ranges_to_use:
            lower = np.array(color_range.lower_hsv)
            upper = np.array(color_range.upper_hsv)
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def visualize_detection(
        self,
        image: np.ndarray,
        objects: List[ColoredObject],
        draw_bbox: bool = True,
        draw_contour: bool = True,
        draw_label: bool = True
    ) -> np.ndarray:
        """
        Visualize detected colored objects on the image.

        Args:
            image: Original image.
            objects: List of detected objects.
            draw_bbox: Whether to draw bounding boxes.
            draw_contour: Whether to draw contours.
            draw_label: Whether to draw labels.

        Returns:
            Image with visualizations.
        """
        result = image.copy()

        for obj in objects:
            color = (obj.rgb_color[2], obj.rgb_color[1], obj.rgb_color[0])  # RGB to BGR

            if draw_contour:
                cv2.drawContours(result, [obj.contour], -1, color, 2)

            if draw_bbox:
                x, y, w, h = obj.bbox
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            if draw_label:
                x, y, w, h = obj.bbox
                label = f"{obj.color_name}"
                cv2.putText(
                    result, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return result


def detect_colored_objects(
    image_path: str,
    color_types: Optional[List[ColorType]] = None,
    min_area: int = 100
) -> List[ColoredObject]:
    """
    Convenience function to detect colored objects in an image file.

    Args:
        image_path: Path to image file.
        color_types: Optional list of color types to detect.
        min_area: Minimum area threshold.

    Returns:
        List of ColoredObject.
    """
    image = cv2.imread(image_path)
    if image is None:
        return []

    detector = ColorDetector()
    return detector.detect_colors(image, min_area, color_types)


def create_custom_color_range(
    name: str,
    color_type: ColorType,
    h_range: Tuple[int, int],
    s_range: Tuple[int, int] = (100, 255),
    v_range: Tuple[int, int] = (100, 255),
    rgb_display: Optional[Tuple[int, int, int]] = None
) -> ColorRange:
    """
    Create a custom color range.

    Args:
        name: Name of the color.
        color_type: Type of color.
        h_range: Hue range (0-180).
        s_range: Saturation range (0-255).
        v_range: Value range (0-255).
        rgb_display: RGB color for display.

    Returns:
        ColorRange object.
    """
    if rgb_display is None:
        # Generate approximate RGB from HSV center
        h_center = (h_range[0] + h_range[1]) // 2
        hsv = np.uint8([[[h_center, 200, 200]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        rgb_display = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    return ColorRange(
        name=name,
        color_type=color_type,
        lower_hsv=(h_range[0], s_range[0], v_range[0]),
        upper_hsv=(h_range[1], s_range[1], v_range[1]),
        rgb_display=rgb_display
    )
