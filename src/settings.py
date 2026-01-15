"""
Settings Module
Manages application settings including color ranges, fonts, and layout options.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .color_detector import ColorRange, ColorType, DEFAULT_COLOR_RANGES


@dataclass
class FontSettings:
    """Font configuration settings."""
    default_font: str = "Arial"
    title_font: str = "Arial"
    title_size: int = 28
    body_size: int = 12
    caption_size: int = 10


@dataclass
class LayoutSettings:
    """Layout configuration settings."""
    slide_width: float = 13.333  # inches
    slide_height: float = 7.5    # inches
    margin_left: float = 0.5
    margin_right: float = 0.5
    margin_top: float = 0.5
    margin_bottom: float = 0.5
    title_height: float = 1.0


@dataclass
class ProcessingSettings:
    """Processing configuration settings."""
    ocr_language: str = "jpn+eng"
    min_color_area: int = 100
    dpi: int = 150
    detect_shapes: bool = True
    detect_colors: bool = True


@dataclass
class ColorRangeConfig:
    """Serializable color range configuration."""
    name: str
    color_type: str
    lower_hsv: List[int]
    upper_hsv: List[int]
    rgb_display: List[int]

    def to_color_range(self) -> ColorRange:
        """Convert to ColorRange object."""
        return ColorRange(
            name=self.name,
            color_type=ColorType(self.color_type),
            lower_hsv=tuple(self.lower_hsv),
            upper_hsv=tuple(self.upper_hsv),
            rgb_display=tuple(self.rgb_display)
        )

    @classmethod
    def from_color_range(cls, cr: ColorRange) -> 'ColorRangeConfig':
        """Create from ColorRange object."""
        return cls(
            name=cr.name,
            color_type=cr.color_type.value,
            lower_hsv=list(cr.lower_hsv),
            upper_hsv=list(cr.upper_hsv),
            rgb_display=list(cr.rgb_display)
        )


@dataclass
class AppSettings:
    """Main application settings."""
    font: FontSettings = field(default_factory=FontSettings)
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    color_ranges: List[ColorRangeConfig] = field(default_factory=list)
    tesseract_path: str = ""
    last_input_dir: str = ""
    last_output_dir: str = ""

    def __post_init__(self):
        if not self.color_ranges:
            self.color_ranges = [
                ColorRangeConfig.from_color_range(cr)
                for cr in DEFAULT_COLOR_RANGES
            ]


class SettingsManager:
    """Manages application settings persistence."""

    DEFAULT_SETTINGS_FILE = "settings.json"

    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            settings_dir: Directory to store settings. Defaults to config folder.
        """
        if settings_dir:
            self.settings_dir = Path(settings_dir)
        else:
            # Use app directory config folder
            app_dir = Path(__file__).parent.parent
            self.settings_dir = app_dir / "config"

        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.settings_dir / self.DEFAULT_SETTINGS_FILE
        self.settings = AppSettings()

    def load(self) -> AppSettings:
        """
        Load settings from file.

        Returns:
            Loaded AppSettings object.
        """
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.settings = self._dict_to_settings(data)
            except Exception as e:
                print(f"Error loading settings: {e}")
                self.settings = AppSettings()
        else:
            self.settings = AppSettings()

        return self.settings

    def save(self):
        """Save current settings to file."""
        try:
            data = self._settings_to_dict(self.settings)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def reset_to_defaults(self):
        """Reset settings to defaults."""
        self.settings = AppSettings()
        self.save()

    def get_color_ranges(self) -> List[ColorRange]:
        """Get color ranges as ColorRange objects."""
        return [cr.to_color_range() for cr in self.settings.color_ranges]

    def set_color_ranges(self, color_ranges: List[ColorRange]):
        """Set color ranges from ColorRange objects."""
        self.settings.color_ranges = [
            ColorRangeConfig.from_color_range(cr) for cr in color_ranges
        ]

    def add_color_range(self, color_range: ColorRange):
        """Add a new color range."""
        self.settings.color_ranges.append(
            ColorRangeConfig.from_color_range(color_range)
        )

    def remove_color_range(self, name: str):
        """Remove a color range by name."""
        self.settings.color_ranges = [
            cr for cr in self.settings.color_ranges if cr.name != name
        ]

    def _settings_to_dict(self, settings: AppSettings) -> Dict[str, Any]:
        """Convert AppSettings to dictionary."""
        return {
            'font': asdict(settings.font),
            'layout': asdict(settings.layout),
            'processing': asdict(settings.processing),
            'color_ranges': [asdict(cr) for cr in settings.color_ranges],
            'tesseract_path': settings.tesseract_path,
            'last_input_dir': settings.last_input_dir,
            'last_output_dir': settings.last_output_dir
        }

    def _dict_to_settings(self, data: Dict[str, Any]) -> AppSettings:
        """Convert dictionary to AppSettings."""
        font = FontSettings(**data.get('font', {}))
        layout = LayoutSettings(**data.get('layout', {}))
        processing = ProcessingSettings(**data.get('processing', {}))

        color_ranges = []
        for cr_data in data.get('color_ranges', []):
            color_ranges.append(ColorRangeConfig(**cr_data))

        return AppSettings(
            font=font,
            layout=layout,
            processing=processing,
            color_ranges=color_ranges if color_ranges else None,
            tesseract_path=data.get('tesseract_path', ''),
            last_input_dir=data.get('last_input_dir', ''),
            last_output_dir=data.get('last_output_dir', '')
        )


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
        _settings_manager.load()
    return _settings_manager


def get_settings() -> AppSettings:
    """Get current application settings."""
    return get_settings_manager().settings


def save_settings():
    """Save current settings."""
    get_settings_manager().save()
