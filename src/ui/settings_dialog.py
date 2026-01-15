"""
Settings Dialog UI Module
Dialog for configuring application settings.
"""

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
from typing import Optional, List, Tuple
import colorsys

from ..settings import SettingsManager, ColorRangeConfig, save_settings
from ..color_detector import ColorType


class SettingsDialog:
    """Settings dialog window."""

    def __init__(self, parent: tk.Tk, settings_manager: SettingsManager):
        """
        Initialize settings dialog.

        Args:
            parent: Parent window.
            settings_manager: Settings manager instance.
        """
        self.parent = parent
        self.settings_manager = settings_manager
        self.settings = settings_manager.settings

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._setup_ui()
        self._load_settings()
        self._center_dialog()

    def _center_dialog(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

    def _setup_ui(self):
        """Setup the dialog UI."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Font settings tab
        font_frame = ttk.Frame(notebook, padding="10")
        notebook.add(font_frame, text="Font")
        self._setup_font_tab(font_frame)

        # Processing settings tab
        processing_frame = ttk.Frame(notebook, padding="10")
        notebook.add(processing_frame, text="Processing")
        self._setup_processing_tab(processing_frame)

        # Color ranges tab
        color_frame = ttk.Frame(notebook, padding="10")
        notebook.add(color_frame, text="Color Ranges")
        self._setup_color_tab(color_frame)

        # Paths tab
        paths_frame = ttk.Frame(notebook, padding="10")
        notebook.add(paths_frame, text="Paths")
        self._setup_paths_tab(paths_frame)

        # Button frame
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(
            btn_frame,
            text="Reset to Defaults",
            command=self._reset_defaults
        ).pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.dialog.destroy
        ).pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(
            btn_frame,
            text="Save",
            command=self._save_settings
        ).pack(side=tk.RIGHT)

    def _setup_font_tab(self, parent):
        """Setup font settings tab."""
        # Default font
        ttk.Label(parent, text="Default Font:").grid(row=0, column=0, sticky="w", pady=5)
        self.default_font_entry = ttk.Entry(parent, width=30)
        self.default_font_entry.grid(row=0, column=1, pady=5, padx=(10, 0))

        # Title font
        ttk.Label(parent, text="Title Font:").grid(row=1, column=0, sticky="w", pady=5)
        self.title_font_entry = ttk.Entry(parent, width=30)
        self.title_font_entry.grid(row=1, column=1, pady=5, padx=(10, 0))

        # Title size
        ttk.Label(parent, text="Title Size:").grid(row=2, column=0, sticky="w", pady=5)
        self.title_size_var = tk.IntVar()
        self.title_size_spinbox = ttk.Spinbox(
            parent, from_=8, to=72, textvariable=self.title_size_var, width=10
        )
        self.title_size_spinbox.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))

        # Body size
        ttk.Label(parent, text="Body Size:").grid(row=3, column=0, sticky="w", pady=5)
        self.body_size_var = tk.IntVar()
        self.body_size_spinbox = ttk.Spinbox(
            parent, from_=8, to=72, textvariable=self.body_size_var, width=10
        )
        self.body_size_spinbox.grid(row=3, column=1, sticky="w", pady=5, padx=(10, 0))

        # Caption size
        ttk.Label(parent, text="Caption Size:").grid(row=4, column=0, sticky="w", pady=5)
        self.caption_size_var = tk.IntVar()
        self.caption_size_spinbox = ttk.Spinbox(
            parent, from_=6, to=36, textvariable=self.caption_size_var, width=10
        )
        self.caption_size_spinbox.grid(row=4, column=1, sticky="w", pady=5, padx=(10, 0))

    def _setup_processing_tab(self, parent):
        """Setup processing settings tab."""
        # OCR language
        ttk.Label(parent, text="OCR Language:").grid(row=0, column=0, sticky="w", pady=5)
        self.ocr_lang_entry = ttk.Entry(parent, width=30)
        self.ocr_lang_entry.grid(row=0, column=1, pady=5, padx=(10, 0))
        ttk.Label(
            parent,
            text="(e.g., 'jpn+eng' for Japanese and English)",
            font=('Helvetica', 8)
        ).grid(row=0, column=2, sticky="w", padx=(5, 0))

        # Min color area
        ttk.Label(parent, text="Min Color Area:").grid(row=1, column=0, sticky="w", pady=5)
        self.min_area_var = tk.IntVar()
        self.min_area_spinbox = ttk.Spinbox(
            parent, from_=10, to=10000, textvariable=self.min_area_var, width=10
        )
        self.min_area_spinbox.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))

        # DPI
        ttk.Label(parent, text="DPI:").grid(row=2, column=0, sticky="w", pady=5)
        self.dpi_var = tk.IntVar()
        self.dpi_spinbox = ttk.Spinbox(
            parent, from_=72, to=600, textvariable=self.dpi_var, width=10
        )
        self.dpi_spinbox.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))

        # Detect shapes checkbox
        self.detect_shapes_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Detect Shapes",
            variable=self.detect_shapes_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        # Detect colors checkbox
        self.detect_colors_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Detect Colors",
            variable=self.detect_colors_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

    def _setup_color_tab(self, parent):
        """Setup color ranges tab."""
        # Listbox for color ranges
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.color_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            height=10
        )
        self.color_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.color_listbox.yview)

        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Add",
            command=self._add_color_range
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            btn_frame,
            text="Edit",
            command=self._edit_color_range
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            btn_frame,
            text="Remove",
            command=self._remove_color_range
        ).pack(side=tk.LEFT)

    def _setup_paths_tab(self, parent):
        """Setup paths settings tab."""
        # Tesseract path
        ttk.Label(parent, text="Tesseract Path:").grid(row=0, column=0, sticky="w", pady=5)
        self.tesseract_entry = ttk.Entry(parent, width=40)
        self.tesseract_entry.grid(row=0, column=1, pady=5, padx=(10, 10))

        ttk.Button(
            parent,
            text="Browse...",
            command=self._browse_tesseract
        ).grid(row=0, column=2, pady=5)

        ttk.Label(
            parent,
            text="(Leave empty to use system default)",
            font=('Helvetica', 8)
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

    def _load_settings(self):
        """Load current settings into UI."""
        # Font settings
        self.default_font_entry.insert(0, self.settings.font.default_font)
        self.title_font_entry.insert(0, self.settings.font.title_font)
        self.title_size_var.set(self.settings.font.title_size)
        self.body_size_var.set(self.settings.font.body_size)
        self.caption_size_var.set(self.settings.font.caption_size)

        # Processing settings
        self.ocr_lang_entry.insert(0, self.settings.processing.ocr_language)
        self.min_area_var.set(self.settings.processing.min_color_area)
        self.dpi_var.set(self.settings.processing.dpi)
        self.detect_shapes_var.set(self.settings.processing.detect_shapes)
        self.detect_colors_var.set(self.settings.processing.detect_colors)

        # Paths
        self.tesseract_entry.insert(0, self.settings.tesseract_path)

        # Color ranges
        self._update_color_listbox()

    def _update_color_listbox(self):
        """Update color ranges listbox."""
        self.color_listbox.delete(0, tk.END)
        for cr in self.settings.color_ranges:
            self.color_listbox.insert(tk.END, f"{cr.name} ({cr.color_type})")

    def _browse_tesseract(self):
        """Browse for Tesseract executable."""
        file_path = filedialog.askopenfilename(
            title="Select Tesseract Executable",
            filetypes=[("Executable", "*.exe"), ("All Files", "*.*")]
        )
        if file_path:
            self.tesseract_entry.delete(0, tk.END)
            self.tesseract_entry.insert(0, file_path)

    def _add_color_range(self):
        """Add a new color range."""
        dialog = ColorRangeDialog(self.dialog)
        self.dialog.wait_window(dialog.dialog)

        if dialog.result:
            self.settings.color_ranges.append(dialog.result)
            self._update_color_listbox()

    def _edit_color_range(self):
        """Edit selected color range."""
        selection = self.color_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a color range to edit.")
            return

        idx = selection[0]
        color_range = self.settings.color_ranges[idx]

        dialog = ColorRangeDialog(self.dialog, color_range)
        self.dialog.wait_window(dialog.dialog)

        if dialog.result:
            self.settings.color_ranges[idx] = dialog.result
            self._update_color_listbox()

    def _remove_color_range(self):
        """Remove selected color range."""
        selection = self.color_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a color range to remove.")
            return

        idx = selection[0]
        if messagebox.askyesno("Confirm", "Remove selected color range?"):
            self.settings.color_ranges.pop(idx)
            self._update_color_listbox()

    def _save_settings(self):
        """Save settings and close dialog."""
        # Font settings
        self.settings.font.default_font = self.default_font_entry.get()
        self.settings.font.title_font = self.title_font_entry.get()
        self.settings.font.title_size = self.title_size_var.get()
        self.settings.font.body_size = self.body_size_var.get()
        self.settings.font.caption_size = self.caption_size_var.get()

        # Processing settings
        self.settings.processing.ocr_language = self.ocr_lang_entry.get()
        self.settings.processing.min_color_area = self.min_area_var.get()
        self.settings.processing.dpi = self.dpi_var.get()
        self.settings.processing.detect_shapes = self.detect_shapes_var.get()
        self.settings.processing.detect_colors = self.detect_colors_var.get()

        # Paths
        self.settings.tesseract_path = self.tesseract_entry.get()

        # Save to file
        save_settings()

        messagebox.showinfo("Success", "Settings saved successfully.")
        self.dialog.destroy()

    def _reset_defaults(self):
        """Reset to default settings."""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            self.settings_manager.reset_to_defaults()
            self.settings = self.settings_manager.settings

            # Clear and reload UI
            self.default_font_entry.delete(0, tk.END)
            self.title_font_entry.delete(0, tk.END)
            self.ocr_lang_entry.delete(0, tk.END)
            self.tesseract_entry.delete(0, tk.END)

            self._load_settings()


class ColorRangeDialog:
    """Dialog for adding/editing color ranges."""

    def __init__(
        self,
        parent: tk.Toplevel,
        color_range: Optional[ColorRangeConfig] = None
    ):
        """
        Initialize color range dialog.

        Args:
            parent: Parent window.
            color_range: Existing color range to edit, or None for new.
        """
        self.parent = parent
        self.existing = color_range
        self.result: Optional[ColorRangeConfig] = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Edit Color Range" if color_range else "Add Color Range")
        self.dialog.geometry("400x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._setup_ui()

        if color_range:
            self._load_color_range(color_range)

    def _setup_ui(self):
        """Setup dialog UI."""
        frame = ttk.Frame(self.dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Name
        ttk.Label(frame, text="Name:").grid(row=0, column=0, sticky="w", pady=5)
        self.name_entry = ttk.Entry(frame, width=30)
        self.name_entry.grid(row=0, column=1, columnspan=2, pady=5, padx=(10, 0))

        # Color type
        ttk.Label(frame, text="Color Type:").grid(row=1, column=0, sticky="w", pady=5)
        self.type_var = tk.StringVar()
        type_combo = ttk.Combobox(
            frame,
            textvariable=self.type_var,
            values=[ct.value for ct in ColorType],
            width=27
        )
        type_combo.grid(row=1, column=1, columnspan=2, pady=5, padx=(10, 0))

        # HSV ranges
        ttk.Label(frame, text="Hue Range (0-180):").grid(row=2, column=0, sticky="w", pady=5)
        self.h_low_var = tk.IntVar(value=0)
        self.h_high_var = tk.IntVar(value=180)
        ttk.Spinbox(frame, from_=0, to=180, textvariable=self.h_low_var, width=5).grid(
            row=2, column=1, sticky="w", pady=5, padx=(10, 5)
        )
        ttk.Label(frame, text="-").grid(row=2, column=1, pady=5)
        ttk.Spinbox(frame, from_=0, to=180, textvariable=self.h_high_var, width=5).grid(
            row=2, column=2, sticky="w", pady=5
        )

        ttk.Label(frame, text="Saturation Range (0-255):").grid(row=3, column=0, sticky="w", pady=5)
        self.s_low_var = tk.IntVar(value=100)
        self.s_high_var = tk.IntVar(value=255)
        ttk.Spinbox(frame, from_=0, to=255, textvariable=self.s_low_var, width=5).grid(
            row=3, column=1, sticky="w", pady=5, padx=(10, 5)
        )
        ttk.Label(frame, text="-").grid(row=3, column=1, pady=5)
        ttk.Spinbox(frame, from_=0, to=255, textvariable=self.s_high_var, width=5).grid(
            row=3, column=2, sticky="w", pady=5
        )

        ttk.Label(frame, text="Value Range (0-255):").grid(row=4, column=0, sticky="w", pady=5)
        self.v_low_var = tk.IntVar(value=100)
        self.v_high_var = tk.IntVar(value=255)
        ttk.Spinbox(frame, from_=0, to=255, textvariable=self.v_low_var, width=5).grid(
            row=4, column=1, sticky="w", pady=5, padx=(10, 5)
        )
        ttk.Label(frame, text="-").grid(row=4, column=1, pady=5)
        ttk.Spinbox(frame, from_=0, to=255, textvariable=self.v_high_var, width=5).grid(
            row=4, column=2, sticky="w", pady=5
        )

        # Display color
        ttk.Label(frame, text="Display Color:").grid(row=5, column=0, sticky="w", pady=5)
        self.color_preview = tk.Canvas(frame, width=50, height=25, bg="white")
        self.color_preview.grid(row=5, column=1, sticky="w", pady=5, padx=(10, 5))
        self.display_color = (255, 0, 0)

        ttk.Button(
            frame,
            text="Choose...",
            command=self._choose_color
        ).grid(row=5, column=2, pady=5)

        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.dialog.destroy
        ).pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(
            btn_frame,
            text="OK",
            command=self._save
        ).pack(side=tk.RIGHT)

    def _load_color_range(self, cr: ColorRangeConfig):
        """Load existing color range into UI."""
        self.name_entry.insert(0, cr.name)
        self.type_var.set(cr.color_type)

        self.h_low_var.set(cr.lower_hsv[0])
        self.h_high_var.set(cr.upper_hsv[0])
        self.s_low_var.set(cr.lower_hsv[1])
        self.s_high_var.set(cr.upper_hsv[1])
        self.v_low_var.set(cr.lower_hsv[2])
        self.v_high_var.set(cr.upper_hsv[2])

        self.display_color = tuple(cr.rgb_display)
        self._update_color_preview()

    def _choose_color(self):
        """Open color chooser dialog."""
        color = colorchooser.askcolor(
            initialcolor=self.display_color,
            title="Choose Display Color"
        )
        if color[0]:
            self.display_color = tuple(int(c) for c in color[0])
            self._update_color_preview()

    def _update_color_preview(self):
        """Update color preview canvas."""
        hex_color = f"#{self.display_color[0]:02x}{self.display_color[1]:02x}{self.display_color[2]:02x}"
        self.color_preview.config(bg=hex_color)

    def _save(self):
        """Save and close dialog."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name.")
            return

        color_type = self.type_var.get()
        if not color_type:
            messagebox.showwarning("Warning", "Please select a color type.")
            return

        self.result = ColorRangeConfig(
            name=name,
            color_type=color_type,
            lower_hsv=[self.h_low_var.get(), self.s_low_var.get(), self.v_low_var.get()],
            upper_hsv=[self.h_high_var.get(), self.s_high_var.get(), self.v_high_var.get()],
            rgb_display=list(self.display_color)
        )

        self.dialog.destroy()
