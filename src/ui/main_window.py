"""
Main Window UI Module
Tkinter-based desktop application interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
from typing import Optional, List
import os

from ..converter import Converter, ConversionProgress
from ..settings import get_settings_manager, save_settings


class MainWindow:
    """Main application window."""

    WINDOW_TITLE = "PDF/Image to PowerPoint Converter"
    WINDOW_SIZE = "700x500"

    SUPPORTED_FILES = [
        ("All Supported Files", "*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif;*.gif"),
        ("PDF Files", "*.pdf"),
        ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif;*.gif"),
        ("All Files", "*.*")
    ]

    def __init__(self):
        """Initialize main window."""
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.resizable(True, True)

        self.settings_manager = get_settings_manager()

        self.input_files: List[str] = []
        self.output_path: Optional[str] = None
        self.is_converting = False

        self._setup_ui()
        self._center_window()

    def _center_window(self):
        """Center window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _setup_ui(self):
        """Setup the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(
            main_frame,
            text="PDF/Image to PowerPoint Converter",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Input section
        self._setup_input_section(main_frame)

        # Output section
        self._setup_output_section(main_frame)

        # File list
        self._setup_file_list(main_frame)

        # Progress section
        self._setup_progress_section(main_frame)

        # Button section
        self._setup_button_section(main_frame)

        # Status bar
        self._setup_status_bar(main_frame)

    def _setup_input_section(self, parent):
        """Setup input file selection section."""
        input_frame = ttk.LabelFrame(parent, text="Input Files", padding="10")
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Selected:").grid(row=0, column=0, sticky="w")

        self.input_label = ttk.Label(input_frame, text="No files selected")
        self.input_label.grid(row=0, column=1, sticky="w", padx=(10, 10))

        select_btn = ttk.Button(
            input_frame,
            text="Select Files...",
            command=self._select_input_files
        )
        select_btn.grid(row=0, column=2)

    def _setup_output_section(self, parent):
        """Setup output file selection section."""
        output_frame = ttk.LabelFrame(parent, text="Output File", padding="10")
        output_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Save as:").grid(row=0, column=0, sticky="w")

        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=(10, 10))

        browse_btn = ttk.Button(
            output_frame,
            text="Browse...",
            command=self._select_output_file
        )
        browse_btn.grid(row=0, column=2)

    def _setup_file_list(self, parent):
        """Setup file list display."""
        list_frame = ttk.LabelFrame(parent, text="Files to Convert", padding="10")
        list_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            height=8
        )
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.file_listbox.yview)

        # List buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Remove Selected",
            command=self._remove_selected_file
        ).grid(row=0, column=0, padx=(0, 5))

        ttk.Button(
            btn_frame,
            text="Clear All",
            command=self._clear_all_files
        ).grid(row=0, column=1)

    def _setup_progress_section(self, parent):
        """Setup progress bar and status."""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, pady=(5, 0))

    def _setup_button_section(self, parent):
        """Setup main action buttons."""
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=5, column=0, pady=(0, 10))

        self.convert_btn = ttk.Button(
            btn_frame,
            text="Convert",
            command=self._start_conversion,
            width=15
        )
        self.convert_btn.grid(row=0, column=0, padx=5)

        self.settings_btn = ttk.Button(
            btn_frame,
            text="Settings",
            command=self._open_settings,
            width=15
        )
        self.settings_btn.grid(row=0, column=1, padx=5)

    def _setup_status_bar(self, parent):
        """Setup status bar at bottom."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=6, column=0, sticky="ew")

        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)

    def _select_input_files(self):
        """Open file dialog to select input files."""
        initial_dir = self.settings_manager.settings.last_input_dir or os.path.expanduser("~")

        files = filedialog.askopenfilenames(
            title="Select PDF or Image Files",
            initialdir=initial_dir,
            filetypes=self.SUPPORTED_FILES
        )

        if files:
            self.input_files = list(files)
            self._update_file_list()

            # Update last input directory
            if self.input_files:
                self.settings_manager.settings.last_input_dir = str(
                    Path(self.input_files[0]).parent
                )
                save_settings()

                # Auto-set output path if not set
                if not self.output_entry.get():
                    base_name = Path(self.input_files[0]).stem
                    output_dir = Path(self.input_files[0]).parent
                    self.output_entry.delete(0, tk.END)
                    self.output_entry.insert(0, str(output_dir / f"{base_name}_output.pptx"))

    def _select_output_file(self):
        """Open file dialog to select output file."""
        initial_dir = self.settings_manager.settings.last_output_dir or os.path.expanduser("~")

        file_path = filedialog.asksaveasfilename(
            title="Save PowerPoint As",
            initialdir=initial_dir,
            defaultextension=".pptx",
            filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")]
        )

        if file_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file_path)

            # Update last output directory
            self.settings_manager.settings.last_output_dir = str(Path(file_path).parent)
            save_settings()

    def _update_file_list(self):
        """Update the file listbox."""
        self.file_listbox.delete(0, tk.END)

        for file_path in self.input_files:
            self.file_listbox.insert(tk.END, Path(file_path).name)

        count = len(self.input_files)
        self.input_label.config(text=f"{count} file(s) selected")

    def _remove_selected_file(self):
        """Remove selected file from list."""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            self.input_files.pop(idx)
            self._update_file_list()

    def _clear_all_files(self):
        """Clear all files from list."""
        self.input_files = []
        self._update_file_list()

    def _start_conversion(self):
        """Start the conversion process."""
        if self.is_converting:
            return

        if not self.input_files:
            messagebox.showwarning("Warning", "Please select input files.")
            return

        output_path = self.output_entry.get().strip()
        if not output_path:
            messagebox.showwarning("Warning", "Please specify an output file.")
            return

        self.is_converting = True
        self.convert_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        # Run conversion in background thread
        thread = threading.Thread(
            target=self._run_conversion,
            args=(output_path,),
            daemon=True
        )
        thread.start()

    def _run_conversion(self, output_path: str):
        """Run conversion in background thread."""
        try:
            converter = Converter()
            total_files = len(self.input_files)

            for i, input_file in enumerate(self.input_files):
                file_name = Path(input_file).name
                self._update_status(f"Converting {file_name}...")

                def progress_callback(progress: ConversionProgress):
                    # Calculate overall progress
                    file_progress = progress.percentage / 100
                    overall = ((i + file_progress) / total_files) * 100
                    self.root.after(0, self._update_progress, overall, progress.step_name)

                # Generate output path for each file
                if total_files == 1:
                    file_output = output_path
                else:
                    base_name = Path(input_file).stem
                    output_dir = Path(output_path).parent
                    file_output = str(output_dir / f"{base_name}.pptx")

                success = converter.convert(input_file, file_output, progress_callback)

                if not success:
                    self.root.after(0, self._show_error, f"Failed to convert: {file_name}")
                    continue

            self.root.after(0, self._conversion_complete, output_path)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

        finally:
            self.is_converting = False
            self.root.after(0, lambda: self.convert_btn.config(state=tk.NORMAL))

    def _update_progress(self, percentage: float, step_name: str):
        """Update progress bar and label."""
        self.progress_var.set(percentage)
        self.progress_label.config(text=step_name)

    def _update_status(self, message: str):
        """Update status bar."""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def _conversion_complete(self, output_path: str):
        """Handle conversion completion."""
        self.progress_var.set(100)
        self.progress_label.config(text="Conversion complete!")
        self.status_label.config(text="Ready")

        messagebox.showinfo(
            "Success",
            f"Conversion complete!\nOutput saved to:\n{output_path}"
        )

    def _show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)
        self.status_label.config(text="Error occurred")
        self.progress_label.config(text="Error")

    def _open_settings(self):
        """Open settings dialog."""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.root, self.settings_manager)
        self.root.wait_window(dialog.dialog)

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Entry point for the application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
