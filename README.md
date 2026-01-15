# InsiteSlideFromPDF

PDF/Image to PowerPoint Converter - A Windows desktop application that converts PDF documents and images into PowerPoint presentations with intelligent color object detection.

## Features

- **PDF Processing**: Extract text and images from PDF files using PyMuPDF
- **Image OCR**: Extract text from images using Tesseract OCR with Japanese and English support
- **Color Detection**: Automatically detect colored objects (red, blue, yellow, green, etc.) using OpenCV
- **Shape Recognition**: Identify geometric shapes within images
- **PowerPoint Generation**: Create professional PPTX files using python-pptx
- **User-Friendly Interface**: Simple tkinter GUI with progress tracking
- **Customizable Settings**: Configure color ranges, fonts, and processing options

## Requirements

- Python 3.8+
- Tesseract OCR (for image text extraction)
- Windows, macOS, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/InsiteSlideFromPDF.git
cd InsiteSlideFromPDF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Windows**: Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-jpn`

## Usage

### Running the Application

```bash
python main.py
```

### Using the GUI

1. **Select Files**: Click "Select Files..." to choose PDF or image files
2. **Set Output**: Specify the output PowerPoint file path
3. **Convert**: Click "Convert" to start the conversion process
4. **Configure**: Use "Settings" to customize color detection and font options

### Programmatic Usage

```python
from src import Converter, convert_file

# Simple conversion
convert_file("input.pdf", "output.pptx")

# With progress callback
def on_progress(progress):
    print(f"{progress.percentage:.1f}% - {progress.step_name}")

converter = Converter()
converter.convert("input.pdf", "output.pptx", progress_callback=on_progress)
```

## Configuration

Settings are stored in `config/settings.json` and can be modified through the GUI or directly:

### Color Detection

Configure HSV color ranges for detection:
- **Red**: Warning/Alert objects
- **Blue**: Information objects
- **Yellow**: Caution objects
- **Green**: Success/Positive objects

### Processing Options

- **OCR Language**: Set language codes (e.g., "jpn+eng")
- **Min Color Area**: Minimum pixel area for color detection
- **DPI**: Resolution for PDF rendering

## Project Structure

```
InsiteSlideFromPDF/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
│   └── settings.json
└── src/
    ├── __init__.py
    ├── pdf_processor.py    # PDF text/image extraction
    ├── image_processor.py  # OCR and shape detection
    ├── color_detector.py   # Color range detection
    ├── pptx_generator.py   # PowerPoint generation
    ├── converter.py        # Main conversion logic
    ├── settings.py         # Settings management
    └── ui/
        ├── __init__.py
        ├── main_window.py     # Main application window
        └── settings_dialog.py # Settings configuration dialog
```

## Dependencies

- **PyMuPDF**: PDF parsing and rendering
- **pytesseract**: OCR text extraction
- **opencv-python**: Image processing and color detection
- **python-pptx**: PowerPoint file generation
- **Pillow**: Image handling
- **tkinter**: GUI framework (included with Python)

## License

MIT License
