# 🎵 Audio Watermarking System

A robust audio watermarking application using DWT-DCT approach with image scrambling techniques. Features a modern Flask web interface for easy watermark embedding and extraction.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Features

- **Modern Web Interface** - User-friendly Flask-based web application
- **Multiple Embedding Modes**:
  - Brute Binary
  - Brute Grayscale
  - Delta DCT
- **Image Scrambling Techniques**:
  - Arnold Transform
  - Lower Triangular Mapping
  - Upper Triangular Mapping
- **Quality Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SNR (Signal-to-Noise Ratio)
  - Correlation Analysis
- **Real-time Processing** - Instant watermark embedding and extraction
- **Download Results** - Easy download of watermarked audio and extracted images

## 🔬 Technical Approach

This project implements audio watermarking using:
- **DWT (Discrete Wavelet Transform)** - For audio decomposition
- **DCT (Discrete Cosine Transform)** - For coefficient transformation
- **Image Scrambling** - Arnold or Triangular mapping for enhanced security
- **Blind Extraction** - Can extract watermarks without original audio (depending on mode)

### Scientific References
- [Blind Audio Watermarking Based On Discrete Wavelet and Cosine Transform](https://ieeexplore.ieee.org/abstract/document/7150750)
- [Novel secured scheme for blind audio/speech norm-space watermarking by Arnold algorithm](https://www.sciencedirect.com/science/article/pii/S016516841830272X)
- [2D Triangular Mappings and Their Applications in Scrambling Rectangle Image](https://www.researchgate.net/publication/26557013_2D_Triangular_Mappings_and_Their_Applications_in_Scrambling_Rectangle_Image)

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Audio-Watermark-DWT-DCT.git
cd Audio-Watermark-DWT-DCT
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
```

3. **Activate the virtual environment**

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Web Interface (Recommended)

1. **Start the Flask application**
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Embed a Watermark**:
   - Upload an audio file (WAV format)
   - Upload a watermark image (PNG, JPG, JPEG)
   - Select embedding and scrambling modes
   - Click "Embed Watermark"
   - View quality metrics and download results

4. **Extract a Watermark**:
   - Upload the watermarked audio file
   - Upload the original audio file
   - Select the same modes used for embedding
   - Click "Extract Watermark"
   - Download the extracted watermark image

### Command Line Interface

```bash
python main.py --source audio.wav \
               --watermark image.png \
               --type-watermark BINARY \
               --embedding-mode bruteBinary \
               --scrambling-mode lower \
               --output results/
```

**Parameters:**
- `--source`: Input audio file (WAV format)
- `--watermark`: Watermark image file
- `--type-watermark`: `BINARY` or `GRAYSCALE`
- `--embedding-mode`: `bruteBinary`, `bruteGray`, or `delta`
- `--scrambling-mode`: `arnold`, `lower`, or `upper`
- `--output`: Output directory for results

## 📁 Project Structure

```
Audio-Watermark-DWT-DCT/
├── app.py                              # Flask web application
├── main.py                             # Core watermarking pipeline
├── audio_managing.py                   # Audio processing functions
├── image_managing.py                   # Image scrambling functions
├── watermark_embedding_extraction.py   # Embedding/extraction algorithms
├── metrics.py                          # Quality metrics calculation
├── attacks.py                          # Robustness testing
├── utils.py                            # Utility functions
├── requirements.txt                    # Python dependencies
├── templates/                          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── extract_result.html
├── uploads/                            # Uploaded files (gitignored)
└── outputs/                            # Generated outputs (gitignored)
```

## 📊 Quality Metrics

The system provides comprehensive quality metrics:

- **Correlation**: Measures similarity between original and extracted watermark (threshold: 0.7)
- **PSNR**: Peak Signal-to-Noise Ratio for watermark quality (higher is better)
- **SNR**: Signal-to-Noise Ratio for audio quality comparison (higher is better)

## 🔧 Supported Formats

- **Audio**: WAV (mono or stereo - automatically converted)
- **Images**: PNG, JPG, JPEG
- **Watermark Types**: Binary (black & white) or Grayscale

## 🎯 Use Cases

- **Copyright Protection** - Embed ownership information in audio files
- **Audio Authentication** - Verify audio file authenticity
- **Content Tracking** - Track audio content distribution
- **Research** - Academic research in digital watermarking

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**Port 5000 already in use:**
```bash
# Change port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Audio file not supported:**
- Ensure your audio is in WAV format
- Convert using online tools or: `ffmpeg -i input.mp3 output.wav`

## 👥 Authors

- [Maria Ausilia Napoli Spatafora](https://github.com/ausilianapoli)
- [Mattia Litrico](https://github.com/mattia1997)

**Enhanced Web Interface & Bug Fixes** by [Your Name]

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original project for the academic course of Multimedia at University of Catania
- Master Degree in Computer Science
- Flask framework for web interface
- PyWavelets for wavelet transforms
- SciPy for signal processing

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## ⭐ Star this repository if you find it helpful!