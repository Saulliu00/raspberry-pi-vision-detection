# Raspberry Pi 5 Vision Detection System

A modular, comprehensive computer vision system designed specifically for Raspberry Pi 5 with camera module. This system provides multiple object detection methods including YOLO, Haar cascades, and contour-based detection.

## Features

- **Multiple Detection Methods**: YOLO, Haar cascades, and contour-based detection
- **Modular Architecture**: Separate modules for easy testing and maintenance
- **Raspberry Pi Optimized**: Uses `rpicam-still` for optimal camera performance
- **Configurable**: Extensive configuration options through INI files
- **Real-time Processing**: Support for continuous detection loops
- **Performance Monitoring**: Built-in timing and performance analysis
- **Automatic Logging**: Comprehensive logging and result storage
- **Easy Testing**: Each module can be tested independently

## System Requirements

- Raspberry Pi 5 (recommended) or Raspberry Pi 4
- Raspberry Pi Camera Module (v2 or v3)
- Python 3.8+
- At least 4GB RAM (8GB recommended for YOLO)
- 16GB+ SD card

## Quick Installation

1. Clone or download the project files
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the automated setup:
   ```bash
   ./setup.sh
   ```
4. Follow the prompts to complete installation

## Manual Installation

If you prefer manual installation:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv libcamera-apps

# Create virtual environment
python3 -m venv vision_env
source vision_env/bin/activate

# Install Python packages
pip install -r requirements.txt

# Make scripts executable
chmod +x *.py
```

## Project Structure

```
vision-detection-system/
├── camera_module.py      # Camera operations and image capture
├── object_detection.py   # Object detection algorithms
├── vision_utils.py       # Utility functions and helpers
├── main.py              # Main system integration
├── requirements.txt     # Python dependencies
├── setup.sh            # Automated setup script
├── maintenance.sh      # System maintenance script
├── vision_config.ini   # Configuration file (auto-generated)
├── output/             # Detection results and images
└── README.md           # This file
```

## Module Overview

### 1. Camera Module (`camera_module.py`)

Handles all camera operations:
- Image capture using `rpicam-still` (primary method)
- OpenCV camera support (fallback method)
- Image preprocessing and enhancement
- Image saving and management

**Test independently:**
```bash
python3 camera_module.py
```

### 2. Object Detection (`object_detection.py`)

Provides multiple detection methods:
- **YOLO**: Advanced neural network detection
- **Haar Cascades**: Fast face/feature detection
- **Contour Detection**: Color-based object detection

**Test independently:**
```bash
python3 object_detection.py
```

### 3. Vision Utils (`vision_utils.py`)

Utility functions and classes:
- Configuration management
- File operations and logging
- Image processing utilities
- Performance monitoring
- Color presets for detection

**Test independently:**
```bash
python3 vision_utils.py
```

### 4. Main System (`main.py`)

Integrates all modules with:
- Command-line interface
- Single and continuous detection modes
- Real-time processing
- Result visualization and storage

## Usage Examples

### Basic Usage

```bash
# Activate virtual environment
source vision_env/bin/activate

# Single detection
python3 main.py --mode single

# Continuous detection every 2 seconds
python3 main.py --mode continuous --interval 2
```

### Advanced Usage

```bash
# Detect red objects using contour detection
python3 main.py --mode single --method contour --color red

# Use YOLO for advanced object detection
python3 main.py --mode single --method yolo

# Face detection with Haar cascades
python3 main.py --mode single --method haar

# Continuous detection with custom config
python3 main.py --mode continuous --interval 1 --config my_config.ini --verbose
```

### Testing Individual Functions

Each module can be tested independently to verify functionality:

```bash
# Test camera capture and preprocessing
python3 camera_module.py

# Test object detection algorithms
python3 object_detection.py

# Test utility functions
python3 vision_utils.py
```

## Configuration

The system uses an INI configuration file (`vision_config.ini`) that's automatically created on first run. Key sections:

### Camera Settings
```ini
[CAMERA]
width = 640
height = 480
timeout = 3000
method = rpicam
```

### Detection Settings
```ini
[DETECTION]
method = contour
confidence_threshold = 0.5
target_color = red
min_area = 500
```

### Output Settings
```ini
[OUTPUT]
save_images = true
output_directory = ./output
draw_detections = true
```

## Detection Methods

### 1. Contour Detection (Default)
- **Best for**: Colored objects, simple shapes
- **Advantages**: Fast, lightweight, no model files needed
- **Use case**: Detecting balls, markers, specific colored objects

### 2. YOLO Detection
- **Best for**: General object recognition
- **Advantages**: Recognizes many object classes, high accuracy
- **Use case**: Detecting people, vehicles, animals, household items

### 3. Haar Cascade Detection
- **Best for**: Face detection, specific features
- **Advantages**: Fast, reliable for trained features
- **Use case**: Face detection, eye tracking, smile detection

## Performance Optimization

### For Raspberry Pi 5:
- Use lower resolution for faster processing: `640x480` or `320x240`
- Increase GPU memory: Add `gpu_mem=128` to `/boot/config.txt`
- Use contour detection for real-time applications
- Enable camera interface in `raspi-config`

### For Better Detection:
- Ensure good lighting conditions
- Use stable camera mounting
- Adjust color thresholds for your environment
- Use appropriate detection method for your use case

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
rpicam-still --list-cameras

# Test camera capture
rpicam-still --timeout 2000 --output test.jpg
```

### Common Problems

1. **Camera not detected**:
   - Check camera cable connection
   - Enable camera in `sudo raspi-config`
   - Restart the Pi

2. **Permission errors**:
   - Make scripts executable: `chmod +x *.py`
   - Check file ownership

3. **Poor detection performance**:
   - Improve lighting conditions
   - Adjust color thresholds in config
   - Use appropriate detection method

4. **Memory issues with YOLO**:
   - Increase swap space
   - Use smaller image resolution
   - Consider using contour detection instead

### Log Analysis
```bash
# View recent logs if using systemd service
journalctl -u vision-detection -f

# Check output directory for saved results
ls -la output/
```

## System Maintenance

Use the included maintenance script:

```bash
./maintenance.sh
```

This will:
- Check system status and resources
- Analyze recent logs
- Clean up old output files
- Check for updates

## API Integration

The system can be easily integrated into larger projects:

```python
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import ConfigManager

# Initialize components
camera = RaspberryPiCamera(640, 480)
detector = ObjectDetector()
config = ConfigManager()

# Capture and detect
image = camera.capture_image_rpicam()
detections = detector.detect_objects_contour(image)

# Process results
for detection in detections:
    print(f"Found {detection['class']} at {detection['bbox']}")
```

## Service Installation

For automatic startup, the setup script can install a systemd service:

```bash
# Enable and start service
sudo systemctl enable vision-detection
sudo systemctl start vision-detection

# Check status
sudo systemctl status vision-detection

# View logs
journalctl -u vision-detection -f
```

## Contributing

This modular design makes it easy to extend:

1. **Add new detection methods**: Extend the `ObjectDetector` class
2. **Add new camera sources**: Extend the `RaspberryPiCamera` class
3. **Add new utilities**: Add functions to `vision_utils.py`
4. **Modify main logic**: Update `main.py` for new features

## Hardware Recommendations

- **Raspberry Pi 5 8GB**: Best performance for all detection methods
- **Raspberry Pi Camera Module v3**: Latest features and quality
- **High-speed SD card**: Class 10 or better for good I/O performance
- **Active cooling**: Helps maintain performance during continuous operation

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Test individual modules to isolate problems
3. Review the configuration settings
4. Check system resources and camera connection

---

**Happy detecting! 🎯📷**