# Raspberry Pi 5 Vision Detection System

A modular, comprehensive computer vision system designed specifically for Raspberry Pi 5 with camera module. This system provides multiple object detection methods including YOLO, Haar cascades, and contour-based detection, with **# Raspberry Pi 5 Vision Detection System

A modular, comprehensive computer vision system designed specifically for Raspberry Pi 5 with camera module. This system provides multiple object detection methods including YOLO, Haar cascades, and contour-based detection, with **automatic system detection** and **full resolution capture** for maximum image quality.

## 🆕 New Features (v2.0)

- **🔍 Auto-Detection**: Automatically detects system timezone and camera's native resolution
- **📸 Full Resolution**: Uses camera's maximum resolution by default (up to 12MP on Camera Module 3)
- **🔄 Image Rotation**: Built-in 180° rotation (configurable for different mounting orientations)
- **⚡ Smart Performance**: Optional detection resizing for speed while maintaining image quality
- **🌍 Timezone Aware**: Auto-detects system timezone for accurate folder naming
- **📁 Structured Storage**: Organized photo storage with date-based folders
- **🎛️ Flexible Configuration**: Easy switching between quality and performance modes

## Features

- **Multiple Detection Methods**: YOLO, Haar cascades, and contour-based detection
- **Modular Architecture**: Separate modules for easy testing and maintenance
- **Auto-Detection**: Automatic timezone and camera resolution detection
- **Full Resolution Support**: Maximum image quality with performance optimization options
- **Raspberry Pi Optimized**: Uses `rpicam-still` for optimal camera performance
- **Smart Resizing**: Optional detection resizing while preserving original quality
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
- 32GB+ SD card (for full resolution images)

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

## Project Structure

```
vision-detection-system/
├── camera_module.py      # Camera operations with auto-detection
├── object_detection.py   # Object detection algorithms
├── vision_utils.py       # Utility functions and helpers
├── main.py              # Main system integration
├── example_usage.py     # Comprehensive usage examples
├── requirements.txt     # Python dependencies
├── setup.sh            # Automated setup script
├── maintenance.sh      # System maintenance script
├── vision_config.ini   # Configuration file (auto-generated)
├── pictures/           # Full resolution photos (organized by date)
├── output/             # Detection results and metadata
└── README.md           # This file
```

## 🚀 Quick Start Examples

### **Basic Auto-Detection Mode**
```bash
# Uses auto-detected timezone and full camera resolution
python3 main.py --mode single
```

### **High Performance Mode** 
```bash
# Full resolution capture, resized detection for speed
python3 main.py --mode continuous --interval 1 --resize-detection
```

### **Test All Features**
```bash
# Run comprehensive examples including rotation tests
python3 example_usage.py
```

### **Custom Resolution**
```bash
# Override auto-detection for specific resolution
python3 main.py --mode single --resolution 1080p
```

## Resolution Modes

### **Auto Mode (Default - Recommended)**
- **Full Resolution**: Uses camera's native resolution (up to 4608x2592 on Camera Module 3)
- **Best Quality**: Maximum detail for analysis and storage
- **Auto-Detection**: Automatically detects optimal settings

### **Performance Mode**  
- **Smart Resizing**: Captures at full resolution, detects on smaller image
- **Best of Both**: Quality preservation + detection speed
- **Configurable**: Customize detection resolution (640x480, 1280x720, etc.)

### **Compatibility Mode**
- **Fixed Resolution**: Traditional fixed resolution (1080p, 720p, 480p)
- **Predictable**: Consistent performance across different cameras
- **Legacy Support**: Compatible with older systems

## Configuration Examples

### **Maximum Quality with Rotation (Default)**
```ini
[CAMERA]
width = auto                    # Auto-detect native resolution
height = auto                   # Auto-detect native resolution  
timezone = auto                 # Auto-detect system timezone
use_full_resolution = true      # Use camera's maximum resolution
rotation_degrees = 180          # Rotate for upside-down mounting

[DETECTION]
resize_for_detection = false    # Detect on full resolution
```

### **Balanced Performance**
```ini
[CAMERA] 
width = auto                    # Auto-detect for capture
height = auto
use_full_resolution = true      # Capture at full resolution

[DETECTION]
resize_for_detection = true     # Resize for detection speed
detection_width = 640           # Detection resolution
detection_height = 480
```

### **High Speed Mode**
```ini
[CAMERA]
width = 1280                    # Fixed resolution
height = 720
use_full_resolution = false

[DETECTION] 
resize_for_detection = false    # No additional resizing needed
```

## Usage Examples

### **Basic Usage with Auto-Detection**
```python
from camera_module import RaspberryPiCamera

# Initialize with full auto-detection
camera = RaspberryPiCamera()  # Timezone and resolution auto-detected

# Capture at full resolution  
image = camera.capture_image_rpicam(
    save_to_pictures=True,
    prefix="detection",
    use_full_resolution=True  # Default
)

print(f"Captured: {image.shape[1]}x{image.shape[0]} pixels")
print(f"Timezone: {camera.timezone}")
```

### **Performance Optimized Detection with Rotation**
```python
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector

camera = RaspberryPiCamera()
detector = ObjectDetector()

# Capture full resolution
full_image = camera.capture_image_rpicam(use_full_resolution=True)

# Apply rotation and preprocessing  
corrected_image = camera.preprocess_image(full_image, rotate_degrees=180)

# Resize for fast detection
detection_image = camera.resize_image(corrected_image, (640, 480))

# Detect on small image
detections = detector.detect_objects_contour(detection_image, ColorPresets.RED)

# Scale coordinates back to full resolution
scale_x = corrected_image.shape[1] / detection_image.shape[1]
for detection in detections:
    x, y, w, h = detection['bbox']
    detection['bbox'] = (int(x * scale_x), int(y * scale_y), 
                        int(w * scale_x), int(h * scale_y))

# Draw on full resolution image
result = detector.draw_detections(corrected_image, detections)
``` (default)
    "left_side": 90,    # Camera rotated left
    "right_side": 270   # Camera rotated right
}

# Capture and correct orientation
image = camera.capture_image_rpicam(use_full_resolution=True)
corrected = camera.preprocess_image(image, rotate_degrees=mounting_options["upside_down"])
```

### **Command Line Options**

```bash
# Auto-detection with full resolution (default)
python3 main.py --mode single

# Continuous detection with performance optimization  
python3 main.py --mode continuous --interval 2 --resize-detection

# Override resolution
python3 main.py --mode single --resolution 1080p

# Specific detection method with color
python3 main.py --mode single --method contour --color red

# High performance continuous mode
python3 main.py --mode continuous --interval 0.5 --resolution 720p --resize-detection
```

## Image Quality Comparison

| Mode | Capture Resolution | Detection Resolution | Quality | Speed | Storage |
|------|-------------------|---------------------|---------|-------|---------|
| **Auto Full** | 4608x2592 | 4608x2592 | ⭐⭐⭐⭐⭐ | ⭐⭐ | High |
| **Auto Balanced** | 4608x2592 | 640x480 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High |
| **1080p** | 1920x1080 | 1920x1080 | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium |
| **720p** | 1280x720 | 1280x720 | ⭐⭐⭐ | ⭐⭐⭐⭐ | Low |

## Photo Management

### **Automatic Organization**
```
pictures/
├── 2024-03-15_PST/           # Date + Timezone
│   ├── detection_20240315_143022_123.jpg
│   ├── detection_20240315_143025_456.jpg
│   └── surveillance_20240315_150000_789.jpg
├── 2024-03-16_PST/
│   └── detection_20240316_090000_012.jpg
└── 2024-03-17_UTC/           # Timezone changes reflected
    └── detection_20240317_120000_345.jpg
```

### **Photo Management Tools**
```python
# Get photo collection info
info = camera.get_pictures_info()
print(f"Total photos: {info['total_images']}")
print(f"Storage folders: {len(info['existing_date_folders'])}")

# Cleanup old photos
removed = camera.cleanup_old_pictures(days_to_keep=30)
print(f"Cleaned up {removed} old photos")
```

## Performance Optimization

### **For Maximum Quality**
- Use `use_full_resolution=true` in config
- Disable `resize_for_detection`
- Use contour detection for real-time applications
- Ensure good lighting conditions

### **For Maximum Speed** 
- Enable `resize_for_detection=true`
- Set detection resolution to 640x480 or lower
- Use YOLO only when necessary
- Consider fixed resolution modes

### **For Balanced Performance**
- Use auto-detection with `resize_for_detection=true`
- Detection resolution: 1280x720 or 640x480
- Full resolution capture for storage
- Monitor performance with built-in timing

## Hardware Recommendations

### **For Full Resolution (4608x2592)**
- **Raspberry Pi 5 8GB**: Required for comfortable operation
- **High-speed SD card**: Class 10 U3 or better
- **Active cooling**: Essential for continuous operation
- **Camera Module 3**: Latest 12MP sensor

### **For High Performance**
- **Raspberry Pi 5 4GB**: Sufficient for optimized modes
- **Good SD card**: Class 10 minimum
- **Camera Module 2/3**: Both work well

## Detection Methods Comparison

| Method | Speed | Accuracy | Use Case | Full Res Support |
|--------|-------|----------|----------|-----------------|
| **Contour** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Colored objects | ✅ Excellent |
| **Haar** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Faces, features | ✅ Good |
| **YOLO** | ⭐⭐ | ⭐⭐⭐⭐⭐ | General objects | ⚠️ Requires resize |

## Troubleshooting

### **Auto-Detection Issues**
```bash
# Check detected settings
python3 camera_module.py

# Override if needed
python3 main.py --resolution 1080p --mode single
```

### **Performance Issues**
```bash
# Enable detection resizing
python3 main.py --mode continuous --resize-detection

# Use lower resolution
python3 main.py --resolution 720p --mode continuous
```

### **Storage Issues**
```python
# Check storage usage including rotated images
from camera_module import RaspberryPiCamera
camera = RaspberryPiCamera()
info = camera.get_pictures_info()
print(f"Total photos: {info['total_images']}")

# Cleanup old files (including rotated variants)
removed = camera.cleanup_old_pictures(days_to_keep=7)
```

### **Mounting Orientation Problems**
```bash
# Test all mounting orientations quickly
python3 example_usage.py  # Runs rotation examples

# Or test specific orientation
python3 main.py --mode single --rotation 90 --verbose
```

### **Common Rotation Fixes**

| Problem | Solution | Command |
|---------|----------|---------|
| **Image upside down** | Use 180° rotation | `--rotation 180` |
| **Image sideways (left)** | Use 270° rotation | `--rotation 270` |  
| **Image sideways (right)** | Use 90° rotation | `--rotation 90` |
| **Custom mounting angle** | Use specific degree | `--rotation 45` |

## Advanced Configuration

### **Advanced Configuration**

### **Custom Timezone with Rotation**
```ini
[CAMERA]
timezone = America/New_York  # Override auto-detection
rotation_degrees = 90        # Side-mounted camera
```

### **Performance Tuning with Rotation**
```ini
[CAMERA]
use_full_resolution = true   # Quality
timeout = 2000              # Faster capture
rotation_degrees = 180      # Upside-down mounting

[DETECTION]  
resize_for_detection = true  # Speed
detection_width = 640       # Performance
```

## Rotation Feature Details

### **Supported Rotation Angles**

- **0°**: No rotation (normal orientation)
- **90°**: Counter-clockwise rotation (left side mounting)
- **180°**: Upside-down correction (default for inverted mounting)
- **270°**: Clockwise rotation (right side mounting)
- **Custom**: Any angle (e.g., 45°, 135°) with automatic dimension adjustment

### **Performance Impact**

| Rotation | Processing Time | Notes |
|----------|----------------|-------|
| **0°** | Fastest | No processing needed |
| **90°, 180°, 270°** | Very Fast | Optimized OpenCV operations |
| **Custom angles** | Moderate | General rotation with dimension calculation |

### **Automatic Dimension Handling**

```python
# Original image: 4608x2592
original = camera.capture_image_rpicam()

# 0° rotation: 4608x2592 (no change)  
rotated_0 = camera.preprocess_image(original, rotate_degrees=0)

# 90° rotation: 2592x4608 (dimensions swapped)
rotated_90 = camera.preprocess_image(original, rotate_degrees=90)

# 180° rotation: 4608x2592 (dimensions preserved)
rotated_180 = camera.preprocess_image(original, rotate_degrees=180)

# 45° custom: Larger canvas to prevent cropping
rotated_45 = camera.preprocess_image(original, rotate_degrees=45)
```

## API Integration

```python
# Initialize with auto-detection and apply rotation
camera = RaspberryPiCamera()

# Full resolution capture with rotation
image = camera.capture_image_rpicam(use_full_resolution=True)
corrected_image = camera.preprocess_image(image, rotate_degrees=180)

# Performance optimized detection  
detection_img = camera.resize_image(corrected_image, (640, 480))
detections = detector.detect_objects_contour(detection_img)

# Scale results back to full resolution
scale_factor = corrected_image.shape[1] / detection_img.shape[1]
for detection in detections:
    # Scale bounding box coordinates
    pass

# Save full resolution result with rotation applied
camera.save_image(corrected_image, prefix="detection_result")
```

## Troubleshooting

### **Rotation Issues**
```python
# Test different rotations to find correct orientation
from camera_module import RaspberryPiCamera

camera = RaspberryPiCamera()
image = camera.capture_image_rpicam()

# Test all standard rotations
for angle in [0, 90, 180, 270]:
    rotated = camera.preprocess_image(image, rotate_degrees=angle)
    camera.save_image(rotated, prefix=f"test_rotation_{angle}deg")
    print(f"Saved test image with {angle}° rotation")
```

## License

This project is open source and available under the MIT License.

---

**🎯 Now with Auto-Detection and Full Resolution Support! 📸**