# Raspberry Pi 5 Vision Detection System for Parts Defect Detection

A comprehensive computer vision system designed specifically for Raspberry Pi 5 with camera module for **parts defect detection**. This system provides YOLO and Haar cascade detection methods with **automatic CSV data logging** and **full resolution capture** for maximum image quality.

## 🆕 Key Features

- **🔍 Parts Defect Detection**: Optimized for industrial parts inspection and defect identification
- **📊 CSV Data Logging**: Automatic logging of detection results with date, time, filename, and results
- **📸 Full Resolution**: Uses camera's maximum resolution by default (up to 12MP on Camera Module 3)
- **🤖 Multiple Detection Methods**: YOLO and Haar cascade detection
- **⚡ Smart Performance**: Optional detection resizing for speed while maintaining image quality
- **🌍 Timezone Aware**: Auto-detects system timezone for accurate logging
- **📁 Structured Storage**: Organized photo storage with date-based folders
- **🎛️ Flexible Configuration**: Easy switching between quality and performance modes

## Detection Methods

### **YOLO (You Only Look Once)**
- **Best for**: General object detection, complex parts identification
- **Pros**: High accuracy, can detect multiple object classes
- **Cons**: Requires more processing power
- **Use Case**: Complex industrial parts with various defect types

### **Haar Cascades** 
- **Best for**: Specific feature detection (faces, edges, patterns)
- **Pros**: Fast, lightweight, works well for trained patterns
- **Cons**: Requires specific training for each defect type
- **Use Case**: Specific defect patterns or geometric anomalies

## CSV Data Logging

The system automatically logs all detection results to a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Detection date | 2024-03-15 |
| `time` | Detection time | 14:30:22 |
| `capture_filename` | Image filename | detection_20240315_143022_123.jpg |
| `detection_count` | Number of objects detected | 2 |
| `detected_objects` | Objects with confidence scores | defect(0.95); part(0.87) |
| `processing_time` | Processing time in seconds | 0.245 |
| `image_resolution` | Image resolution | 1920x1080 |
| `detection_method` | Detection method used | yolo |

## System Requirements

- Raspberry Pi 5 (recommended) or Raspberry Pi 4
- Raspberry Pi Camera Module (v2 or v3) 
- Python 3.8+
- At least 4GB RAM (8GB recommended for YOLO)
- 32GB+ SD card (for full resolution images and logs)

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
├── camera_module.py         # Camera operations with auto-detection
├── object_detection.py      # Object detection algorithms (YOLO, Haar)
├── vision_utils.py          # Utilities and CSV data logger
├── main.py                  # Main system integration
├── requirements.txt         # Python dependencies
├── setup.sh                # Automated setup script
├── vision_config.ini       # Configuration file (auto-generated)
├── detection_log.csv       # CSV data log (auto-generated)
├── pictures/               # Full resolution photos (organized by date)
├── output/                 # Detection results and metadata
└── README.md              # This file
```

## 🚀 Quick Start Examples

### **Basic Single Detection**
```bash
# Single detection with auto-logging to CSV
python3 main.py --mode single
```

### **Continuous Parts Inspection**
```bash
# Continuous detection every 2 seconds with CSV logging
python3 main.py --mode continuous --interval 2
```

### **High Performance Mode**
```bash
# Full resolution capture, resized detection for speed
python3 main.py --mode continuous --interval 1 --resize-detection
```

### **View Detection Statistics**
```bash
# Show CSV log statistics
python3 main.py --mode stats
```

### **View Recent Detection Logs**
```bash
# Show last 20 detection entries
python3 main.py --mode logs
```

## Configuration Examples

### **Maximum Quality for Parts Inspection (Default)**
```ini
[CAMERA]
width = auto                    # Auto-detect native resolution
height = auto                   # Auto-detect native resolution  
use_full_resolution = true      # Use camera's maximum resolution
rotation_degrees = 180          # Rotate for mounting orientation

[DETECTION]
method = yolo                   # YOLO for complex defect detection
resize_for_detection = false    # Detect on full resolution

[CSV_LOGGING]
enabled = true                  # Enable CSV data logging
csv_file = detection_log.csv    # CSV log file name
```

### **High Performance Mode**
```ini
[CAMERA] 
use_full_resolution = true      # Capture at full resolution

[DETECTION]
method = yolo                   # YOLO detection
resize_for_detection = true     # Resize for detection speed
detection_width = 640           # Detection resolution
detection_height = 480

[CSV_LOGGING]
enabled = true                  # Keep CSV logging enabled
```

### **Haar Cascade for Specific Defects**
```ini
[DETECTION]
method = haar                   # Use Haar cascade detection
haar_cascade = frontalface_default  # Or custom trained cascade

[CSV_LOGGING]
enabled = true
csv_file = parts_inspection_log.csv
```

## Usage Examples

### **Basic Parts Inspection**
```python
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import CSVDataLogger

# Initialize components
camera = RaspberryPiCamera()  # Auto-detect settings
detector = ObjectDetector()
csv_logger = CSVDataLogger("parts_inspection.csv")

# Load YOLO model
detector.load_yolo_model()

# Capture and detect
image = camera.capture_image_rpicam(save_to_pictures=True, prefix="parts_check")
detections = detector.detect_objects_yolo(image)

# Log results
csv_logger.log_detection(
    capture_filename="parts_check_001.jpg",
    detections=detections,
    processing_time=0.245,
    image_resolution=(image.shape[1], image.shape[0]),
    detection_method="yolo"
)

print(f"Found {len(detections)} objects/defects")
```

### **Command Line Usage**

```bash
# YOLO detection with full resolution
python3 main.py --mode single --method yolo

# Haar cascade detection for specific patterns  
python3 main.py --mode single --method haar

# Continuous inspection every 5 seconds
python3 main.py --mode continuous --interval 5 --method yolo

# Performance mode with detection resizing
python3 main.py --mode continuous --resize-detection --interval 1

# Custom CSV file and disable for testing
python3 main.py --mode single --csv-file test_run.csv
python3 main.py --mode single --disable-csv

# View system statistics
python3 main.py --mode stats

# View recent detection logs
python3 main.py --mode logs
```

## CSV Data Analysis

### **View Statistics**
```bash
python3 main.py --mode stats
```

Output:
```
--- CSV Log Statistics ---
Total detections: 150
Total objects found: 45
Average processing time: 0.234s
Most common object: defect
Detection methods used: yolo, haar
Date range: 2024-03-10 to 2024-03-15
```

### **View Recent Logs**
```bash
python3 main.py --mode logs
```

### **Programmatic CSV Analysis**
```python
from vision_utils import CSVDataLogger

logger = CSVDataLogger("detection_log.csv")

# Get recent detections
recent = logger.get_recent_logs(10)
for log in recent:
    print(f"{log['date']} {log['time']}: {log['detection_count']} objects")

# Get statistics
stats = logger.get_detection_statistics()
print(f"Total defects found: {stats['total_objects_found']}")
print(f"Average detection time: {stats['average_processing_time']:.3f}s")
```

## Performance Optimization

### **For Maximum Quality**
- Use `method = yolo` with `use_full_resolution = true`
- Disable `resize_for_detection` 
- Ensure good lighting conditions
- Use Raspberry Pi 5 8GB for best performance

### **For Maximum Speed**
- Enable `resize_for_detection = true`
- Set detection resolution to 640x480
- Use `method = haar` for specific defects
- Consider `method = yolo` with resizing for general detection

### **For Continuous Inspection**
- Use performance mode with 1-2 second intervals
- Enable CSV logging for quality tracking
- Monitor disk space for images and logs
- Set up log cleanup with `cleanup_days`

## Detection Method Comparison

| Method | Speed | Accuracy | Best Use Case | Training Required |
|--------|-------|----------|---------------|-------------------|
| **YOLO** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Complex defects, multiple object types | Pre-trained available |
| **Haar** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Specific patterns, geometric defects | Yes, for custom defects |

## Troubleshooting

### **Camera Issues**
```bash
# Test camera
rpicam-still --list-cameras
rpicam-still --timeout 1000 --output test.jpg

# Check camera connection
python3 camera_module.py
```

### **Detection Issues**
```bash
# Test object detection
python3 object_detection.py

# Check YOLO model files
ls -la *.weights *.cfg *.names
```

### **CSV Logging Issues**
```bash
# Check CSV file
head -n 5 detection_log.csv

# Test CSV logging
python3 -c "from vision_utils import CSVDataLogger; CSVDataLogger('test.csv')"
```

### **Performance Issues**
- **High memory usage**: Enable `resize_for_detection`
- **Slow detection**: Use Haar cascades or reduce resolution
- **Storage full**: Enable log cleanup or reduce image resolution

## Advanced Features

### **Custom YOLO Models**
Train custom YOLO models for specific defect types:
1. Collect defect images
2. Label with tools like LabelImg
3. Train custom YOLO model
4. Replace model files in system

### **Custom Haar Cascades**  
Train Haar cascades for specific defect patterns:
1. Collect positive/negative samples
2. Use OpenCV training tools
3. Load custom cascade in configuration

### **Integration with Quality Systems**
```python
# Export CSV data for quality analysis
import pandas as pd

df = pd.read_csv('detection_log.csv')
daily_defects = df.groupby('date')['detection_count'].sum()
print("Daily defect counts:")
print(daily_defects)
```

## System Monitoring

### **Service Management** (if systemd service installed)
```bash
# Start continuous inspection service
sudo systemctl start vision-detection

# Check service status
sudo systemctl status vision-detection

# View live logs
journalctl -u vision-detection -f

# Stop service
sudo systemctl stop vision-detection
```

### **Maintenance**
```bash
# Run maintenance script
./maintenance.sh

# Manual cleanup
python3 -c "from vision_utils import CSVDataLogger; CSVDataLogger().cleanup_old_logs(7)"
```

## License

This project is open source and available under the MIT License.

---

**🔧 Now optimized for industrial parts defect detection with comprehensive CSV data logging! 📊**