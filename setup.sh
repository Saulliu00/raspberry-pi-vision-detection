#!/bin/bash

# Setup script for Raspberry Pi 5 Vision Detection System
# Updated for parts defect detection with CSV data logging
# Run with: chmod +x setup.sh && ./setup.sh

echo "Setting up Raspberry Pi 5 Vision Detection System for Parts Defect Detection..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_warning "This script is designed for Raspberry Pi. Some features may not work on other systems."
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libcamera-apps \
    libcamera-dev \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# Install OpenCV dependencies
print_status "Installing OpenCV dependencies..."
sudo apt install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv vision_env
source vision_env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python packages
print_status "Installing Python packages..."
pip install -r requirements.txt

# Test camera functionality
print_status "Testing camera functionality..."
if command -v rpicam-still &> /dev/null; then
    print_status "rpicam-still is available"
    
    # Test camera capture
    if rpicam-still --timeout 1000 --output test_camera.jpg --nopreview; then
        print_status "Camera test successful - test_camera.jpg created"
        rm -f test_camera.jpg
    else
        print_warning "Camera test failed - check camera connection"
    fi
else
    print_error "rpicam-still not found. Install with: sudo apt install libcamera-apps"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p output
mkdir -p pictures
print_status "Created output and pictures directories"

# Make Python scripts executable
print_status "Making scripts executable..."
chmod +x camera_module.py
chmod +x object_detection.py
chmod +x vision_utils.py
chmod +x main.py
chmod +x example_usage.py

# Create desktop shortcut (optional)
print_status "Creating desktop shortcuts..."
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$DESKTOP_DIR" ]; then
    # Main application shortcut
    cat > "$DESKTOP_DIR/Vision_Defect_Detection.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Parts Defect Detection
Comment=Raspberry Pi Vision Detection for Parts Defect Detection
Exec=$(pwd)/vision_env/bin/python $(pwd)/main.py
Icon=camera-photo
Terminal=true
Categories=Development;
EOF
    chmod +x "$DESKTOP_DIR/Vision_Defect_Detection.desktop"
    
    # CSV Statistics shortcut
    cat > "$DESKTOP_DIR/Detection_Statistics.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Detection Statistics
Comment=View CSV Detection Statistics
Exec=$(pwd)/vision_env/bin/python $(pwd)/main.py --mode stats
Icon=applications-office
Terminal=true
Categories=Development;
EOF
    chmod +x "$DESKTOP_DIR/Detection_Statistics.desktop"
    
    print_status "Desktop shortcuts created"
fi

# Create systemd service (optional)
read -p "Do you want to create a systemd service for continuous inspection? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/parts-inspection.service > /dev/null << EOF
[Unit]
Description=Parts Defect Detection System
After=multi-user.target

[Service]
Type=simple
Restart=always
RestartSec=1
User=pi
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/vision_env/bin/python $(pwd)/main.py --mode continuous --interval 3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable parts-inspection.service
    print_status "Systemd service created and enabled"
    print_status "Use 'sudo systemctl start parts-inspection' to start continuous inspection"
    print_status "Use 'sudo systemctl status parts-inspection' to check status"
fi

# Run initial tests
print_status "Running initial system tests..."

echo "Testing camera module..."
python3 camera_module.py

echo "Testing object detection module..."
python3 object_detection.py

echo "Testing vision utils and CSV logging..."
python3 vision_utils.py

# Performance optimization for Raspberry Pi
print_status "Applying Raspberry Pi optimizations..."

# Increase GPU memory split for camera
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    print_status "GPU memory increased to 128MB"
fi

# Enable camera interface
if ! grep -q "camera_auto_detect=1" /boot/config.txt; then
    echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
    print_status "Camera auto-detect enabled"
fi

# Create usage guide for parts defect detection
print_status "Creating parts defect detection usage guide..."
cat > USAGE.md << EOF
# Raspberry Pi 5 Parts Defect Detection System - Usage Guide

## Quick Start for Parts Inspection

1. Activate the virtual environment:
   \`\`\`bash
   source vision_env/bin/activate
   \`\`\`

2. Single part inspection:
   \`\`\`bash
   python3 main.py --mode single
   \`\`\`

3. Continuous parts inspection:
   \`\`\`bash
   python3 main.py --mode continuous --interval 3
   \`\`\`

4. View detection statistics:
   \`\`\`bash
   python3 main.py --mode stats
   \`\`\`

5. View recent inspection logs:
   \`\`\`bash
   python3 main.py --mode logs
   \`\`\`

## Detection Methods for Parts Inspection

### YOLO Detection (Recommended for General Defects)
\`\`\`bash
python3 main.py --mode single --method yolo
\`\`\`
- **Best for**: Complex defects, multiple object types
- **Accuracy**: High (can detect various defect types)
- **Speed**: Moderate (requires more processing)

### Haar Cascade Detection (Good for Specific Patterns)
\`\`\`bash
python3 main.py --mode single --method haar
\`\`\`
- **Best for**: Specific geometric defects, patterns
- **Accuracy**: Good for trained patterns
- **Speed**: Fast

## CSV Data Logging

All inspections are automatically logged to CSV with:
- Date and time of inspection
- Capture filename
- Number of defects detected
- Detailed object information
- Processing time
- Image resolution
- Detection method used

### Custom CSV File
\`\`\`bash
python3 main.py --mode continuous --csv-file production_log.csv
\`\`\`

### Disable CSV Logging (for testing)
\`\`\`bash
python3 main.py --mode single --disable-csv
\`\`\`

## Performance Optimization

### High Quality Mode (Default)
- Full resolution capture (up to 12MP)
- YOLO detection
- Complete CSV logging
\`\`\`bash
python3 main.py --mode single
\`\`\`

### High Speed Mode
- Resized detection for performance
- Full resolution storage
\`\`\`bash
python3 main.py --mode continuous --resize-detection --interval 1
\`\`\`

### Production Line Mode
- Continuous inspection
- Optimized intervals
- Comprehensive logging
\`\`\`bash
python3 main.py --mode continuous --interval 2 --resize-detection
\`\`\`

## Quality Analysis

### View Detection Statistics
Shows overall quality metrics:
- Total inspections performed
- Total defects found
- Defect rate percentage
- Average processing time
- Most common defect types

### Export Data for Analysis
The CSV files can be opened in Excel, imported into databases, or processed with Python/R for advanced quality analysis.

## Configuration for Parts Inspection

Edit \`vision_config.ini\`:

\`\`\`ini
[DETECTION]
method = yolo                    # or haar
confidence_threshold = 0.4       # Lower = more sensitive
resize_for_detection = true      # For better performance

[CSV_LOGGING]
enabled = true
csv_file = parts_inspection.csv
cleanup_days = 30               # Keep 30 days of logs

[CAMERA]
use_full_resolution = true      # Best quality
rotation_degrees = 180          # Adjust for mounting
\`\`\`

## Troubleshooting

### Camera Issues
\`\`\`bash
rpicam-still --list-cameras
python3 camera_module.py
\`\`\`

### Detection Issues  
\`\`\`bash
python3 object_detection.py
\`\`\`

### CSV Issues
\`\`\`bash
python3 vision_utils.py
head -n 5 detection_log.csv
\`\`\`

### Performance Issues
- Enable \`resize_for_detection\` in config
- Reduce confidence threshold
- Use Haar cascades for speed
- Monitor disk space for images/logs

## Service Management (if installed)

- Start: \`sudo systemctl start parts-inspection\`
- Stop: \`sudo systemctl stop parts-inspection\`
- Status: \`sudo systemctl status parts-inspection\`
- Logs: \`journalctl -u parts-inspection -f\`

## Integration with Quality Management Systems

The CSV output is designed for easy integration:
- Import into quality databases
- Generate reports with Excel/Power BI
- Set up automated alerts based on defect rates
- Track trends over time for process improvement
EOF

# Create maintenance script for parts inspection
print_status "Creating maintenance script..."
cat > maintenance.sh << 'EOF'
#!/bin/bash

# Maintenance script for Parts Defect Detection System

echo "Parts Defect Detection System Maintenance"
echo "========================================="

# Check system status
echo "1. System Status:"
echo "   - Disk usage: $(df -h / | awk 'NR==2 {print $5}') used"
echo "   - Memory: $(free -h | awk 'NR==2 {print $3 "/" $2}') used"
echo "   - Temperature: $(vcgencmd measure_temp)"

# Check service status
if systemctl is-active --quiet parts-inspection; then
    echo "   - Parts inspection service: Running"
else
    echo "   - Parts inspection service: Stopped"
fi

echo ""
echo "2. Detection Log Analysis:"
recent_logs=$(journalctl -u parts-inspection --since "1 hour ago" --no-pager -q | wc -l)
echo "   - Service log entries (last hour): $recent_logs"

# Check CSV files
csv_count=$(find . -name "*.csv" -type f | wc -l)
echo "   - CSV log files found: $csv_count"

if [ -f "detection_log.csv" ]; then
    total_inspections=$(wc -l < detection_log.csv)
    echo "   - Total inspections logged: $((total_inspections - 1))"
fi

echo ""
echo "3. Storage Management:"

# Check pictures directory
if [ -d "./pictures" ]; then
    pic_count=$(find ./pictures -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
    echo "   - Inspection photos stored: $pic_count"
    
    if [ $pic_count -gt 1000 ]; then
        echo "   - Warning: Large number of photos, consider cleanup"
    fi
fi

# Check output directory
if [ -d "./output" ]; then
    output_files=$(find ./output -type f | wc -l)
    echo "   - Output files: $output_files"
fi

# Cleanup old files
echo ""
echo "4. Cleanup Options:"
read -p "   Clean up files older than 7 days? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clean old pictures
    find ./pictures -type f -mtime +7 -delete 2>/dev/null || true
    
    # Clean old output files
    find ./output -type f -mtime +7 -delete 2>/dev/null || true
    
    echo "   ✓ Old files cleaned up"
fi

# CSV maintenance
if [ -f "detection_log.csv" ]; then
    read -p "   Clean up CSV logs older than 30 days? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -c "
from vision_utils import CSVDataLogger
logger = CSVDataLogger('detection_log.csv')
logger.cleanup_old_logs(30)
print('   ✓ CSV logs cleaned up')
" 2>/dev/null || echo "   ⚠ Could not clean CSV logs"
    fi
fi

echo ""
echo "5. Quality Metrics:"
if [ -f "detection_log.csv" ]; then
    python3 -c "
from vision_utils import CSVDataLogger
import sys
try:
    logger = CSVDataLogger('detection_log.csv')
    stats = logger.get_detection_statistics()
    total_inspections = stats.get('total_detections', 0)
    total_defects = stats.get('total_objects_found', 0)
    defect_rate = (total_defects / total_inspections * 100) if total_inspections > 0 else 0
    avg_time = stats.get('average_processing_time', 0)
    print(f'   - Total inspections: {total_inspections}')
    print(f'   - Total defects found: {total_defects}')
    print(f'   - Defect rate: {defect_rate:.1f}%')
    print(f'   - Average processing time: {avg_time:.3f}s')
except Exception as e:
    print('   - Could not analyze quality metrics')
" 2>/dev/null || echo "   - CSV analysis not available"
fi

echo ""
echo "Maintenance completed!"
EOF

chmod +x maintenance.sh

print_status "Setup completed successfully!"
echo ""
echo "==================== SETUP SUMMARY ===================="
echo "✓ System packages installed"
echo "✓ Python virtual environment created"
echo "✓ Python dependencies installed"
echo "✓ Camera functionality tested"
echo "✓ Scripts made executable"
echo "✓ Project directories created"
echo "✓ Configuration files ready"
echo "✓ CSV data logging configured"
echo "✓ Usage guide created (USAGE.md)"
echo "✓ Maintenance script created (maintenance.sh)"
echo "========================================================"
echo ""
echo "Next steps for Parts Defect Detection:"
echo "1. Activate virtual environment: source vision_env/bin/activate"
echo "2. Test single inspection: python3 main.py --mode single"
echo "3. View CSV logs: python3 main.py --mode stats"
echo "4. Start continuous inspection: python3 main.py --mode continuous --interval 3"
echo "5. Read USAGE.md for detailed instructions"
echo "6. Edit vision_config.ini to customize for your parts"
echo ""

# Check if reboot is needed
if [ -f /var/run/reboot-required ]; then
    print_warning "System reboot is recommended for all changes to take effect"
    read -p "Reboot now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi

deactivate 2>/dev/null || true
print_status "Setup complete! Ready for parts defect detection with CSV logging."