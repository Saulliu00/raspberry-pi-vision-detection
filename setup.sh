#!/bin/bash

# Setup script for Raspberry Pi 5 Vision Detection System
# Run with: chmod +x setup.sh && ./setup.sh

echo "Setting up Raspberry Pi 5 Vision Detection System..."

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

# Create output directory
print_status "Creating output directory..."
mkdir -p output

# Make Python scripts executable
print_status "Making scripts executable..."
chmod +x camera_module.py
chmod +x object_detection.py
chmod +x vision_utils.py
chmod +x main.py

# Create desktop shortcut (optional)
print_status "Creating desktop shortcuts..."
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$DESKTOP_DIR" ]; then
    cat > "$DESKTOP_DIR/Vision_System.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Vision Detection System
Comment=Raspberry Pi Vision Detection
Exec=$(pwd)/vision_env/bin/python $(pwd)/main.py
Icon=camera-photo
Terminal=true
Categories=Development;
EOF
    chmod +x "$DESKTOP_DIR/Vision_System.desktop"
    print_status "Desktop shortcut created"
fi

# Create systemd service (optional)
read -p "Do you want to create a systemd service for auto-start? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/vision-detection.service > /dev/null << EOF
[Unit]
Description=Raspberry Pi Vision Detection System
After=multi-user.target

[Service]
Type=simple
Restart=always
RestartSec=1
User=pi
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/vision_env/bin/python $(pwd)/main.py --mode continuous --interval 2
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable vision-detection.service
    print_status "Systemd service created and enabled"
    print_status "Use 'sudo systemctl start vision-detection' to start the service"
    print_status "Use 'sudo systemctl status vision-detection' to check status"
fi

# Run initial tests
print_status "Running initial system tests..."

echo "Testing camera module..."
python3 camera_module.py

echo "Testing object detection module..."
python3 object_detection.py

echo "Testing vision utils module..."
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

# Create usage guide
print_status "Creating usage guide..."
cat > USAGE.md << EOF
# Raspberry Pi 5 Vision Detection System - Usage Guide

## Quick Start

1. Activate the virtual environment:
   \`\`\`bash
   source vision_env/bin/activate
   \`\`\`

2. Run single detection:
   \`\`\`bash
   python3 main.py --mode single
   \`\`\`

3. Run continuous detection:
   \`\`\`bash
   python3 main.py --mode continuous --interval 2
   \`\`\`

## Testing Individual Modules

- Test camera: \`python3 camera_module.py\`
- Test detection: \`python3 object_detection.py\`
- Test utilities: \`python3 vision_utils.py\`

## Command Line Options

- \`--config CONFIG_FILE\`: Specify configuration file
- \`--mode {single,continuous}\`: Detection mode
- \`--interval SECONDS\`: Detection interval for continuous mode
- \`--method {yolo,haar,contour}\`: Detection method
- \`--color {red,green,blue,yellow,orange}\`: Target color for contour detection
- \`--verbose\`: Enable verbose logging

## Configuration

Edit \`vision_config.ini\` to customize:
- Camera settings (resolution, timeout)
- Detection parameters (method, thresholds)
- Output settings (save images, directory)
- Color ranges for contour detection

## Examples

1. Detect red objects continuously every 3 seconds:
   \`\`\`bash
   python3 main.py --mode continuous --interval 3 --method contour --color red
   \`\`\`

2. Single detection with YOLO:
   \`\`\`bash
   python3 main.py --mode single --method yolo
   \`\`\`

3. Face detection with Haar cascades:
   \`\`\`bash
   python3 main.py --mode single --method haar
   \`\`\`

## Troubleshooting

- Camera not working: Check \`rpicam-still --list-cameras\`
- Permission errors: Make sure scripts are executable
- Package issues: Reinstall with \`pip install -r requirements.txt\`
- Low performance: Reduce image resolution in config

## Service Management (if installed)

- Start: \`sudo systemctl start vision-detection\`
- Stop: \`sudo systemctl stop vision-detection\`
- Status: \`sudo systemctl status vision-detection\`
- Logs: \`journalctl -u vision-detection -f\`
EOF

# Create maintenance script
print_status "Creating maintenance script..."
cat > maintenance.sh << 'EOF'
#!/bin/bash

# Maintenance script for Vision Detection System

echo "Vision Detection System Maintenance"
echo "=================================="

# Check system status
echo "1. System Status:"
echo "   - Disk usage: $(df -h / | awk 'NR==2 {print $5}') used"
echo "   - Memory: $(free -h | awk 'NR==2 {print $3 "/" $2}') used"
echo "   - Temperature: $(vcgencmd measure_temp)"

# Check service status
if systemctl is-active --quiet vision-detection; then
    echo "   - Vision service: Running"
else
    echo "   - Vision service: Stopped"
fi

echo ""
echo "2. Log Analysis:"
recent_logs=$(journalctl -u vision-detection --since "1 hour ago" --no-pager -q | wc -l)
echo "   - Log entries (last hour): $recent_logs"

echo ""
echo "3. Storage Cleanup:"
output_files=$(find ./output -type f -mtime +7 | wc -l)
if [ $output_files -gt 0 ]; then
    echo "   - Old output files (>7 days): $output_files"
    read -p "   Remove old files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find ./output -type f -mtime +7 -delete
        echo "   ✓ Old files removed"
    fi
else
    echo "   - No old files to clean"
fi

echo ""
echo "4. Update Check:"
cd "$(dirname "$0")"
if git status &>/dev/null; then
    git fetch --dry-run 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   - Git repository: Up to date"
    else
        echo "   - Git repository: Updates available"
    fi
else
    echo "   - Not a git repository"
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
echo "✓ Output directory created"
echo "✓ Configuration files generated"
echo "✓ Usage guide created (USAGE.md)"
echo "✓ Maintenance script created (maintenance.sh)"
echo "========================================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source vision_env/bin/activate"
echo "2. Test the system: python3 main.py --mode single"
echo "3. Read USAGE.md for detailed instructions"
echo "4. Edit vision_config.ini to customize settings"
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
print_status "Setup script finished. Happy detecting!"