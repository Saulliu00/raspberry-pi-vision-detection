#!/usr/bin/env python3
"""
Utility functions for the Raspberry Pi 5 vision detection system.
Contains helper functions for file management, configuration, and image processing.
"""

import os
import json
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import configparser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Handles configuration file management."""
    
    def __init__(self, config_file: str = "vision_config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default."""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            logger.info(f"Configuration loaded from {self.config_file}")
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration file."""
        self.config['CAMERA'] = {
            'width': 'auto',  # 'auto' for native resolution, or specific width
            'height': 'auto',  # 'auto' for native resolution, or specific height
            'timeout': '3000',
            'method': 'rpicam',  # or 'opencv'
            'timezone': 'auto',   # 'auto' for system detection, or specific timezone
            'pictures_directory': 'pictures',
            'save_to_pictures': 'true',
            'use_full_resolution': 'true',  # Use camera's native resolution for best quality
            'rotation_degrees': '180'  # Rotate image (0, 90, 180, 270 degrees)
        }
        
        self.config['DETECTION'] = {
            'method': 'contour',  # 'yolo', 'haar', 'contour'
            'confidence_threshold': '0.5',
            'nms_threshold': '0.4',
            'min_area': '500',
            'resize_for_detection': 'false',  # Resize image for detection performance
            'detection_width': '640',  # Width to resize to for detection (if enabled)
            'detection_height': '480'  # Height to resize to for detection (if enabled)
        }
        
        self.config['OUTPUT'] = {
            'save_images': 'true',
            'output_directory': './output',
            'draw_detections': 'true',
            'save_logs': 'true'
        }
        
        self.config['COLORS'] = {
            'red_lower': '0,50,50',
            'red_upper': '10,255,255',
            'green_lower': '40,50,50',
            'green_upper': '80,255,255',
            'blue_lower': '100,50,50',
            'blue_upper': '130,255,255'
        }
        
        self.save_config()
        logger.info(f"Default configuration created: {self.config_file}")
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, fallback=None):
        """Get configuration value."""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback=0):
        """Get integer configuration value."""
        return self.config.getint(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback=False):
        """Get boolean configuration value."""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def set(self, section: str, key: str, value: str):
        """Set configuration value."""
        if section not in self.config:
            self.config.add_section(section)
        self.config.set(section, key, value)
        self.save_config()


class FileManager:
    """Handles file operations and directory management."""
    
    def __init__(self, base_directory: str = "./output"):
        self.base_directory = base_directory
        self.ensure_directory_exists(base_directory)
    
    def ensure_directory_exists(self, directory: str):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    def get_timestamp_filename(self, prefix: str = "image", extension: str = ".jpg") -> str:
        """Generate filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{prefix}_{timestamp}{extension}"
    
    def save_image_with_metadata(self, image: np.ndarray, 
                                detections: List[Dict], 
                                prefix: str = "detection") -> Tuple[str, str]:
        """
        Save image and its detection metadata.
        
        Args:
            image: Image to save
            detections: Detection results
            prefix: Filename prefix
            
        Returns:
            Tuple of (image_path, metadata_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save image
        image_filename = f"{prefix}_{timestamp}.jpg"
        image_path = os.path.join(self.base_directory, image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata_filename = f"{prefix}_{timestamp}.json"
        metadata_path = os.path.join(self.base_directory, metadata_filename)
        
        metadata = {
            'timestamp': timestamp,
            'image_file': image_filename,
            'image_shape': image.shape,
            'detections': detections,
            'detection_count': len(detections)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved image and metadata: {image_filename}")
        return image_path, metadata_path
    
    def load_detection_history(self, limit: int = 100) -> List[Dict]:
        """Load recent detection history from metadata files."""
        metadata_files = []
        
        for filename in os.listdir(self.base_directory):
            if filename.endswith('.json') and 'detection_' in filename:
                filepath = os.path.join(self.base_directory, filename)
                metadata_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time, newest first
        metadata_files.sort(key=lambda x: x[1], reverse=True)
        
        history = []
        for filepath, _ in metadata_files[:limit]:
            try:
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    history.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {filepath}: {e}")
        
        return history
    
    def cleanup_old_files(self, max_age_days: int = 7):
        """Remove files older than specified days."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
        
        removed_count = 0
        for filename in os.listdir(self.base_directory):
            filepath = os.path.join(self.base_directory, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {filepath}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old files")


class ImageProcessor:
    """Advanced image processing utilities."""
    
    @staticmethod
    def apply_filters(image: np.ndarray, filter_type: str = "enhance") -> np.ndarray:
        """
        Apply various image filters.
        
        Args:
            image: Input image
            filter_type: Type of filter ('enhance', 'sharpen', 'blur', 'edge')
            
        Returns:
            Filtered image
        """
        try:
            if filter_type == "enhance":
                # Enhance contrast and brightness
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            elif filter_type == "sharpen":
                kernel = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])
                return cv2.filter2D(image, -1, kernel)
            
            elif filter_type == "blur":
                return cv2.GaussianBlur(image, (15, 15), 0)
            
            elif filter_type == "edge":
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            else:
                return image
                
        except Exception as e:
            logger.error(f"Filter application error: {e}")
            return image
    
    @staticmethod
    def create_color_mask(image: np.ndarray, 
                         lower_bound: np.ndarray, 
                         upper_bound: np.ndarray) -> np.ndarray:
        """Create color-based mask for object detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    @staticmethod
    def calculate_object_properties(contour) -> Dict:
        """Calculate properties of a detected object from its contour."""
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Aspect ratio
            aspect_ratio = w / h if h != 0 else 0
            
            # Extent (object area / bounding rectangle area)
            extent = area / (w * h) if (w * h) != 0 else 0
            
            # Solidity (object area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
            
            return {
                'area': area,
                'perimeter': perimeter,
                'centroid': (cx, cy),
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity
            }
            
        except Exception as e:
            logger.error(f"Error calculating object properties: {e}")
            return {}
    
    @staticmethod
    def resize_maintain_aspect(image: np.ndarray, 
                              target_width: int = None, 
                              target_height: int = None) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        h, w = image.shape[:2]
        
        if target_width is None and target_height is None:
            return image
        
        if target_width is None:
            # Calculate width based on target height
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
        
        elif target_height is None:
            # Calculate height based on target width
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
        
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


class PerformanceMonitor:
    """Monitor system performance and processing times."""
    
    def __init__(self):
        self.timing_data = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return elapsed time."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            
            if operation not in self.timing_data:
                self.timing_data[operation] = []
            
            self.timing_data[operation].append(elapsed)
            del self.start_times[operation]
            
            return elapsed
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average processing time for an operation."""
        if operation in self.timing_data and self.timing_data[operation]:
            return sum(self.timing_data[operation]) / len(self.timing_data[operation])
        return 0.0
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance report."""
        report = {}
        
        for operation, times in self.timing_data.items():
            if times:
                report[operation] = {
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times),
                    'total': sum(times)
                }
        
        return report
    
    def log_performance(self):
        """Log performance statistics."""
        report = self.get_performance_report()
        logger.info("Performance Report:")
        for operation, stats in report.items():
            logger.info(f"  {operation}: avg={stats['average']:.3f}s, "
                       f"min={stats['min']:.3f}s, max={stats['max']:.3f}s, "
                       f"count={stats['count']}")


class ColorPresets:
    """Predefined color ranges for common object detection."""
    
    # HSV color ranges
    RED = {
        'lower': np.array([0, 50, 50]),
        'upper': np.array([10, 255, 255])
    }
    
    RED_WRAP = {  # For red colors that wrap around in HSV
        'lower': np.array([170, 50, 50]),
        'upper': np.array([180, 255, 255])
    }
    
    GREEN = {
        'lower': np.array([40, 50, 50]),
        'upper': np.array([80, 255, 255])
    }
    
    BLUE = {
        'lower': np.array([100, 50, 50]),
        'upper': np.array([130, 255, 255])
    }
    
    YELLOW = {
        'lower': np.array([20, 50, 50]),
        'upper': np.array([30, 255, 255])
    }
    
    ORANGE = {
        'lower': np.array([10, 50, 50]),
        'upper': np.array([25, 255, 255])
    }
    
    @classmethod
    def get_color_range(cls, color_name: str) -> Optional[Dict]:
        """Get color range by name."""
        return getattr(cls, color_name.upper(), None)


def test_vision_utils():
    """Test function to verify utility functions."""
    print("Testing Vision Utils Module...")
    
    # Test ConfigManager
    print("\n1. Testing ConfigManager...")
    config = ConfigManager("test_config.ini")
    print(f"✓ Camera width: {config.getint('CAMERA', 'width')}")
    print(f"✓ Detection method: {config.get('DETECTION', 'method')}")
    print(f"✓ Save images: {config.getboolean('OUTPUT', 'save_images')}")
    
    # Test FileManager
    print("\n2. Testing FileManager...")
    file_manager = FileManager("./test_output")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_detections = [
        {'class': 'test_object', 'confidence': 0.95, 'bbox': (100, 100, 50, 50)}
    ]
    
    img_path, meta_path = file_manager.save_image_with_metadata(test_image, test_detections)
    print(f"✓ Saved test image: {img_path}")
    print(f"✓ Saved metadata: {meta_path}")
    
    # Test ImageProcessor
    print("\n3. Testing ImageProcessor...")
    processor = ImageProcessor()
    
    enhanced = processor.apply_filters(test_image, "enhance")
    print(f"✓ Enhanced image shape: {enhanced.shape}")
    
    # Test color mask
    red_mask = processor.create_color_mask(test_image, 
                                          ColorPresets.RED['lower'], 
                                          ColorPresets.RED['upper'])
    print(f"✓ Created red color mask: {red_mask.shape}")
    
    # Test PerformanceMonitor
    print("\n4. Testing PerformanceMonitor...")
    monitor = PerformanceMonitor()
    
    monitor.start_timer("test_operation")
    time.sleep(0.1)  # Simulate processing
    elapsed = monitor.end_timer("test_operation")
    print(f"✓ Measured operation time: {elapsed:.3f}s")
    
    # Test ColorPresets
    print("\n5. Testing ColorPresets...")
    red_range = ColorPresets.get_color_range("red")
    if red_range:
        print(f"✓ Red color range: {red_range['lower']} to {red_range['upper']}")
    
    green_range = ColorPresets.get_color_range("green")
    if green_range:
        print(f"✓ Green color range: {green_range['lower']} to {green_range['upper']}")
    
    print("\nVision utils module testing completed!")
    
    # Cleanup test files
    try:
        os.remove("test_config.ini")
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        os.rmdir("./test_output")
        print("✓ Test files cleaned up")
    except Exception as e:
        print(f"Note: Some test files may remain: {e}")


if __name__ == "__main__":
    test_vision_utils()