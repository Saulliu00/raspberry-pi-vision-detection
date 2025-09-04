#!/usr/bin/env python3
"""
Utility functions for the Raspberry Pi 5 vision detection system.
Contains helper functions for file management, configuration, CSV data logging, and image processing.
"""

import os
import json
import time
import logging
import cv2
import numpy as np
import csv
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import configparser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVDataLogger:
    """Handles CSV data logging for detection results."""
    
    def __init__(self, csv_file: str = "detection_log.csv"):
        """
        Initialize CSV data logger.
        
        Args:
            csv_file: Path to CSV log file
        """
        self.csv_file = csv_file
        self.fieldnames = [
            'date',
            'time', 
            'capture_filename',
            'detection_count',
            'detected_objects',
            'processing_time',
            'image_resolution',
            'detection_method'
        ]
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        try:
            if not os.path.exists(self.csv_file):
                with open(self.csv_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
                logger.info(f"Created new CSV log file: {self.csv_file}")
            else:
                logger.info(f"Using existing CSV log file: {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")
    
    def log_detection(self, 
                     capture_filename: str,
                     detections: List[Dict],
                     processing_time: float,
                     image_resolution: Tuple[int, int],
                     detection_method: str = "unknown"):
        """
        Log detection results to CSV.
        
        Args:
            capture_filename: Name of the captured image file
            detections: List of detection results
            processing_time: Time taken for processing in seconds
            image_resolution: Image resolution as (width, height)
            detection_method: Detection method used (yolo, haar, etc.)
        """
        try:
            now = datetime.now()
            
            # Format detected objects as a readable string
            if detections:
                objects_summary = "; ".join([
                    f"{det['class']}({det['confidence']:.2f})" 
                    for det in detections
                ])
            else:
                objects_summary = "No objects detected"
            
            # Prepare row data
            row_data = {
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'capture_filename': capture_filename,
                'detection_count': len(detections),
                'detected_objects': objects_summary,
                'processing_time': f"{processing_time:.3f}",
                'image_resolution': f"{image_resolution[0]}x{image_resolution[1]}",
                'detection_method': detection_method
            }
            
            # Write to CSV
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(row_data)
            
            logger.info(f"Logged detection to CSV: {len(detections)} objects detected")
            
        except Exception as e:
            logger.error(f"Failed to log to CSV: {e}")
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict]:
        """
        Get recent detection logs from CSV.
        
        Args:
            limit: Maximum number of recent logs to return
            
        Returns:
            List of recent log entries
        """
        try:
            recent_logs = []
            
            if os.path.exists(self.csv_file):
                with open(self.csv_file, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    all_rows = list(reader)
                    
                    # Return the most recent entries (assuming CSV is chronological)
                    recent_logs = all_rows[-limit:] if len(all_rows) > limit else all_rows
                    recent_logs.reverse()  # Most recent first
            
            return recent_logs
            
        except Exception as e:
            logger.error(f"Failed to read recent logs: {e}")
            return []
    
    def get_detection_statistics(self) -> Dict:
        """
        Get statistics from the detection log.
        
        Returns:
            Dictionary with detection statistics
        """
        try:
            stats = {
                'total_detections': 0,
                'total_objects_found': 0,
                'average_processing_time': 0,
                'most_common_object': 'N/A',
                'detection_methods_used': set(),
                'date_range': {'start': None, 'end': None}
            }
            
            if not os.path.exists(self.csv_file):
                return stats
            
            processing_times = []
            object_counts = {}
            dates = []
            
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    stats['total_detections'] += 1
                    stats['total_objects_found'] += int(row.get('detection_count', 0))
                    stats['detection_methods_used'].add(row.get('detection_method', 'unknown'))
                    
                    # Processing times
                    try:
                        proc_time = float(row.get('processing_time', 0))
                        processing_times.append(proc_time)
                    except:
                        pass
                    
                    # Object counting
                    objects_str = row.get('detected_objects', '')
                    if objects_str and objects_str != "No objects detected":
                        for obj_info in objects_str.split(';'):
                            obj_name = obj_info.strip().split('(')[0]
                            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                    
                    # Date tracking
                    date_str = row.get('date', '')
                    if date_str:
                        dates.append(date_str)
            
            # Calculate averages and most common
            if processing_times:
                stats['average_processing_time'] = sum(processing_times) / len(processing_times)
            
            if object_counts:
                stats['most_common_object'] = max(object_counts, key=object_counts.get)
            
            if dates:
                stats['date_range']['start'] = min(dates)
                stats['date_range']['end'] = max(dates)
            
            stats['detection_methods_used'] = list(stats['detection_methods_used'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get detection statistics: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up old log entries, keeping only recent days.
        
        Args:
            days_to_keep: Number of days of logs to keep
        """
        try:
            if not os.path.exists(self.csv_file):
                return
            
            cutoff_date = datetime.now().date()
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
            
            temp_file = self.csv_file + '.tmp'
            rows_kept = 0
            
            with open(self.csv_file, 'r', newline='') as infile, \
                 open(temp_file, 'w', newline='') as outfile:
                
                reader = csv.DictReader(infile)
                writer = csv.DictWriter(outfile, fieldnames=self.fieldnames)
                writer.writeheader()
                
                for row in reader:
                    try:
                        row_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                        if row_date >= cutoff_date:
                            writer.writerow(row)
                            rows_kept += 1
                    except:
                        # Keep rows with invalid dates
                        writer.writerow(row)
                        rows_kept += 1
            
            # Replace original file with cleaned version
            os.replace(temp_file, self.csv_file)
            logger.info(f"CSV cleanup completed: {rows_kept} rows kept")
            
        except Exception as e:
            logger.error(f"Failed to cleanup CSV logs: {e}")


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
            'method': 'yolo',  # 'yolo' or 'haar' (removed contour)
            'confidence_threshold': '0.5',
            'nms_threshold': '0.4',
            'resize_for_detection': 'false',  # Resize image for detection performance
            'detection_width': '640',  # Width to resize to for detection (if enabled)
            'detection_height': '480',  # Height to resize to for detection (if enabled)
            'haar_cascade': 'frontalface_default'  # Default Haar cascade to use
        }
        
        self.config['OUTPUT'] = {
            'save_images': 'true',
            'output_directory': './output',
            'draw_detections': 'true',
            'save_logs': 'true'
        }
        
        self.config['CSV_LOGGING'] = {
            'enabled': 'true',
            'csv_file': 'detection_log.csv',
            'cleanup_days': '30'  # Days of logs to keep
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


def test_vision_utils():
    """Test function to verify utility functions."""
    print("Testing Vision Utils Module...")
    
    # Test ConfigManager
    print("\n1. Testing ConfigManager...")
    config = ConfigManager("test_config.ini")
    print(f"✓ Camera width: {config.get('CAMERA', 'width')}")
    print(f"✓ Detection method: {config.get('DETECTION', 'method')}")
    print(f"✓ Save images: {config.getboolean('OUTPUT', 'save_images')}")
    print(f"✓ CSV logging enabled: {config.getboolean('CSV_LOGGING', 'enabled')}")
    
    # Test CSVDataLogger
    print("\n2. Testing CSV Data Logger...")
    csv_logger = CSVDataLogger("test_detection_log.csv")
    
    # Test logging some sample detections
    sample_detections = [
        {'class': 'defect', 'confidence': 0.95, 'bbox': (100, 100, 50, 50)},
        {'class': 'part', 'confidence': 0.87, 'bbox': (200, 150, 60, 40)}
    ]
    
    csv_logger.log_detection(
        capture_filename="test_image_001.jpg",
        detections=sample_detections,
        processing_time=0.245,
        image_resolution=(1920, 1080),
        detection_method="yolo"
    )
    
    # Log another entry with no detections
    csv_logger.log_detection(
        capture_filename="test_image_002.jpg", 
        detections=[],
        processing_time=0.123,
        image_resolution=(1920, 1080),
        detection_method="haar"
    )
    
    print("✓ Sample detections logged to CSV")
    
    # Test getting recent logs
    recent = csv_logger.get_recent_logs(5)
    print(f"✓ Retrieved {len(recent)} recent log entries")
    
    # Test statistics
    stats = csv_logger.get_detection_statistics()
    print(f"✓ Total detections logged: {stats.get('total_detections', 0)}")
    print(f"✓ Total objects found: {stats.get('total_objects_found', 0)}")
    print(f"✓ Average processing time: {stats.get('average_processing_time', 0):.3f}s")
    
    # Test FileManager
    print("\n3. Testing FileManager...")
    file_manager = FileManager("./test_output")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_detections = [
        {'class': 'test_defect', 'confidence': 0.95, 'bbox': (100, 100, 50, 50)}
    ]
    
    img_path, meta_path = file_manager.save_image_with_metadata(test_image, test_detections)
    print(f"✓ Saved test image: {img_path}")
    print(f"✓ Saved metadata: {meta_path}")
    
    # Test ImageProcessor
    print("\n4. Testing ImageProcessor...")
    processor = ImageProcessor()
    
    enhanced = processor.apply_filters(test_image, "enhance")
    print(f"✓ Enhanced image shape: {enhanced.shape}")
    
    # Test PerformanceMonitor
    print("\n5. Testing PerformanceMonitor...")
    monitor = PerformanceMonitor()
    
    monitor.start_timer("test_operation")
    time.sleep(0.1)  # Simulate processing
    elapsed = monitor.end_timer("test_operation")
    print(f"✓ Measured operation time: {elapsed:.3f}s")
    
    print("\nVision utils module testing completed!")
    
    # Cleanup test files
    try:
        os.remove("test_config.ini")
        os.remove("test_detection_log.csv")
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