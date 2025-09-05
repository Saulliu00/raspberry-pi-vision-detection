#!/usr/bin/env python3
"""
Main vision detection system for Raspberry Pi 5.
Integrates camera, object detection, and utility modules with CSV data logging.
(Fixed version: cleans syntax, unifies RAW/POST naming, consistent save helpers.)
"""

import sys
import time
import logging
import argparse
import signal
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import cv2
from datetime import datetime

# Import our custom modules
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import ConfigManager, FileManager, PerformanceMonitor, CSVDataLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionSystem:
    """Main vision detection system class."""

    def __init__(self, config_file: str = "vision_config.ini"):
        """Initialize the vision system."""
        self.config = ConfigManager(config_file)
        self.file_manager = FileManager(self.config.get('OUTPUT', 'output_directory', './output'))
        self.performance_monitor = PerformanceMonitor()

        # Initialize CSV data logger
        csv_enabled = self.config.getboolean('CSV_LOGGING', 'enabled', True)
        if csv_enabled:
            csv_file = self.config.get('CSV_LOGGING', 'csv_file', 'detection_log.csv')
            self.csv_logger = CSVDataLogger(csv_file)
            logger.info(f"CSV logging enabled: {csv_file}")
        else:
            self.csv_logger = None
            logger.info("CSV logging disabled")

        # Initialize camera with auto-detection and full resolution support
        camera_width = self.config.get('CAMERA', 'width')
        camera_height = self.config.get('CAMERA', 'height')
        camera_timezone = self.config.get('CAMERA', 'timezone')
        pictures_dir = self.config.get('CAMERA', 'pictures_directory', 'pictures')
        use_full_resolution = self.config.getboolean('CAMERA', 'use_full_resolution', True)

        # Convert 'auto' or None values to actual None for auto-detection
        if camera_width and str(camera_width).lower() != 'auto':
            camera_width = int(camera_width)
        else:
            camera_width = None

        if camera_height and str(camera_height).lower() != 'auto':
            camera_height = int(camera_height)
        else:
            camera_height = None

        if camera_timezone and str(camera_timezone).lower() != 'auto':
            pass  # Use the specified timezone
        else:
            camera_timezone = None  # Auto-detect

        self.camera = RaspberryPiCamera(
            width=camera_width,
            height=camera_height,
            timezone=camera_timezone,
            base_picture_dir=pictures_dir
        )

        self.use_full_resolution = use_full_resolution

        # Initialize detector
        self.detector = ObjectDetector()
        self.setup_detector()

        self.running = False
        logger.info(
            f"Vision system initialized - Resolution: {self.camera.width}x{self.camera.height}, "
            f"Full res mode: {use_full_resolution}"
        )

    def setup_detector(self):
        """Setup the object detector based on configuration."""
        detection_method = self.config.get('DETECTION', 'method', 'yolo')

        if detection_method == 'yolo':
            success = self.detector.load_yolo_model()
            if not success:
                logger.warning("YOLO model failed to load, falling back to haar detection")
                self.config.set('DETECTION', 'method', 'haar')
                self.config.set('DETECTION', 'haar_cascade', 'frontalface_default')
                # Try to load haar cascade
                haar_success = self.detector.load_haar_cascade('frontalface_default')
                if not haar_success:
                    logger.error("Both YOLO and Haar cascade failed to load!")

        elif detection_method == 'haar':
            cascade_name = self.config.get('DETECTION', 'haar_cascade', 'frontalface_default')
            success = self.detector.load_haar_cascade(cascade_name)
            if success:
                logger.info(f"Loaded Haar cascade: {cascade_name}")
            else:
                logger.error(f"Failed to load Haar cascade: {cascade_name}")

        logger.info(f"Detector setup complete for method: {self.config.get('DETECTION', 'method', 'yolo')}")

    def capture_and_detect(self) -> Optional[Dict]:
        """Capture image and perform object detection with proper naming convention."""
        try:
            self.performance_monitor.start_timer("total_processing")

            # --- Capture RAW ---
            self.performance_monitor.start_timer("image_capture")
            camera_method = self.config.get('CAMERA', 'method', 'rpicam')
            camera_timeout = self.config.getint('CAMERA', 'timeout', 3000)
            save_pictures = self.config.getboolean('CAMERA', 'save_to_pictures', True)

            if camera_method == 'rpicam':
                raw_image = self.camera.capture_image_rpicam(
                    timeout=camera_timeout,
                    save_to_pictures=save_pictures,
                    prefix="raw",                  # RAW naming
                    use_full_resolution=self.use_full_resolution
                )
            else:
                raw_image = self.camera.capture_image_opencv(
                    save_to_pictures=save_pictures,
                    prefix="raw",                  # RAW naming
                    use_full_resolution=self.use_full_resolution
                )

            self.performance_monitor.end_timer("image_capture")

            if raw_image is None:
                logger.error("Failed to capture raw image")
                return None

            # Current capture index for consistent RAW/POST pairing
            current_index = getattr(self.camera, "_capture_index", None)

            # Most recent RAW filename (from pictures dir)
            raw_filename = self._get_most_recent_capture_filename()

            # --- Preprocess ---
            self.performance_monitor.start_timer("preprocessing")
            rotation_degrees = self.config.getint('CAMERA', 'rotation_degrees', 180)
            processed_image = self.camera.preprocess_image(raw_image, rotate_degrees=rotation_degrees)
            self.performance_monitor.end_timer("preprocessing")

            # --- Detect ---
            self.performance_monitor.start_timer("object_detection")
            detections = self.perform_detection(processed_image)
            self.performance_monitor.end_timer("object_detection")

            # --- Draw (optional) ---
            detection_result_image = processed_image
            if self.config.getboolean('OUTPUT', 'draw_detections', True):
                detection_result_image = self.detector.draw_detections(processed_image, detections)

            # --- Save POST image paired with RAW index ---
            if save_pictures:
                self.performance_monitor.start_timer("save_processed")
                success, post_filename = self.save_processed_image(
                    detection_result_image,
                    original_index=current_index,
                    prefix="post"
                )
                if not success:
                    post_filename = "post_save_failed.jpg"
                self.performance_monitor.end_timer("save_processed")
            else:
                post_filename = "post_not_saved.jpg"

            # --- Save to output (optional) ---
            if self.config.getboolean('OUTPUT', 'save_images', True):
                self.performance_monitor.start_timer("file_save")
                tag = f"detection_{current_index:06d}" if current_index is not None else "detection"
                img_path, meta_path = self.file_manager.save_image_with_metadata(
                    detection_result_image, detections, tag
                )
                self.performance_monitor.end_timer("file_save")
                logger.info(f"Detection results saved: {img_path}")

            total_time = self.performance_monitor.end_timer("total_processing")

            # --- CSV log (optional) ---
            if self.csv_logger:
                detection_method = self.config.get('DETECTION', 'method', 'unknown')
                self.csv_logger.log_detection(
                    capture_filename=raw_filename or "unknown",
                    detections=detections,
                    processing_time=total_time,
                    image_resolution=(processed_image.shape[1], processed_image.shape[0]),
                    detection_method=detection_method
                )

            # --- Results dict (what print_detection_summary expects) ---
            results = {
                'timestamp': time.time(),
                'detections': detections,
                'detection_count': len(detections),
                'processing_time': total_time,
                'image_shape': processed_image.shape,
                'capture_filename': raw_filename,
                'raw_filename': raw_filename,
                'post_filename': post_filename,
                'capture_index': current_index
            }

            logger.info(f"Detection completed: {len(detections)} objects found in {total_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error in capture_and_detect: {e}")
            return None

    def _get_most_recent_capture_filename(self) -> Optional[str]:
        """Get the filename of the most recently captured image."""
        try:
            pictures_info = self.camera.get_pictures_info()
            if pictures_info and pictures_info.get('existing_date_folders'):
                # Get the most recent date folder
                most_recent_folder = pictures_info['existing_date_folders'][-1]
                folder_path = most_recent_folder['path']

                # Get all image files in the folder
                image_files: List[Tuple[str, float]] = []
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(folder_path, filename)
                        try:
                            image_files.append((filename, os.path.getmtime(filepath)))
                        except OSError:
                            continue

                # Sort by modification time and return the most recent
                if image_files:
                    image_files.sort(key=lambda x: x[1], reverse=True)
                    return image_files[0][0]

            return None
        except Exception as e:
            logger.warning(f"Could not determine capture filename: {e}")
            return None

    def perform_detection(self, image: np.ndarray) -> List[Dict]:
        """Perform object detection based on configured method."""
        detection_method = self.config.get('DETECTION', 'method', 'yolo')

        # Resize image for detection if specified (perf) while preserving original for output
        detection_image = image
        resize_for_detection = self.config.getboolean('DETECTION', 'resize_for_detection', False)
        detection_width = self.config.getint('DETECTION', 'detection_width', 640)
        detection_height = self.config.getint('DETECTION', 'detection_height', 480)

        if resize_for_detection and (image.shape[1] > detection_width or image.shape[0] > detection_height):
            logger.info(
                f"Resizing image for detection: {image.shape[1]}x{image.shape[0]} -> "
                f"{detection_width}x{detection_height}"
            )
            detection_image = self.camera.resize_image(image, (detection_width, detection_height))

        if detection_method == 'yolo':
            confidence_threshold = float(self.config.get('DETECTION', 'confidence_threshold', '0.5'))
            nms_threshold = float(self.config.get('DETECTION', 'nms_threshold', '0.4'))
            detections = self.detector.detect_objects_yolo(
                detection_image, confidence_threshold, nms_threshold
            )

        elif detection_method == 'haar':
            cascade_name = self.config.get('DETECTION', 'haar_cascade', 'frontalface_default')
            detections = self.detector.detect_objects_haar(detection_image, cascade_name)

        else:
            logger.error(f"Unknown detection method: {detection_method}")
            detections = []

        # Scale detection coordinates back to original image size if we resized
        if resize_for_detection and detections:
            scale_x = image.shape[1] / detection_image.shape[1]
            scale_y = image.shape[0] / detection_image.shape[0]

            for detection in detections:
                x, y, w, h = detection['bbox']
                detection['bbox'] = (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                )
                if 'center' in detection and detection['center'] is not None:
                    detection['center'] = (
                        int(detection['center'][0] * scale_x),
                        int(detection['center'][1] * scale_y)
                    )

        return detections

    def run_continuous(self, interval: float = 1.0):
        """Run continuous detection loop."""
        self.running = True
        logger.info(f"Starting continuous detection (interval: {interval}s)")

        try:
            while self.running:
                start_time = time.time()

                results = self.capture_and_detect()
                if results:
                    self.print_detection_summary(results)

                # Wait for next iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def run_single(self):
        """Run single detection."""
        logger.info("Running single detection")
        results = self.capture_and_detect()

        if results:
            self.print_detection_summary(results)
            return results
        else:
            logger.error("Single detection failed")
            return None

    def save_processed_image(self, image: np.ndarray, original_index: int = None, prefix: str = "post") -> Tuple[bool, str]:
        """
        Save processed image for object detection with consistent naming.

        Args:
            image: Processed image to save
            original_index: Index from the original raw capture (to maintain pairing)
            prefix: Filename prefix ("post" for processed detection images)

        Returns:
            Tuple of (success, filename)
        """
        try:
            # Timezone-aware if camera provides tz; otherwise naive local time
            tz = getattr(self.camera, 'tz', None)
            now = datetime.now(tz) if tz else datetime.now()
            date_str = now.strftime("%Y%m%d")
            timestamp = now.strftime("%H%M%S_%f")[:-3]

            # Use same index as RAW if provided; otherwise try to get a next index
            if original_index is not None:
                index_val = original_index
            else:
                next_idx_fn = getattr(self.camera, 'get_next_capture_index', None)
                if callable(next_idx_fn):
                    index_val = next_idx_fn()
                else:
                    index_val = (getattr(self.camera, '_capture_index', 0) or 0) + 1

            filename = f"{prefix}_{index_val:06d}_{date_str}_{timestamp}.jpg"

            # Get today's date folder from camera helper
            if hasattr(self.camera, '_get_date_folder_path'):
                date_folder = self.camera._get_date_folder_path()
            else:
                # Fallback: use pictures_directory root
                date_folder = self.config.get('CAMERA', 'pictures_directory', 'pictures')
                os.makedirs(date_folder, exist_ok=True)

            save_path = os.path.join(date_folder, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Convert RGB back to BGR for saving with OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            success = cv2.imwrite(save_path, image_bgr)
            if success:
                logger.info(f"POST image saved as: {filename}")
                return True, filename
            else:
                logger.error(f"Failed to save POST image: {filename}")
                return False, filename
        except Exception as e:
            logger.error(f"POST image save error: {e}")
            return False, "save_error.jpg"

    def print_detection_summary(self, results: Dict):
        """Print summary of detection results with new naming convention."""
        print(f"\n--- Detection Summary ---")
        print(f"Timestamp: {time.ctime(results['timestamp'])}")
        print(f"Capture Index: {results.get('capture_index', 'N/A')}")
        print(f"RAW image: {results.get('raw_filename', 'N/A')}")
        print(f"POST image: {results.get('post_filename', 'N/A')}")
        print(f"Objects detected: {results['detection_count']}")
        print(f"Processing time: {results['processing_time']:.3f}s")
        print(f"Image size: {results['image_shape'][1]}x{results['image_shape'][0]} pixels")
        print(f"Full resolution mode: {'Enabled' if getattr(self, 'use_full_resolution', False) else 'Disabled'}")

        if results['detections']:
            print("\nDetected objects:")
            for i, detection in enumerate(results['detections'], 1):
                bbox = detection['bbox']
                conf = detection.get('confidence', detection.get('score', 0.0))
                cls_name = detection.get('class', detection.get('label', 'unknown'))
                print(
                    f"  {i}. {cls_name}: confidence={conf:.2f}, "
                    f"bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                )

        # Show CSV logging status
        if self.csv_logger:
            print("✓ Results logged to CSV")
        else:
            print("⚠ CSV logging disabled")

        print("------------------------\n")

    def get_csv_statistics(self):
        """Get and display CSV statistics."""
        if not self.csv_logger:
            print("CSV logging is not enabled")
            return

        stats = self.csv_logger.get_detection_statistics()
        print("\n--- CSV Log Statistics ---")
        print(f"Total detections: {stats.get('total_detections', 0)}")
        print(f"Total objects found: {stats.get('total_objects_found', 0)}")
        print(f"Average processing time: {stats.get('average_processing_time', 0):.3f}s")
        print(f"Average confidence score: {stats.get('average_confidence_score', 0):.3f}")
        print(f"Most common object: {stats.get('most_common_object', 'N/A')}")
        methods = stats.get('detection_methods_used', [])
        print(f"Detection methods used: {', '.join(methods)}")

        date_range = stats.get('date_range', {})
        if date_range.get('start') and date_range.get('end'):
            print(f"Date range: {date_range['start']} to {date_range['end']}")

        print("-------------------------\n")

    def show_recent_logs(self, limit: int = 10):
        """Show recent CSV log entries."""
        if not self.csv_logger:
            print("CSV logging is not enabled")
            return

        recent_logs = self.csv_logger.get_recent_logs(limit)
        print(f"\n--- Recent {len(recent_logs)} Log Entries ---")

        for i, log in enumerate(recent_logs, 1):
            print(f"{i}. {log.get('date', 'N/A')} {log.get('time', 'N/A')}")
            print(f"   File: {log.get('capture_filename', 'N/A')}")
            print(f"   Objects: {log.get('detection_count', 0)} ({log.get('detected_objects', 'None')})")
            print(f"   Method: {log.get('detection_method', 'N/A')} ({log.get('processing_time', 'N/A')}s)")
            print()

        print("---------------------------\n")

    def stop(self):
        """Stop the vision system."""
        self.running = False
        self.camera.cleanup()
        self.performance_monitor.log_performance()

        # Cleanup old CSV logs if configured
        if self.csv_logger:
            cleanup_days = self.config.getint('CSV_LOGGING', 'cleanup_days', 30)
            if cleanup_days > 0:
                self.csv_logger.cleanup_old_logs(cleanup_days)

        logger.info("Vision system stopped")

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'running': self.running,
            'config_method': self.config.get('DETECTION', 'method'),
            'camera_resolution': (self.camera.width, self.camera.height),
            'csv_logging': self.csv_logger is not None,
            'performance_stats': self.performance_monitor.get_performance_report()
        }


def signal_handler(signum, frame):
    """Handle system signals gracefully."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Vision Detection System with CSV Logging")

    parser.add_argument('--config', '-c',
                       default='vision_config.ini',
                       help='Configuration file path')

    parser.add_argument('--mode', '-m',
                       choices=['single', 'continuous', 'stats', 'logs'],
                       default='single',
                       help='Detection mode: single detection, continuous, show stats, or show recent logs')

    parser.add_argument('--interval', '-i',
                       type=float,
                       default=1.0,
                       help='Detection interval for continuous mode (seconds)')

    parser.add_argument('--method',
                       choices=['yolo', 'haar'],
                       help='Override detection method')

    parser.add_argument('--resolution',
                       choices=['auto', 'full', '1080p', '720p', '480p'],
                       help='Override camera resolution')

    parser.add_argument('--rotation',
                       type=int,
                       choices=[0, 90, 180, 270],
                       help='Rotate captured images (0, 90, 180, 270 degrees)')

    parser.add_argument('--resize-detection',
                       action='store_true',
                       help='Resize images for detection performance')

    parser.add_argument('--disable-csv',
                       action='store_true',
                       help='Disable CSV logging for this session')

    parser.add_argument('--csv-file',
                       help='Override CSV log file path')

    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize vision system
        vision_system = VisionSystem(args.config)

        # Override CSV settings if specified
        if args.disable_csv:
            vision_system.config.set('CSV_LOGGING', 'enabled', 'false')
            vision_system.csv_logger = None
            logger.info("CSV logging disabled for this session")

        if args.csv_file:
            vision_system.config.set('CSV_LOGGING', 'csv_file', args.csv_file)
            if not args.disable_csv:
                vision_system.csv_logger = CSVDataLogger(args.csv_file)
                logger.info(f"Using custom CSV file: {args.csv_file}")

        # Override config with command line arguments
        if args.method:
            vision_system.config.set('DETECTION', 'method', args.method)
            vision_system.setup_detector()  # Reinitialize detector

        if args.resolution:
            if args.resolution in ('auto', 'full'):
                vision_system.config.set('CAMERA', 'width', 'auto')
                vision_system.config.set('CAMERA', 'height', 'auto')
                vision_system.config.set('CAMERA', 'use_full_resolution', 'true')
            elif args.resolution == '1080p':
                vision_system.config.set('CAMERA', 'width', '1920')
                vision_system.config.set('CAMERA', 'height', '1080')
                vision_system.config.set('CAMERA', 'use_full_resolution', 'false')
            elif args.resolution == '720p':
                vision_system.config.set('CAMERA', 'width', '1280')
                vision_system.config.set('CAMERA', 'height', '720')
                vision_system.config.set('CAMERA', 'use_full_resolution', 'false')
            elif args.resolution == '480p':
                vision_system.config.set('CAMERA', 'width', '640')
                vision_system.config.set('CAMERA', 'height', '480')
                vision_system.config.set('CAMERA', 'use_full_resolution', 'false')

        if args.resize_detection:
            vision_system.config.set('DETECTION', 'resize_for_detection', 'true')

        if args.rotation is not None:
            vision_system.config.set('CAMERA', 'rotation_degrees', str(args.rotation))

        # Run based on mode
        if args.mode == 'stats':
            vision_system.get_csv_statistics()
        elif args.mode == 'logs':
            vision_system.show_recent_logs(20)  # Show last 20 entries
        elif args.mode == 'continuous':
            vision_system.run_continuous(args.interval)
        else:
            results = vision_system.run_single()
            if results is None:
                sys.exit(1)

    except Exception as e:
        logger.error(f"Vision system error: {e}")
        sys.exit(1)

    finally:
        if 'vision_system' in locals():
            vision_system.stop()


if __name__ == "__main__":
    main()