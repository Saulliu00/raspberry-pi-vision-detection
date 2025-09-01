#!/usr/bin/env python3
"""
Main vision detection system for Raspberry Pi 5.
Integrates camera, object detection, and utility modules.
"""

import sys
import time
import logging
import argparse
import signal
from typing import Dict, List, Optional
import numpy as np

# Import our custom modules
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import ConfigManager, FileManager, PerformanceMonitor, ColorPresets

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
        
        # Initialize camera
        camera_width = self.config.getint('CAMERA', 'width', 640)
        camera_height = self.config.getint('CAMERA', 'height', 480)
        self.camera = RaspberryPiCamera(width=camera_width, height=camera_height)
        
        # Initialize detector
        self.detector = ObjectDetector()
        self.setup_detector()
        
        self.running = False
        logger.info("Vision system initialized")
    
    def setup_detector(self):
        """Setup the object detector based on configuration."""
        detection_method = self.config.get('DETECTION', 'method', 'contour')
        
        if detection_method == 'yolo':
            success = self.detector.load_yolo_model()
            if not success:
                logger.warning("YOLO model failed to load, falling back to contour detection")
                self.config.set('DETECTION', 'method', 'contour')
        
        elif detection_method == 'haar':
            # Load common Haar cascades
            cascades_to_load = ['frontalface_default', 'eye', 'smile']
            for cascade_name in cascades_to_load:
                success = self.detector.load_haar_cascade(cascade_name)
                if success:
                    logger.info(f"Loaded Haar cascade: {cascade_name}")
        
        logger.info(f"Detector setup complete for method: {detection_method}")
    
    def capture_and_detect(self) -> Optional[Dict]:
        """Capture image and perform object detection."""
        try:
            self.performance_monitor.start_timer("total_processing")
            
            # Capture image
            self.performance_monitor.start_timer("image_capture")
            camera_method = self.config.get('CAMERA', 'method', 'rpicam')
            camera_timeout = self.config.getint('CAMERA', 'timeout', 3000)
            
            if camera_method == 'rpicam':
                image = self.camera.capture_image_rpicam(timeout=camera_timeout)
            else:
                image = self.camera.capture_image_opencv()
            
            self.performance_monitor.end_timer("image_capture")
            
            if image is None:
                logger.error("Failed to capture image")
                return None
            
            # Preprocess image
            self.performance_monitor.start_timer("preprocessing")
            processed_image = self.camera.preprocess_image(image)
            self.performance_monitor.end_timer("preprocessing")
            
            # Perform detection
            self.performance_monitor.start_timer("object_detection")
            detections = self.perform_detection(processed_image)
            self.performance_monitor.end_timer("object_detection")
            
            # Draw detections if enabled
            result_image = processed_image
            if self.config.getboolean('OUTPUT', 'draw_detections', True):
                result_image = self.detector.draw_detections(processed_image, detections)
            
            # Save results if enabled
            if self.config.getboolean('OUTPUT', 'save_images', True):
                self.performance_monitor.start_timer("file_save")
                img_path, meta_path = self.file_manager.save_image_with_metadata(
                    result_image, detections, "detection"
                )
                self.performance_monitor.end_timer("file_save")
                logger.info(f"Results saved: {img_path}")
            
            total_time = self.performance_monitor.end_timer("total_processing")
            
            # Prepare results
            results = {
                'timestamp': time.time(),
                'detections': detections,
                'detection_count': len(detections),
                'processing_time': total_time,
                'image_shape': processed_image.shape
            }
            
            logger.info(f"Detection completed: {len(detections)} objects found in {total_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in capture_and_detect: {e}")
            return None
    
    def perform_detection(self, image: np.ndarray) -> List[Dict]:
        """Perform object detection based on configured method."""
        detection_method = self.config.get('DETECTION', 'method', 'contour')
        
        if detection_method == 'yolo':
            confidence_threshold = float(self.config.get('DETECTION', 'confidence_threshold', '0.5'))
            nms_threshold = float(self.config.get('DETECTION', 'nms_threshold', '0.4'))
            return self.detector.detect_objects_yolo(
                image, confidence_threshold, nms_threshold
            )
        
        elif detection_method == 'haar':
            cascade_name = self.config.get('DETECTION', 'haar_cascade', 'frontalface_default')
            return self.detector.detect_objects_haar(image, cascade_name)
        
        else:  # contour detection
            # Get color range from config or use default red
            color_name = self.config.get('DETECTION', 'target_color', 'red')
            color_range = ColorPresets.get_color_range(color_name)
            
            if color_range is None:
                # Use custom color range from config
                lower_str = self.config.get('COLORS', f'{color_name}_lower', '0,50,50')
                upper_str = self.config.get('COLORS', f'{color_name}_upper', '10,255,255')
                
                lower = np.array([int(x) for x in lower_str.split(',')])
                upper = np.array([int(x) for x in upper_str.split(',')])
                color_range = {'lower': lower, 'upper': upper}
            
            min_area = self.config.getint('DETECTION', 'min_area', 500)
            return self.detector.detect_objects_contour(image, color_range, min_area)
    
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
    
    def print_detection_summary(self, results: Dict):
        """Print summary of detection results."""
        print(f"\n--- Detection Summary ---")
        print(f"Timestamp: {time.ctime(results['timestamp'])}")
        print(f"Objects detected: {results['detection_count']}")
        print(f"Processing time: {results['processing_time']:.3f}s")
        print(f"Image size: {results['image_shape']}")
        
        if results['detections']:
            print("\nDetected objects:")
            for i, detection in enumerate(results['detections'], 1):
                bbox = detection['bbox']
                print(f"  {i}. {detection['class']}: "
                      f"confidence={detection['confidence']:.2f}, "
                      f"bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
        print("------------------------\n")
    
    def stop(self):
        """Stop the vision system."""
        self.running = False
        self.camera.cleanup()
        self.performance_monitor.log_performance()
        logger.info("Vision system stopped")
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'running': self.running,
            'config_method': self.config.get('DETECTION', 'method'),
            'camera_resolution': (
                self.config.getint('CAMERA', 'width'),
                self.config.getint('CAMERA', 'height')
            ),
            'performance_stats': self.performance_monitor.get_performance_report()
        }


def signal_handler(signum, frame):
    """Handle system signals gracefully."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Vision Detection System")
    
    parser.add_argument('--config', '-c', 
                       default='vision_config.ini',
                       help='Configuration file path')
    
    parser.add_argument('--mode', '-m',
                       choices=['single', 'continuous'],
                       default='single',
                       help='Detection mode')
    
    parser.add_argument('--interval', '-i',
                       type=float,
                       default=1.0,
                       help='Detection interval for continuous mode (seconds)')
    
    parser.add_argument('--method', 
                       choices=['yolo', 'haar', 'contour'],
                       help='Override detection method')
    
    parser.add_argument('--color',
                       choices=['red', 'green', 'blue', 'yellow', 'orange'],
                       help='Target color for contour detection')
    
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
        
        # Override config with command line arguments
        if args.method:
            vision_system.config.set('DETECTION', 'method', args.method)
            vision_system.setup_detector()  # Reinitialize detector
        
        if args.color:
            vision_system.config.set('DETECTION', 'target_color', args.color)
        
        # Run detection
        if args.mode == 'continuous':
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