#!/usr/bin/env python3
"""
Example usage script for the Raspberry Pi 5 Vision Detection System.
Demonstrates parts defect detection with CSV data logging.
"""

import time
import logging
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import ConfigManager, FileManager, PerformanceMonitor, CSVDataLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_defect_detection():
    """Example 1: Basic parts defect detection with CSV logging."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Parts Defect Detection with CSV Logging")
    print("="*60)
    
    # Initialize components
    camera = RaspberryPiCamera()  # Auto-detect settings
    detector = ObjectDetector()
    csv_logger = CSVDataLogger("example_defect_log.csv")
    
    print(f"✓ Auto-detected timezone: {camera.timezone}")
    print(f"✓ Auto-detected resolution: {camera.width}x{camera.height}")
    print(f"✓ CSV logging initialized: example_defect_log.csv")
    
    # Load YOLO model for general object detection
    print("\nLoading YOLO model for defect detection...")
    if detector.load_yolo_model():
        print("✓ YOLO model loaded successfully")
        
        # Capture image
        print("\nCapturing part for inspection...")
        image = camera.capture_image_rpicam(
            save_to_pictures=True,
            prefix="parts_inspection",
            use_full_resolution=True
        )
        
        if image is not None:
            height, width = image.shape[:2]
            print(f"✓ Captured: {width}x{height} pixels")
            
            # Perform detection
            start_time = time.time()
            detections = detector.detect_objects_yolo(image, confidence_threshold=0.3)
            processing_time = time.time() - start_time
            
            print(f"✓ Detection completed: {len(detections)} objects found in {processing_time:.3f}s")
            
            # Log to CSV
            csv_logger.log_detection(
                capture_filename="parts_inspection_001.jpg",
                detections=detections,
                processing_time=processing_time,
                image_resolution=(width, height),
                detection_method="yolo"
            )
            print("✓ Results logged to CSV")
            
            # Show detection details
            if detections:
                print("\nDetected objects/defects:")
                for i, det in enumerate(detections, 1):
                    print(f"  {i}. {det['class']}: {det['confidence']:.2f} confidence")
    else:
        print("✗ YOLO model loading failed, trying Haar cascade...")
        # Fallback to Haar cascade
        if detector.load_haar_cascade("frontalface_default"):
            print("✓ Haar cascade loaded as fallback")
            
            image = camera.capture_image_rpicam(save_to_pictures=True, prefix="haar_test")
            if image:
                detections = detector.detect_objects_haar(image, "frontalface_default")
                csv_logger.log_detection(
                    capture_filename="haar_test_001.jpg", 
                    detections=detections,
                    processing_time=0.1,
                    image_resolution=image.shape[1::-1],
                    detection_method="haar"
                )
                print(f"✓ Haar detection: {len(detections)} objects, logged to CSV")
    
    return camera, detector, csv_logger


def example_2_continuous_inspection():
    """Example 2: Continuous parts inspection with performance monitoring."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Continuous Parts Inspection")
    print("="*60)
    
    # Initialize system
    camera = RaspberryPiCamera()
    detector = ObjectDetector()
    csv_logger = CSVDataLogger("continuous_inspection_log.csv")
    performance_monitor = PerformanceMonitor()
    
    print("✓ Continuous inspection system initialized")
    
    # Load detection method
    if detector.load_yolo_model():
        detection_method = "yolo"
        print("✓ Using YOLO for inspection")
    elif detector.load_haar_cascade("frontalface_default"):
        detection_method = "haar"
        print("✓ Using Haar cascade for inspection")
    else:
        print("✗ No detection method available")
        return
    
    print("\nStarting continuous inspection for 30 seconds...")
    print("Press Ctrl+C to stop early")
    
    inspection_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < 30:  # Run for 30 seconds
            inspection_count += 1
            print(f"\nInspection #{inspection_count}")
            
            # Time the entire process
            performance_monitor.start_timer("full_inspection")
            
            # Capture
            performance_monitor.start_timer("capture")
            image = camera.capture_image_rpicam(
                save_to_pictures=True,
                prefix=f"inspection_{inspection_count:03d}",
                use_full_resolution=True
            )
            capture_time = performance_monitor.end_timer("capture")
            
            if image:
                # Detect
                performance_monitor.start_timer("detection")
                if detection_method == "yolo":
                    detections = detector.detect_objects_yolo(image, confidence_threshold=0.4)
                else:
                    detections = detector.detect_objects_haar(image, "frontalface_default")
                detection_time = performance_monitor.end_timer("detection")
                
                # Log
                performance_monitor.start_timer("logging")
                total_time = performance_monitor.end_timer("full_inspection")
                
                csv_logger.log_detection(
                    capture_filename=f"inspection_{inspection_count:03d}.jpg",
                    detections=detections,
                    processing_time=total_time,
                    image_resolution=(image.shape[1], image.shape[0]),
                    detection_method=detection_method
                )
                performance_monitor.end_timer("logging")
                
                print(f"  Capture: {capture_time:.3f}s, Detection: {detection_time:.3f}s")
                print(f"  Found: {len(detections)} objects, Total: {total_time:.3f}s")
            
            # Wait before next inspection (simulate production line timing)
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nInspection stopped by user")
    
    print(f"\n✓ Completed {inspection_count} inspections")
    
    # Show performance summary
    print("\nPerformance Summary:")
    performance_monitor.log_performance()
    
    return csv_logger


def example_3_csv_analysis():
    """Example 3: Analyze CSV data from previous inspections."""
    print("\n" + "="*60)
    print("EXAMPLE 3: CSV Data Analysis")
    print("="*60)
    
    # Use the CSV logger from previous examples
    csv_files = ["example_defect_log.csv", "continuous_inspection_log.csv"]
    
    for csv_file in csv_files:
        try:
            csv_logger = CSVDataLogger(csv_file)
            print(f"\nAnalyzing {csv_file}:")
            
            # Get statistics
            stats = csv_logger.get_detection_statistics()
            print(f"  Total inspections: {stats.get('total_detections', 0)}")
            print(f"  Total objects/defects found: {stats.get('total_objects_found', 0)}")
            print(f"  Average processing time: {stats.get('average_processing_time', 0):.3f}s")
            print(f"  Most common detection: {stats.get('most_common_object', 'N/A')}")
            print(f"  Methods used: {', '.join(stats.get('detection_methods_used', []))}")
            
            # Show recent entries
            recent = csv_logger.get_recent_logs(5)
            if recent:
                print(f"  Last 5 entries:")
                for i, log in enumerate(recent, 1):
                    print(f"    {i}. {log.get('date')} {log.get('time')}: "
                          f"{log.get('detection_count')} objects ({log.get('detection_method')})")
        
        except Exception as e:
            print(f"  Could not analyze {csv_file}: {e}")


def example_4_comprehensive_system():
    """Example 4: Complete system with configuration and monitoring."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comprehensive Inspection System")
    print("="*60)
    
    # Initialize with custom configuration
    config = ConfigManager("comprehensive_config.ini")
    
    # Set up optimal configuration for parts inspection
    config.set('CAMERA', 'use_full_resolution', 'true')
    config.set('DETECTION', 'method', 'yolo')
    config.set('DETECTION', 'confidence_threshold', '0.4')
    config.set('DETECTION', 'resize_for_detection', 'true')
    config.set('CSV_LOGGING', 'enabled', 'true')
    config.set('CSV_LOGGING', 'csv_file', 'comprehensive_inspection.csv')
    
    print("✓ Configuration optimized for parts inspection")
    
    # Initialize all components
    camera = RaspberryPiCamera()
    detector = ObjectDetector()
    csv_logger = CSVDataLogger("comprehensive_inspection.csv")
    file_manager = FileManager('./comprehensive_output')
    performance_monitor = PerformanceMonitor()
    
    print(f"✓ System initialized with {camera.width}x{camera.height} resolution")
    
    # Load detection model
    if not detector.load_yolo_model():
        print("⚠ YOLO not available, using Haar cascade")
        detector.load_haar_cascade("frontalface_default")
    
    # Perform comprehensive inspection
    print("\nPerforming comprehensive inspection...")
    
    performance_monitor.start_timer("complete_process")
    
    # Capture at full resolution
    performance_monitor.start_timer("capture")
    image = camera.capture_image_rpicam(
        save_to_pictures=True,
        prefix="comprehensive_inspection",
        use_full_resolution=True
    )
    performance_monitor.end_timer("capture")
    
    if image is not None:
        # Apply preprocessing
        performance_monitor.start_timer("preprocessing")
        processed_image = camera.preprocess_image(image, rotate_degrees=180)
        performance_monitor.end_timer("preprocessing")
        
        # Resize for performance while keeping original quality
        performance_monitor.start_timer("resize")
        detection_image = camera.resize_image(processed_image, (640, 480))
        performance_monitor.end_timer("resize")
        
        # Perform detection on resized image
        performance_monitor.start_timer("detection")
        detections = detector.detect_objects_yolo(detection_image, confidence_threshold=0.4)
        performance_monitor.end_timer("detection")
        
        # Scale coordinates back to full resolution
        if detections:
            scale_x = processed_image.shape[1] / detection_image.shape[1]
            scale_y = processed_image.shape[0] / detection_image.shape[0]
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                detection['bbox'] = (
                    int(x * scale_x), int(y * scale_y),
                    int(w * scale_x), int(h * scale_y)
                )
        
        # Draw results on full resolution image
        performance_monitor.start_timer("draw_results")
        result_image = detector.draw_detections(processed_image, detections)
        performance_monitor.end_timer("draw_results")
        
        # Save comprehensive results
        performance_monitor.start_timer("save_results")
        img_path, meta_path = file_manager.save_image_with_metadata(
            result_image, detections, "comprehensive_inspection"
        )
        performance_monitor.end_timer("save_results")
        
        # Log to CSV
        total_time = performance_monitor.end_timer("complete_process")
        
        csv_logger.log_detection(
            capture_filename=os.path.basename(img_path),
            detections=detections,
            processing_time=total_time,
            image_resolution=(processed_image.shape[1], processed_image.shape[0]),
            detection_method="yolo"
        )
        
        # Show comprehensive results
        print(f"\n✓ Comprehensive inspection completed:")
        print(f"   Original resolution: {image.shape[1]}x{image.shape[0]}")
        print(f"   Detection resolution: 640x480 (performance optimized)")
        print(f"   Objects/defects found: {len(detections)}")
        print(f"   Total processing time: {total_time:.3f}s")
        print(f"   Results saved to: {img_path}")
        print(f"   Results logged to CSV")
        
        # Performance breakdown
        print(f"\n✓ Performance breakdown:")
        performance_monitor.log_performance()
    
    return config, csv_logger


def example_5_quality_monitoring():
    """Example 5: Quality monitoring and trend analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Quality Monitoring Dashboard")
    print("="*60)
    
    # Simulate a day's worth of inspections with varying defect rates
    csv_logger = CSVDataLogger("quality_monitoring.csv")
    camera = RaspberryPiCamera()
    detector = ObjectDetector()
    
    print("Simulating quality monitoring data...")
    
    # Load a detection method
    detection_available = detector.load_yolo_model() or detector.load_haar_cascade("frontalface_default")
    
    if detection_available:
        # Simulate several inspections with different outcomes
        inspection_scenarios = [
            {"defects": 0, "description": "Good part"},
            {"defects": 1, "description": "Minor defect detected"},
            {"defects": 0, "description": "Good part"},
            {"defects": 2, "description": "Multiple defects"},
            {"defects": 0, "description": "Good part"},
            {"defects": 1, "description": "Surface defect"},
            {"defects": 0, "description": "Good part"},
            {"defects": 3, "description": "Major defects - reject"},
            {"defects": 0, "description": "Good part"},
            {"defects": 1, "description": "Edge defect"}
        ]
        
        for i, scenario in enumerate(inspection_scenarios, 1):
            # Capture image for inspection
            image = camera.capture_image_rpicam(
                save_to_pictures=False,  # Don't clutter pictures folder
                use_full_resolution=True
            )
            
            if image:
                # Simulate detection results based on scenario
                simulated_detections = []
                for j in range(scenario["defects"]):
                    simulated_detections.append({
                        'class': f'defect_type_{j+1}',
                        'confidence': 0.85 + (j * 0.05),
                        'bbox': (100 + j*50, 100 + j*30, 40, 40),
                        'center': (120 + j*50, 120 + j*30)
                    })
                
                # Log the simulated inspection
                csv_logger.log_detection(
                    capture_filename=f"quality_inspection_{i:03d}.jpg",
                    detections=simulated_detections,
                    processing_time=0.2 + (i * 0.01),  # Slight variation in processing time
                    image_resolution=(image.shape[1], image.shape[0]),
                    detection_method="yolo"
                )
                
                print(f"  Inspection {i:2d}: {scenario['defects']} defects - {scenario['description']}")
                time.sleep(0.1)  # Small delay for realism
    
    # Analyze the quality data
    print(f"\n✓ Quality Analysis:")
    stats = csv_logger.get_detection_statistics()
    
    total_inspections = stats.get('total_detections', 0)
    total_defects = stats.get('total_objects_found', 0)
    defect_rate = (total_defects / total_inspections * 100) if total_inspections > 0 else 0
    
    print(f"   Total inspections: {total_inspections}")
    print(f"   Total defects found: {total_defects}")
    print(f"   Defect rate: {defect_rate:.1f}%")
    print(f"   Average processing time: {stats.get('average_processing_time', 0):.3f}s")
    
    # Show quality trend
    recent_logs = csv_logger.get_recent_logs(10)
    if recent_logs:
        print(f"\n   Recent quality trend (last 10 inspections):")
        for i, log in enumerate(reversed(recent_logs), 1):
            defect_count = int(log.get('detection_count', 0))
            status = "✓ PASS" if defect_count == 0 else f"✗ FAIL ({defect_count} defects)"
            print(f"     {i:2d}. {log.get('time', 'N/A')}: {status}")
    
    return csv_logger


def main():
    """Run all examples for parts defect detection."""
    print("Raspberry Pi 5 Vision Detection System - Parts Defect Detection Examples")
    print("This script demonstrates parts inspection with CSV data logging:")
    print("• YOLO and Haar cascade detection methods")
    print("• Automatic CSV data logging")
    print("• Full resolution capture for quality")
    print("• Performance optimization options")
    print("• Quality monitoring and analysis")
    
    try:
        # Run all examples
        camera, detector, csv_logger1 = example_1_basic_defect_detection()
        csv_logger2 = example_2_continuous_inspection()
        example_3_csv_analysis()
        config, csv_logger3 = example_4_comprehensive_system()
        csv_logger4 = example_5_quality_monitoring()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Benefits for Parts Defect Detection:")
        print("✓ Automated defect detection with YOLO and Haar cascades")
        print("✓ Comprehensive CSV data logging for quality tracking")
        print("✓ Full resolution capture for detailed inspection")
        print("✓ Performance optimization for production environments")
        print("✓ Quality monitoring and trend analysis capabilities")
        print("✓ Structured data storage for integration with QMS")
        
        print(f"\nGenerated CSV Files:")
        csv_files = [
            "example_defect_log.csv",
            "continuous_inspection_log.csv", 
            "comprehensive_inspection.csv",
            "quality_monitoring.csv"
        ]
        
        for csv_file in csv_files:
            try:
                import os
                if os.path.exists(csv_file):
                    print(f"• {csv_file} - Ready for analysis")
            except:
                pass
        
        print(f"\nCheck your files in:")
        print(f"• ./pictures/ (inspection photos)")
        print(f"• ./comprehensive_output/ (detailed results)")
        print(f"• ./*.csv (data logs for analysis)")
        
        print(f"\nFor production use:")
        print(f"• Integrate CSV data with your Quality Management System")
        print(f"• Set up automated alerts based on defect rates")
        print(f"• Use trend analysis for process improvement")
        print(f"• Consider training custom YOLO models for specific defects")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        logging.exception("Error in examples")


if __name__ == "__main__":
    main()