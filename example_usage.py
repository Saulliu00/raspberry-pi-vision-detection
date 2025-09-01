#!/usr/bin/env python3
"""
Example usage script for the Raspberry Pi 5 Vision Detection System.
Demonstrates all the new features including auto-detection and full resolution.
"""

import time
import logging
from camera_module import RaspberryPiCamera
from object_detection import ObjectDetector
from vision_utils import ConfigManager, FileManager, PerformanceMonitor, ColorPresets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_auto_detection():
    """Example 1: Full auto-detection with best quality."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Auto-Detection with Full Resolution")
    print("="*60)
    
    # Initialize camera with full auto-detection
    camera = RaspberryPiCamera()  # Everything auto-detected
    
    print(f"✓ Auto-detected timezone: {camera.timezone}")
    print(f"✓ Auto-detected resolution: {camera.width}x{camera.height}")
    print(f"✓ Pictures directory: {camera.base_picture_dir}")
    
    # Capture at full resolution
    print("\nCapturing image at full resolution...")
    image = camera.capture_image_rpicam(
        save_to_pictures=True,
        prefix="auto_detection_full_res",
        use_full_resolution=True
    )
    
    if image is not None:
        height, width = image.shape[:2]
        print(f"✓ Captured: {width}x{height} pixels ({width*height:,} total pixels)")
        print(f"✓ File size: ~{(width*height*3)/1024/1024:.1f} MB (uncompressed)")
    
    return camera


def example_2_specific_timezone():
    """Example 2: Specific timezone with custom directory."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Timezone and Directory")
    print("="*60)
    
    # Initialize with specific settings
    camera = RaspberryPiCamera(
        timezone='US/Pacific',
        base_picture_dir='surveillance_photos'
    )
    
    print(f"✓ Using timezone: {camera.timezone}")
    print(f"✓ Pictures directory: {camera.base_picture_dir}")
    
    # Show current date folder
    info = camera.get_pictures_info()
    print(f"✓ Today's folder: {info['current_date_folder']}")
    
    return camera


def example_3_performance_comparison():
    """Example 3: Compare full resolution vs resized for detection."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Performance Comparison")
    print("="*60)
    
    camera = RaspberryPiCamera()
    detector = ObjectDetector()
    
    # Load contour detection (fastest method)
    print("Setting up contour detection for red objects...")
    
    # Capture one image at full resolution
    print("\n1. Capturing full resolution image...")
    start_time = time.time()
    full_res_image = camera.capture_image_rpicam(
        save_to_pictures=True,
        prefix="perf_test_full",
        use_full_resolution=True
    )
    capture_time_full = time.time() - start_time
    
    if full_res_image is not None:
        height, width = full_res_image.shape[:2]
        print(f"✓ Full resolution: {width}x{height} in {capture_time_full:.2f}s")
        
        # Test detection on full resolution
        start_time = time.time()
        detections_full = detector.detect_objects_contour(
            full_res_image, 
            ColorPresets.RED, 
            min_area=1000
        )
        detection_time_full = time.time() - start_time
        print(f"✓ Detection on full res: {len(detections_full)} objects in {detection_time_full:.2f}s")
        
        # Test detection on resized version
        print("\n2. Testing detection on resized version...")
        resized_image = camera.resize_image(full_res_image, (640, 480))
        start_time = time.time()
        detections_resized = detector.detect_objects_contour(
            resized_image, 
            ColorPresets.RED, 
            min_area=100  # Smaller min area for smaller image
        )
        detection_time_resized = time.time() - start_time
        print(f"✓ Detection on 640x480: {len(detections_resized)} objects in {detection_time_resized:.2f}s")
        
        # Performance summary
        print(f"\n3. Performance Summary:")
        print(f"   Full resolution detection: {detection_time_full:.2f}s")
        print(f"   Resized detection: {detection_time_resized:.2f}s")
        print(f"   Speed improvement: {detection_time_full/detection_time_resized:.1f}x faster")
        print(f"   Quality trade-off: Original saved at full resolution")
    
    return camera


def example_4_comprehensive_system():
    """Example 4: Complete system with all features."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Complete System Integration")
    print("="*60)
    
    # Initialize with custom config
    config = ConfigManager("example_config.ini")
    
    # Set up for full resolution capture but resized detection
    config.set('CAMERA', 'use_full_resolution', 'true')
    config.set('DETECTION', 'resize_for_detection', 'true')
    config.set('DETECTION', 'detection_width', '640')
    config.set('DETECTION', 'detection_height', '480')
    config.set('DETECTION', 'method', 'contour')
    config.set('DETECTION', 'target_color', 'red')
    
    print("✓ Configuration set for optimal quality + performance")
    
    # Initialize components
    camera = RaspberryPiCamera()
    detector = ObjectDetector()
    file_manager = FileManager('./example_output')
    performance_monitor = PerformanceMonitor()
    
    print(f"✓ System initialized with {camera.width}x{camera.height} resolution")
    
    # Perform complete detection cycle
    print("\nPerforming complete detection cycle...")
    
    performance_monitor.start_timer("total_process")
    
    # Capture full resolution
    performance_monitor.start_timer("capture")
    image = camera.capture_image_rpicam(
        save_to_pictures=True,
        prefix="complete_system",
        use_full_resolution=True
    )
    performance_monitor.end_timer("capture")
    
    if image is not None:
        # Resize for detection (but keep original for saving)
        performance_monitor.start_timer("resize")
        detection_image = camera.resize_image(image, (640, 480))
        performance_monitor.end_timer("resize")
        
        # Perform detection
        performance_monitor.start_timer("detection")
        detections = detector.detect_objects_contour(
            detection_image,
            ColorPresets.RED,
            min_area=500
        )
        performance_monitor.end_timer("detection")
        
        # Draw results on original full resolution image
        performance_monitor.start_timer("draw_results")
        if detections:
            # Scale detection coordinates back to full resolution
            scale_x = image.shape[1] / detection_image.shape[1]
            scale_y = image.shape[0] / detection_image.shape[0]
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                detection['bbox'] = (
                    int(x * scale_x), int(y * scale_y),
                    int(w * scale_x), int(h * scale_y)
                )
        
        result_image = detector.draw_detections(image, detections)
        performance_monitor.end_timer("draw_results")
        
        # Save results
        performance_monitor.start_timer("save")
        img_path, meta_path = file_manager.save_image_with_metadata(
            result_image, detections, "complete_detection"
        )
        performance_monitor.end_timer("save")
        
        total_time = performance_monitor.end_timer("total_process")
        
        # Show results
        height, width = image.shape[:2]
        print(f"\n✓ Results:")
        print(f"   Original image: {width}x{height} pixels")
        print(f"   Detection image: 640x480 pixels")
        print(f"   Objects found: {len(detections)}")
        print(f"   Total processing time: {total_time:.2f}s")
        print(f"   Saved to: {img_path}")
        
        # Performance breakdown
        performance_monitor.log_performance()
    
    return camera, config


def example_6_rotation_features():
    """Example 6: Image rotation features."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Image Rotation Features")
    print("="*60)
    
    camera = RaspberryPiCamera()
    
    # Capture one base image
    print("Capturing base image...")
    base_image = camera.capture_image_rpicam(
        save_to_pictures=False,  # Don't save the base image
        use_full_resolution=True
    )
    
    if base_image is not None:
        height, width = base_image.shape[:2]
        print(f"✓ Base image captured: {width}x{height}")
        
        # Test all rotation angles
        print(f"\nTesting all rotation angles:")
        rotation_angles = [0, 90, 180, 270]
        
        for angle in rotation_angles:
            print(f"\n   Testing {angle}° rotation...")
            
            # Apply preprocessing with rotation
            rotated_image = camera.preprocess_image(base_image, rotate_degrees=angle)
            rot_height, rot_width = rotated_image.shape[:2]
            
            print(f"   ✓ Rotated to: {rot_width}x{rot_height}")
            
            # Save rotated image
            save_success = camera.save_image(
                rotated_image, 
                prefix=f"rotation_{angle}deg", 
                use_pictures_folder=True
            )
            
            if save_success:
                print(f"   ✓ Saved {angle}° rotation")
                
            # Show dimension changes
            if angle in [90, 270]:
                if rot_width == height and rot_height == width:
                    print(f"   ✓ Dimensions correctly swapped for {angle}° rotation")
            elif angle == 180:
                if rot_width == width and rot_height == height:
                    print(f"   ✓ Dimensions preserved for {angle}° rotation")
        
        # Test custom angle (45 degrees)
        print(f"\n   Testing custom 45° rotation...")
        custom_rotated = camera.preprocess_image(base_image, rotate_degrees=45)
        custom_height, custom_width = custom_rotated.shape[:2]
        print(f"   ✓ 45° rotation result: {custom_width}x{custom_height}")
        
        if camera.save_image(custom_rotated, prefix="rotation_45deg_custom", use_pictures_folder=True):
            print(f"   ✓ Saved 45° custom rotation")
        
        # Performance comparison
        print(f"\n4. Performance comparison:")
        import time
        
        # Test speed of different rotations
        for angle in [0, 90, 180, 270]:
            start_time = time.time()
            for _ in range(5):  # Do it 5 times for better average
                rotated = camera.preprocess_image(base_image, rotate_degrees=angle)
            avg_time = (time.time() - start_time) / 5
            print(f"   {angle}° rotation: {avg_time:.3f}s average")
    
    return camera


def example_7_camera_mounting_orientations():
    """Example 7: Handle different camera mounting orientations."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Camera Mounting Orientations")
    print("="*60)
    
    mounting_scenarios = [
        {"name": "Normal mounting", "rotation": 0},
        {"name": "Upside down mounting", "rotation": 180},
        {"name": "Left side mounting", "rotation": 90},
        {"name": "Right side mounting", "rotation": 270}
    ]
    
    for scenario in mounting_scenarios:
        print(f"\n   Scenario: {scenario['name']}")
        print(f"   Recommended rotation: {scenario['rotation']}°")
        
        camera = RaspberryPiCamera()
        
        # Capture and process with appropriate rotation
        image = camera.capture_image_rpicam(
            save_to_pictures=False,
            use_full_resolution=True
        )
        
        if image is not None:
            corrected_image = camera.preprocess_image(
                image, 
                rotate_degrees=scenario['rotation']
            )
            
            # Save the corrected image
            save_success = camera.save_image(
                corrected_image,
                prefix=f"mounting_{scenario['name'].replace(' ', '_').lower()}",
                use_pictures_folder=True
            )
            
            if save_success:
                print(f"   ✓ Saved corrected image for {scenario['name']}")
    
    print(f"\n✓ All mounting orientations tested!")
    print(f"   Check your pictures folder to see the results")
    
    return camera
    """Example 5: Photo management and cleanup."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Photo Management")
    print("="*60)
    
    camera = RaspberryPiCamera()
    
    # Get photo collection info
    info = camera.get_pictures_info()
    print(f"✓ Photo collection overview:")
    print(f"   Base directory: {info['base_directory']}")
    print(f"   Total photos: {info['total_images']}")
    print(f"   Date folders: {len(info['existing_date_folders'])}")
    
    # Show breakdown by date
    for folder_info in info['existing_date_folders']:
        print(f"   - {folder_info['folder']}: {folder_info['image_count']} photos")
    
    # Take a few test photos
    print(f"\nTaking 3 test photos...")
    for i in range(3):
        image = camera.capture_image_rpicam(
            save_to_pictures=True,
            prefix=f"management_test_{i+1}",
            use_full_resolution=True
        )
        if image:
            height, width = image.shape[:2]
            print(f"   Photo {i+1}: {width}x{height}")
        time.sleep(1)  # Small delay between shots
    
    # Updated info
    updated_info = camera.get_pictures_info()
    print(f"\n✓ Updated collection: {updated_info['total_images']} total photos")
    
    # Cleanup demo (won't actually remove recent files)
    print(f"\nTesting cleanup (dry run - keeps files from last 365 days)...")
    removed = camera.cleanup_old_pictures(days_to_keep=365)
    print(f"✓ Files that would be removed if older: {removed}")
    
    return camera


def main():
    """Run all examples."""
    print("Raspberry Pi 5 Vision Detection System - Complete Examples")
    print("This script demonstrates all new features:")
    print("• Auto-detection of timezone and camera resolution")
    print("• Full resolution capture for best quality")
    print("• Performance optimization options")
    print("• Comprehensive photo management")
    
    try:
        # Run all examples
        example_1_auto_detection()
        example_2_specific_timezone()
        example_3_performance_comparison()
        example_4_comprehensive_system()
        example_5_photo_management()
        example_6_rotation_features()
        example_7_camera_mounting_orientations()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Benefits of the Updated System:")
        print("✓ Auto-detects system timezone and camera capabilities")
        print("✓ Uses full resolution by default for maximum quality")
        print("✓ Provides performance optimization options when needed")
        print("✓ Maintains structured photo organization")
        print("✓ Keeps resize functionality available but optional")
        print("✓ Comprehensive photo management tools")
        
        print(f"\nCheck your photos in:")
        print(f"• ./pictures/ (main photos)")
        print(f"• ./surveillance_photos/ (example 2)")
        print(f"• ./example_output/ (example 4)")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        logging.exception("Error in examples")


if __name__ == "__main__":
    main()