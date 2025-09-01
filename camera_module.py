#!/usr/bin/env python3
"""
Camera module for Raspberry Pi 5 vision detection system.
Handles camera initialization, image capture, and basic preprocessing.
"""

import cv2
import numpy as np
import subprocess
import os
import time
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaspberryPiCamera:
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize the Raspberry Pi camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        self.temp_file = "/tmp/rpi_capture.jpg"
        
    def capture_image_rpicam(self, timeout: int = 2000, output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Capture image using rpicam-still command.
        
        Args:
            timeout: Capture timeout in milliseconds
            output_path: Optional path to save image, uses temp file if None
            
        Returns:
            numpy array of the captured image or None if failed
        """
        try:
            save_path = output_path or self.temp_file
            
            # Use rpicam-still to capture image
            cmd = [
                "rpicam-still",
                "-t", str(timeout),
                "-o", save_path,
                "--width", str(self.width),
                "--height", str(self.height),
                "--nopreview"  # Disable preview for headless operation
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(f"rpicam-still failed: {result.stderr}")
                return None
                
            # Load the captured image
            if os.path.exists(save_path):
                image = cv2.imread(save_path)
                if image is not None:
                    logger.info(f"Successfully captured image: {image.shape}")
                    return image
                else:
                    logger.error("Failed to load captured image")
                    return None
            else:
                logger.error(f"Captured image file not found: {save_path}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Camera capture timeout")
            return None
        except Exception as e:
            logger.error(f"Camera capture error: {e}")
            return None
    
    def capture_image_opencv(self, camera_index: int = 0) -> Optional[np.ndarray]:
        """
        Capture image using OpenCV (alternative method).
        
        Args:
            camera_index: Camera device index
            
        Returns:
            numpy array of the captured image or None if failed
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not cap.isOpened():
                logger.error("Failed to open camera with OpenCV")
                return None
                
            # Warm up camera
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    continue
                time.sleep(0.1)
            
            # Capture final image
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                logger.info(f"Successfully captured image with OpenCV: {frame.shape}")
                return frame
            else:
                logger.error("Failed to capture image with OpenCV")
                return None
                
        except Exception as e:
            logger.error(f"OpenCV capture error: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing to the image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to RGB (OpenCV uses BGR by default)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Apply basic enhancements
            # Adjust brightness and contrast slightly
            alpha = 1.1  # Contrast control (1.0-3.0)
            beta = 10    # Brightness control (0-100)
            enhanced = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
            
            # Apply slight Gaussian blur to reduce noise
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            logger.info("Image preprocessing completed")
            return denoised
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image
        """
        try:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            logger.info(f"Image resized to {target_size}")
            return resized
        except Exception as e:
            logger.error(f"Image resize error: {e}")
            return image
    
    def save_image(self, image: np.ndarray, path: str) -> bool:
        """
        Save image to specified path.
        
        Args:
            image: Image to save
            path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert RGB back to BGR for saving with OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
                
            success = cv2.imwrite(path, image_bgr)
            if success:
                logger.info(f"Image saved to {path}")
                return True
            else:
                logger.error(f"Failed to save image to {path}")
                return False
        except Exception as e:
            logger.error(f"Image save error: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def test_camera_module():
    """Test function to verify camera functionality."""
    print("Testing Raspberry Pi Camera Module...")
    
    camera = RaspberryPiCamera(width=640, height=480)
    
    # Test rpicam-still capture
    print("\n1. Testing rpicam-still capture...")
    image = camera.capture_image_rpicam(timeout=3000)
    if image is not None:
        print(f"✓ rpicam-still capture successful: {image.shape}")
        
        # Test preprocessing
        processed = camera.preprocess_image(image)
        print(f"✓ Image preprocessing successful: {processed.shape}")
        
        # Test save
        if camera.save_image(processed, "test_rpicam.jpg"):
            print("✓ Image save successful: test_rpicam.jpg")
        
    else:
        print("✗ rpicam-still capture failed")
    
    # Test OpenCV capture (fallback)
    print("\n2. Testing OpenCV capture...")
    image_cv = camera.capture_image_opencv()
    if image_cv is not None:
        print(f"✓ OpenCV capture successful: {image_cv.shape}")
        camera.save_image(image_cv, "test_opencv.jpg")
    else:
        print("✗ OpenCV capture failed")
    
    # Cleanup
    camera.cleanup()
    print("\nCamera module testing completed!")


if __name__ == "__main__":
    test_camera_module()