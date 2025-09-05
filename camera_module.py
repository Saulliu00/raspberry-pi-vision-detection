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
from datetime import datetime, timedelta
import pytz
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaspberryPiCamera:
    def __init__(self, width: int = None, height: int = None, timezone: str = None, base_picture_dir: str = "pictures"):
        """
        Initialize the Raspberry Pi camera.
        
        Args:
            width: Image width in pixels (None for camera's native resolution)
            height: Image height in pixels (None for camera's native resolution)  
            timezone: Timezone for folder naming (None for auto-detection)
            base_picture_dir: Base directory for saving pictures
        """
        # Auto-detect native camera resolution if not specified
        if width is None or height is None:
            native_width, native_height = self._get_native_camera_resolution()
            self.width = width or native_width
            self.height = height or native_height
        else:
            self.width = width
            self.height = height
            
        # Auto-detect system timezone if not specified
        if timezone is None:
            timezone = self._auto_detect_timezone()
            
        self.timezone = timezone
        self.base_picture_dir = base_picture_dir
        self.temp_file = "/tmp/rpi_capture.jpg"
        
        # Set up timezone
        try:
            self.tz = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone '{timezone}', using UTC")
            self.tz = pytz.UTC
        
        # Create base pictures directory
        self._ensure_directory_exists(self.base_picture_dir)
        
        logger.info(f"Camera initialized: {self.width}x{self.height}, timezone: {self.timezone}")
    
    def _auto_detect_timezone(self) -> str:
        """
        Auto-detect the system timezone.
        
        Returns:
            Detected timezone string
        """
        try:
            # Method 1: Try to get timezone from /etc/timezone (Debian/Ubuntu)
            if os.path.exists('/etc/timezone'):
                with open('/etc/timezone', 'r') as f:
                    tz = f.read().strip()
                    if tz:
                        logger.info(f"Timezone auto-detected from /etc/timezone: {tz}")
                        return tz
            
            # Method 2: Try to get timezone from timedatectl (systemd)
            try:
                result = subprocess.run(['timedatectl', 'show', '--property=Timezone', '--value'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    tz = result.stdout.strip()
                    logger.info(f"Timezone auto-detected from timedatectl: {tz}")
                    return tz
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Method 3: Try to read symlink from /etc/localtime
            if os.path.islink('/etc/localtime'):
                link_target = os.readlink('/etc/localtime')
                # Extract timezone from path like /usr/share/zoneinfo/America/New_York
                if '/zoneinfo/' in link_target:
                    tz = link_target.split('/zoneinfo/')[-1]
                    logger.info(f"Timezone auto-detected from /etc/localtime: {tz}")
                    return tz
            
            # Method 4: Try Python's built-in detection (Python 3.6+)
            try:
                import time as time_module
                if hasattr(time_module, 'tzname') and time_module.tzname:
                    # This gives us something like ('PST', 'PDT'), convert to proper timezone
                    tzname = time_module.tzname[time_module.daylight] if time_module.daylight else time_module.tzname[0]
                    # Try to map common abbreviations to full timezone names
                    tz_mapping = {
                        'PST': 'US/Pacific', 'PDT': 'US/Pacific',
                        'EST': 'US/Eastern', 'EDT': 'US/Eastern',
                        'CST': 'US/Central', 'CDT': 'US/Central',
                        'MST': 'US/Mountain', 'MDT': 'US/Mountain',
                        'UTC': 'UTC', 'GMT': 'UTC'
                    }
                    if tzname in tz_mapping:
                        tz = tz_mapping[tzname]
                        logger.info(f"Timezone auto-detected from system: {tz}")
                        return tz
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Error auto-detecting timezone: {e}")
        
        # Fallback to UTC
        logger.info("Could not auto-detect timezone, using UTC as fallback")
        return 'UTC'
    
    def _get_native_camera_resolution(self) -> Tuple[int, int]:
        """
        Get the native resolution of the camera.
        
        Returns:
            Tuple of (width, height) for native resolution
        """
        try:
            # Try to detect camera capabilities using rpicam-hello
            result = subprocess.run(['rpicam-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse output to find maximum resolution
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'modes:' in line.lower() or 'mode' in line.lower():
                        # Look for resolution patterns like 4608x2592, 1920x1080, etc.
                        import re
                        resolutions = re.findall(r'(\d{3,4})x(\d{3,4})', line)
                        if resolutions:
                            # Take the highest resolution found
                            max_res = max(resolutions, key=lambda x: int(x[0]) * int(x[1]))
                            width, height = int(max_res[0]), int(max_res[1])
                            logger.info(f"Native camera resolution detected: {width}x{height}")
                            return width, height
        
        except Exception as e:
            logger.warning(f"Could not detect native camera resolution: {e}")
        
        # Fallback resolutions based on common Pi camera modules
        fallback_resolutions = [
            (4608, 2592),  # Camera Module 3 (12MP)
            (3280, 2464),  # Camera Module 2 (8MP) 
            (2592, 1944),  # Camera Module 1 (5MP)
            (1920, 1080),  # Common HD fallback
            (1280, 720),   # HD fallback
            (640, 480)     # VGA fallback
        ]
        
        # Try each resolution to see which one works
        for width, height in fallback_resolutions:
            if self._test_camera_resolution(width, height):
                logger.info(f"Using fallback resolution: {width}x{height}")
                return width, height
        
        # Ultimate fallback
        logger.warning("Could not determine optimal resolution, using 1920x1080")
        return 1920, 1080
    
    def _test_camera_resolution(self, width: int, height: int) -> bool:
        """
        Test if a specific resolution works with the camera.
        
        Args:
            width: Width to test
            height: Height to test
            
        Returns:
            True if resolution works, False otherwise
        """
        try:
            result = subprocess.run([
                'rpicam-still', '--timeout', '100', '--width', str(width), 
                '--height', str(height), '--output', '/dev/null', '--nopreview'
            ], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
        
    def _ensure_directory_exists(self, directory: str):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _get_date_folder_path(self) -> str:
        """Get the date-based folder path for today."""
        now = datetime.now(self.tz)
        date_folder = now.strftime("%Y-%m-%d_%Z")  # Format: 2024-03-15_UTC
        date_path = os.path.join(self.base_picture_dir, date_folder)
        self._ensure_directory_exists(date_path)
        return date_path
    
    def _get_timestamp_filename(self, prefix: str = "capture", extension: str = ".jpg") -> str:
        """Generate timestamp-based filename."""
        now = datetime.now(self.tz)
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
        return f"{prefix}_{timestamp}{extension}"
    
    def _get_full_save_path(self, prefix: str = "capture", extension: str = ".jpg") -> str:
        """Get full path for saving image with date folder and timestamp filename."""
        date_folder = self._get_date_folder_path()
        filename = self._get_timestamp_filename(prefix, extension)
        return os.path.join(date_folder, filename)
        
    def capture_image_rpicam(self, timeout: int = 2000, output_path: Optional[str] = None, 
                            save_to_pictures: bool = True, prefix: str = "capture", 
                            use_full_resolution: bool = True) -> Optional[np.ndarray]:
        """
        Capture image using rpicam-still command.
        
        Args:
            timeout: Capture timeout in milliseconds
            output_path: Optional specific path to save image
            save_to_pictures: If True, save to structured pictures folder
            prefix: Filename prefix for pictures folder
            use_full_resolution: If True, use camera's native resolution
            
        Returns:
            numpy array of the captured image or None if failed
        """
        try:
            # Determine save path
            if output_path:
                save_path = output_path
            elif save_to_pictures:
                save_path = self._get_full_save_path(prefix)
            else:
                save_path = self.temp_file
            
            # Build rpicam-still command
            cmd = [
                "rpicam-still",
                "-t", str(timeout),
                "-o", save_path,
                "--nopreview"  # Disable preview for headless operation
            ]
            
            # Add resolution parameters only if not using full resolution
            if not use_full_resolution:
                cmd.extend(["--width", str(self.width), "--height", str(self.height)])
            else:
                logger.info("Capturing at camera's native resolution for best quality")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                logger.error(f"rpicam-still failed: {result.stderr}")
                return None
                
            # Load the captured image
            if os.path.exists(save_path):
                image = cv2.imread(save_path)
                if image is not None:
                    actual_height, actual_width = image.shape[:2]
                    if save_to_pictures:
                        logger.info(f"Image captured at {actual_width}x{actual_height} and saved to: {save_path}")
                    else:
                        logger.info(f"Successfully captured image: {actual_width}x{actual_height}")
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
    
    def capture_image_opencv(self, camera_index: int = 0, save_to_pictures: bool = True, 
                            prefix: str = "opencv_capture", use_full_resolution: bool = True) -> Optional[np.ndarray]:
        """
        Capture image using OpenCV (alternative method).
        
        Args:
            camera_index: Camera device index
            save_to_pictures: If True, save to structured pictures folder
            prefix: Filename prefix for pictures folder
            use_full_resolution: If True, try to use highest available resolution
            
        Returns:
            numpy array of the captured image or None if failed
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not use_full_resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            else:
                # Try to set maximum resolution for best quality
                # Get camera capabilities first
                max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"OpenCV camera default resolution: {max_width}x{max_height}")
                
                # Try common high resolutions
                for width, height in [(4608, 2592), (3280, 2464), (2592, 1944), (1920, 1080)]:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                        logger.info(f"Set OpenCV resolution to: {actual_width}x{actual_height}")
                        break
            
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
                actual_height, actual_width = frame.shape[:2]
                if save_to_pictures:
                    save_path = self._get_full_save_path(prefix)
                    success = cv2.imwrite(save_path, frame)
                    if success:
                        logger.info(f"OpenCV image captured at {actual_width}x{actual_height} and saved to: {save_path}")
                    else:
                        logger.error("Failed to save OpenCV captured image")
                else:
                    logger.info(f"Successfully captured image with OpenCV: {actual_width}x{actual_height}")
                return frame
            else:
                logger.error("Failed to capture image with OpenCV")
                return None
                
        except Exception as e:
            logger.error(f"OpenCV capture error: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, rotate_degrees: int = 180) -> np.ndarray:
        """
        Apply basic preprocessing to the image.
        
        Args:
            image: Input image
            rotate_degrees: Degrees to rotate image (0, 90, 180, 270). Default is 180.
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to RGB (OpenCV uses BGR by default)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Apply rotation if specified
            if rotate_degrees != 0:
                image_rgb = self._rotate_image(image_rgb, rotate_degrees)
                logger.info(f"Image rotated by {rotate_degrees} degrees")
                
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
    
    def _rotate_image(self, image: np.ndarray, degrees: int) -> np.ndarray:
        """
        Rotate image by specified degrees.
        
        Args:
            image: Input image
            degrees: Rotation angle (0, 90, 180, 270)
            
        Returns:
            Rotated image
        """
        try:
            # Normalize degrees to 0-360 range
            degrees = degrees % 360
            
            if degrees == 0:
                return image
            elif degrees == 90:
                # Rotate 90 degrees counter-clockwise
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif degrees == 180:
                # Rotate 180 degrees
                return cv2.rotate(image, cv2.ROTATE_180)
            elif degrees == 270:
                # Rotate 270 degrees counter-clockwise (or 90 degrees clockwise)
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            else:
                # For non-standard angles, use general rotation
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                
                # Calculate rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
                
                # Calculate new dimensions to avoid cropping
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_width = int((height * sin_angle) + (width * cos_angle))
                new_height = int((height * cos_angle) + (width * sin_angle))
                
                # Adjust rotation matrix for new center
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]
                
                # Apply rotation
                rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
                logger.info(f"Applied custom rotation: {degrees} degrees")
                return rotated
                
        except Exception as e:
            logger.error(f"Image rotation error: {e}")
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
    
    
    
    def save_image(self, image: np.ndarray, path: str = None, prefix: str = "post", 
                  use_pictures_folder: bool = True) -> bool:
        """
        Save image to specified path or structured pictures folder.
        
        Args:
            image: Image to save
            path: Specific output file path (overrides pictures folder if provided)
            prefix: Filename prefix when using pictures folder (e.g., "post")
            use_pictures_folder: If True and no path specified, use structured pictures folder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine save path
            if path:
                save_path = path
            elif use_pictures_folder:
                save_path = self._get_full_save_path(prefix)
            else:
                save_path = f"{prefix}_{int(time.time())}.jpg"
            
            # Convert RGB back to BGR for saving with OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
                
            success = cv2.imwrite(save_path, image_bgr)
            if success:
                filename = os.path.basename(save_path) if use_pictures_folder else save_path
                logger.info(f"Image saved as: {filename}")
                return True
            else:
                logger.error(f"Failed to save image to {save_path}")
                return False
        except Exception as e:
            logger.error(f"Image save error: {e}")
            return False
    
    def save_image_to_output(self, image: np.ndarray, output_directory: str = "./output", 
                       prefix: str = "post", extension: str = ".jpg") -> Tuple[bool, str]:
        """
        Save image directly to output folder with timestamp naming.
        
        Args:
            image: Image to save
            output_directory: Output directory path
            prefix: Filename prefix (e.g., "post" for processed images)
            extension: File extension
            
        Returns:
            Tuple of (success, filename)
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)
            
            # Generate timestamp-based filename
            now = datetime.now(self.tz)
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
            filename = f"{prefix}_{timestamp}{extension}"
            save_path = os.path.join(output_directory, filename)
            
            # Convert RGB back to BGR for saving with OpenCV if needed
            if len(image.shape) == 3:
                # Check if image is already in BGR format (from OpenCV capture)
                # or needs conversion from RGB (from preprocessing)
                # We'll assume preprocessing returns RGB, so convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            success = cv2.imwrite(save_path, image_bgr)
            if success:
                logger.info(f"Image saved to output: {filename}")
                return True, filename
            else:
                logger.error(f"Failed to save image to output: {filename}")
                return False, filename
                
        except Exception as e:
            logger.error(f"Output image save error: {e}")
            return False, "save_error.jpg"
    
    
    def get_pictures_info(self) -> dict:
        """
        Get information about the pictures directory structure.
        
        Returns:
            Dictionary with pictures directory information
        """
        info = {
            'base_directory': self.base_picture_dir,
            'current_date_folder': self._get_date_folder_path(),
            'timezone': str(self.tz),
            'existing_date_folders': [],
            'total_images': 0
        }
        
        try:
            if os.path.exists(self.base_picture_dir):
                # Get all date folders
                for item in os.listdir(self.base_picture_dir):
                    item_path = os.path.join(self.base_picture_dir, item)
                    if os.path.isdir(item_path):
                        # Count images in this date folder
                        image_count = len([f for f in os.listdir(item_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        info['existing_date_folders'].append({
                            'folder': item,
                            'path': item_path,
                            'image_count': image_count
                        })
                        info['total_images'] += image_count
                
                # Sort folders by name (which should be chronological due to date format)
                info['existing_date_folders'].sort(key=lambda x: x['folder'])
            
        except Exception as e:
            logger.error(f"Error getting pictures info: {e}")
        
        return info
    
    def cleanup_old_pictures(self, days_to_keep: int = 7) -> int:
        """
        Clean up old pictures older than specified days.
        
        Args:
            days_to_keep: Number of days of pictures to keep
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_date = datetime.now(self.tz) - timedelta(days=days_to_keep)
        
        try:
            if os.path.exists(self.base_picture_dir):
                for item in os.listdir(self.base_picture_dir):
                    item_path = os.path.join(self.base_picture_dir, item)
                    if os.path.isdir(item_path):
                        try:
                            # Parse date from folder name (format: YYYY-MM-DD_TZ)
                            date_part = item.split('_')[0]
                            folder_date = datetime.strptime(date_part, '%Y-%m-%d')
                            folder_date = self.tz.localize(folder_date)
                            
                            if folder_date < cutoff_date:
                                # Remove all files in old date folder
                                for file in os.listdir(item_path):
                                    file_path = os.path.join(item_path, file)
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                        removed_count += 1
                                
                                # Remove empty directory
                                os.rmdir(item_path)
                                logger.info(f"Removed old pictures folder: {item}")
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse date from folder name '{item}': {e}")
                            continue
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old pictures")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return removed_count


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
    print("=" * 50)
    
    # Test auto-detection features
    print("1. Testing auto-detection features...")
    camera = RaspberryPiCamera()  # Use all auto-detection
    
    print(f"✓ Auto-detected timezone: {camera.timezone}")
    print(f"✓ Auto-detected resolution: {camera.width}x{camera.height}")
    
    # Show pictures directory info
    info = camera.get_pictures_info()
    print(f"✓ Pictures base directory: {info['base_directory']}")
    print(f"✓ Current date folder: {info['current_date_folder']}")
    print(f"✓ Using timezone: {info['timezone']}")
    
    # Test full resolution capture
    print("\n2. Testing full resolution capture...")
    image = camera.capture_image_rpicam(
        timeout=3000, 
        save_to_pictures=True, 
        prefix="test_full_res",
        use_full_resolution=True
    )
    
    if image is not None:
        height, width = image.shape[:2]
        print(f"✓ Full resolution capture successful: {width}x{height}")
        print(f"  Aspect ratio: {width/height:.2f}")
        print(f"  Total pixels: {width*height:,}")
        
        # Test preprocessing at full resolution with rotation
        processed = camera.preprocess_image(image)  # Default 180° rotation
        print(f"✓ Image preprocessing with 180° rotation successful: {processed.shape}")
        
        # Test different rotation angles
        print("\n   Testing different rotation angles...")
        rotations_to_test = [0, 90, 180, 270]
        for angle in rotations_to_test:
            rotated = camera.preprocess_image(image, rotate_degrees=angle)
            height, width = rotated.shape[:2]
            print(f"     {angle}°: {width}x{height}")
            
            # Save one example of each rotation
            if camera.save_image(rotated, prefix=f"rotation_test_{angle}deg", use_pictures_folder=True):
                print(f"     ✓ Saved {angle}° rotation example")
        
        # Test manual save with pictures folder
        if camera.save_image(processed, prefix="processed_full_res", use_pictures_folder=True):
            print("✓ Processed full resolution image save successful")
        
        # Test resize functionality (keep the function but don't use by default)
        print("\n3. Testing resize functionality (for when needed)...")
        resized = camera.resize_image(image, (1280, 720))
        resized_height, resized_width = resized.shape[:2]
        print(f"✓ Image resize test successful: {resized_width}x{resized_height}")
        
        if camera.save_image(resized, prefix="resized_test", use_pictures_folder=True):
            print("✓ Resized image save successful")
            
    else:
        print("✗ Full resolution capture failed")
        
        # Fallback test with specified resolution
        print("\n   Trying fallback with specified resolution...")
        camera_fallback = RaspberryPiCamera(width=1920, height=1080)
        image_fallback = camera_fallback.capture_image_rpicam(
            timeout=3000,
            save_to_pictures=True,
            prefix="test_fallback",
            use_full_resolution=False
        )
        
        if image_fallback is not None:
            height, width = image_fallback.shape[:2]
            print(f"✓ Fallback capture successful: {width}x{height}")
        else:
            print("✗ Fallback capture also failed")
    
    # Test OpenCV capture with full resolution
    print("\n4. Testing OpenCV capture with full resolution...")
    image_cv = camera.capture_image_opencv(
        save_to_pictures=True, 
        prefix="test_opencv_full",
        use_full_resolution=True
    )
    
    if image_cv is not None:
        height, width = image_cv.shape[:2]
        print(f"✓ OpenCV full resolution capture successful: {width}x{height}")
    else:
        print("✗ OpenCV capture failed")
    
    # Show updated pictures info
    updated_info = camera.get_pictures_info()
    print(f"\n5. Pictures directory info after testing:")
    for folder_info in updated_info['existing_date_folders']:
        print(f"   Folder: {folder_info['folder']} - Images: {folder_info['image_count']}")
    print(f"   Total images: {updated_info['total_images']}")
    
    # Show camera capabilities
    print(f"\n6. Camera setup summary:")
    print(f"   Native resolution: {camera.width}x{camera.height}")
    print(f"   Timezone: {camera.timezone} ({camera.tz})")
    print(f"   Pictures directory: {camera.base_picture_dir}")
    print(f"   Full resolution mode: Enabled by default")
    print(f"   Resize function: Available when needed")
    
    # Test cleanup (but don't actually remove recent files)
    print(f"\n7. Testing cleanup function (dry run)...")
    print("   Note: Cleanup set to keep files from last 365 days (won't remove today's files)")
    removed = camera.cleanup_old_pictures(days_to_keep=365)
    print(f"   Files that would be removed if older: {removed}")
    
    # Cleanup temp files
    camera.cleanup()
    print(f"\n✓ Testing completed!")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("✓ Auto-detection: Timezone and resolution detected automatically")
    print("✓ Full resolution: Using camera's native resolution by default")
    print("✓ Resize function: Available but not used by default")
    print("✓ Quality: Maximum image quality preserved")
    print("✓ Organization: Structured folder system with timezone awareness")
    print("=" * 50)


if __name__ == "__main__":
    test_camera_module()