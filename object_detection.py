#!/usr/bin/env python3
"""
Object detection module for Raspberry Pi 5 vision system.
Supports multiple detection methods: YOLO, Haar Cascades, and contour-based detection.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Optional
import urllib.request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self):
        """Initialize the object detector."""
        self.yolo_net = None
        self.yolo_classes = []
        self.yolo_output_layers = []
        self.haar_cascades = {}
        
    def load_yolo_model(self, 
                       weights_path: str = "yolov4-tiny.weights",
                       config_path: str = "yolov4-tiny.cfg",
                       classes_path: str = "coco.names") -> bool:
        """
        Load YOLO model for object detection.
        
        Args:
            weights_path: Path to YOLO weights file
            config_path: Path to YOLO config file
            classes_path: Path to class names file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download files if they don't exist
            self._download_yolo_files(weights_path, config_path, classes_path)
            
            # Load YOLO network
            self.yolo_net = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            
            # Get output layer names
            layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            
            logger.info(f"YOLO model loaded successfully with {len(self.yolo_classes)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def _download_yolo_files(self, weights_path: str, config_path: str, classes_path: str):
        """Download YOLO files if they don't exist."""
        files_to_download = [
            (weights_path, "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"),
            (config_path, "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"),
            (classes_path, "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
        ]
        
        for file_path, url in files_to_download:
            if not os.path.exists(file_path):
                try:
                    logger.info(f"Downloading {file_path}...")
                    urllib.request.urlretrieve(url, file_path)
                    logger.info(f"Downloaded {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to download {file_path}: {e}")
    
    def detect_objects_yolo(self, image: np.ndarray, 
                           confidence_threshold: float = 0.5,
                           nms_threshold: float = 0.4) -> List[Dict]:
        """
        Detect objects using YOLO model.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        if self.yolo_net is None:
            logger.error("YOLO model not loaded")
            return []
        
        try:
            height, width = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
            outputs = self.yolo_net.forward(self.yolo_output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Process each output
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate bounding box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_name = self.yolo_classes[class_ids[i]] if class_ids[i] < len(self.yolo_classes) else "unknown"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidences[i],
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2)
                    })
            
            logger.info(f"YOLO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def load_haar_cascade(self, cascade_name: str, cascade_path: str = None) -> bool:
        """
        Load Haar cascade classifier.
        
        Args:
            cascade_name: Name identifier for the cascade
            cascade_path: Path to cascade XML file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cascade_path is None:
                # Use built-in OpenCV cascades
                cascade_path = cv2.data.haarcascades + f'haarcascade_{cascade_name}.xml'
            
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                logger.error(f"Failed to load cascade: {cascade_path}")
                return False
            
            self.haar_cascades[cascade_name] = cascade
            logger.info(f"Loaded Haar cascade: {cascade_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Haar cascade {cascade_name}: {e}")
            return False
    
    def detect_objects_haar(self, image: np.ndarray, 
                           cascade_name: str,
                           scale_factor: float = 1.1,
                           min_neighbors: int = 5) -> List[Dict]:
        """
        Detect objects using Haar cascade.
        
        Args:
            image: Input image
            cascade_name: Name of the cascade to use
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            
        Returns:
            List of detected objects with bounding boxes
        """
        if cascade_name not in self.haar_cascades:
            logger.error(f"Haar cascade '{cascade_name}' not loaded")
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            objects = self.haar_cascades[cascade_name].detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
            )
            
            detections = []
            for (x, y, w, h) in objects:
                detections.append({
                    'class': cascade_name,
                    'confidence': 1.0,  # Haar cascades don't provide confidence scores
                    'bbox': (x, y, w, h),
                    'center': (x + w // 2, y + h // 2)
                })
            
            logger.info(f"Haar cascade detected {len(detections)} {cascade_name} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Haar cascade detection error: {e}")
            return []
    
    def detect_objects_contour(self, image: np.ndarray,
                             color_range: Dict = None,
                             min_area: int = 500) -> List[Dict]:
        """
        Detect objects using contour detection with color filtering.
        
        Args:
            image: Input image
            color_range: Dictionary with 'lower' and 'upper' HSV color bounds
            min_area: Minimum contour area for detection
            
        Returns:
            List of detected objects with bounding boxes
        """
        try:
            # Default to detect red objects if no color range specified
            if color_range is None:
                color_range = {
                    'lower': np.array([0, 50, 50]),    # Lower HSV bound for red
                    'upper': np.array([10, 255, 255])  # Upper HSV bound for red
                }
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Create mask
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'class': 'colored_object',
                        'confidence': area / 10000.0,  # Use area as confidence metric
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'area': area
                    })
            
            logger.info(f"Contour detection found {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Contour detection error: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        try:
            result_image = image.copy()
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Choose color based on class
                color = self._get_class_color(class_name)
                
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(result_image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for each class."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        hash_value = hash(class_name) % len(colors)
        return colors[hash_value]


def test_object_detection():
    """Test function to verify object detection functionality."""
    print("Testing Object Detection Module...")
    
    detector = ObjectDetector()
    
    # Create a test image (colored rectangle)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.rectangle(test_image, (300, 300), (400, 400), (0, 255, 0), -1)  # Green rectangle
    
    print("\n1. Testing contour-based detection...")
    # Test contour detection for red objects
    red_range = {
        'lower': np.array([0, 100, 100]),
        'upper': np.array([10, 255, 255])
    }
    detections = detector.detect_objects_contour(test_image, red_range, min_area=1000)
    print(f"✓ Contour detection found {len(detections)} red objects")
    
    # Draw detections
    result_image = detector.draw_detections(test_image, detections)
    cv2.imwrite("test_contour_detection.jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print("✓ Detection results saved to test_contour_detection.jpg")
    
    print("\n2. Testing Haar cascade loading...")
    # Test loading face cascade
    if detector.load_haar_cascade("frontalface_default"):
        print("✓ Haar cascade loaded successfully")
        
        # Test detection on a dummy image
        gray_test = np.ones((480, 640), dtype=np.uint8) * 128
        haar_detections = detector.detect_objects_haar(gray_test, "frontalface_default")
        print(f"✓ Haar detection completed (found {len(haar_detections)} faces)")
    else:
        print("✗ Failed to load Haar cascade")
    
    print("\n3. Testing YOLO model loading...")
    if detector.load_yolo_model():
        print("✓ YOLO model loaded successfully")
        
        # Test YOLO detection
        yolo_detections = detector.detect_objects_yolo(test_image)
        print(f"✓ YOLO detection completed (found {len(yolo_detections)} objects)")
    else:
        print("✗ YOLO model loading failed (this is normal if model files aren't available)")
    
    print("\nObject detection module testing completed!")


if __name__ == "__main__":
    test_object_detection()