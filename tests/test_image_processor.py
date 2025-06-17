"""
Tests for the image processor module.
"""

import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image
import numpy as np

from qdrantingest.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a test image
        self.test_image_path = self.temp_path / "test_image.jpg"
        test_image = Image.new('RGB', (300, 200), color='white')
        test_image.save(self.test_image_path)
        
        # Initialize processor
        self.processor = ImageProcessor(self.temp_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_load_image(self):
        """Test loading an image."""
        # Test loading a valid image
        image = self.processor.load_image("test_image.jpg")
        self.assertIsNotNone(image)
        self.assertEqual(image.width, 300)
        self.assertEqual(image.height, 200)
        
        # Test loading a non-existent image
        image = self.processor.load_image("nonexistent.jpg")
        self.assertIsNone(image)
    
    def test_crop_by_bbox(self):
        """Test cropping an image using a bounding box."""
        image = self.processor.load_image("test_image.jpg")
        
        # Test valid bbox
        bbox = [50, 40, 100, 80]
        cropped = self.processor._crop_by_bbox(image, bbox)
        self.assertEqual(cropped.width, 100)
        self.assertEqual(cropped.height, 80)
        
        # Test bbox that extends beyond image bounds
        bbox = [250, 150, 100, 100]
        cropped = self.processor._crop_by_bbox(image, bbox)
        self.assertEqual(cropped.width, 50)  # Should be clipped to image width
        self.assertEqual(cropped.height, 50)  # Should be clipped to image height
        
        # Test invalid bbox (negative size)
        bbox = [50, 40, -10, 80]
        cropped = self.processor._crop_by_bbox(image, bbox)
        self.assertIsNotNone(cropped)  # Should return a small section instead of failing
    
    def test_crop_object_with_bbox(self):
        """Test cropping an object using a bounding box."""
        # Test valid crop
        cropped = self.processor.crop_object(
            image_filename="test_image.jpg",
            bbox=[50, 40, 100, 80]
        )
        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.width, 100)
        self.assertEqual(cropped.height, 80)
        
        # Test with non-existent image
        cropped = self.processor.crop_object(
            image_filename="nonexistent.jpg",
            bbox=[50, 40, 100, 80]
        )
        self.assertIsNone(cropped)
        
        # Test with no bbox or segmentation
        cropped = self.processor.crop_object(
            image_filename="test_image.jpg"
        )
        self.assertIsNone(cropped)
    
    def test_crop_by_segmentation(self):
        """Test cropping an image using a segmentation mask."""
        # Create an instance with segmentation enabled
        processor = ImageProcessor(self.temp_path, use_segmentation=True)
        image = processor.load_image("test_image.jpg")
        
        # Simple polygon segmentation
        segmentation = [[50, 40, 150, 40, 150, 120, 50, 120]]
        
        # Mock the Image.getbbox() method to return a predictable bounding box
        with patch.object(Image.Image, 'getbbox', return_value=(50, 40, 150, 120)):
            cropped = processor._crop_by_segmentation(image, segmentation)
            self.assertIsNotNone(cropped)
            
            # Should be cropped to the bounding box of the mask
            self.assertEqual(cropped.width, 100)
            self.assertEqual(cropped.height, 80)
    
    def test_crop_object_with_segmentation(self):
        """Test cropping an object using a segmentation mask."""
        # Create an instance with segmentation enabled
        processor = ImageProcessor(self.temp_path, use_segmentation=True)
        
        # Simple polygon segmentation
        segmentation = [[50, 40, 150, 40, 150, 120, 50, 120]]
        
        # Mock the required methods to avoid actual image processing
        with patch.object(processor, '_crop_by_segmentation') as mock_crop:
            mock_crop.return_value = Image.new('RGB', (100, 80))
            
            cropped = processor.crop_object(
                image_filename="test_image.jpg",
                segmentation=segmentation
            )
            
            self.assertIsNotNone(cropped)
            mock_crop.assert_called_once()
    
    def test_preprocess_image(self):
        """Test preprocessing an image."""
        image = self.processor.load_image("test_image.jpg")
        
        # Test default target size
        processed = self.processor.preprocess_image(image)
        self.assertEqual(processed.width, 224)
        self.assertEqual(processed.height, 224)
        
        # Test custom target size
        processed = self.processor.preprocess_image(image, target_size=(160, 120))
        self.assertEqual(processed.width, 160)
        self.assertEqual(processed.height, 120)


if __name__ == '__main__':
    unittest.main()