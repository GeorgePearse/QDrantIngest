"""
Tests for the COCO parser module.
"""

import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, mock_open

from qdrantingest.coco_parser import CocoParser


class TestCocoParser(unittest.TestCase):
    """Test cases for the CocoParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Sample COCO data
        self.sample_coco_data = {
            "images": [
                {"id": 1, "width": 800, "height": 600, "file_name": "image1.jpg"},
                {"id": 2, "width": 800, "height": 600, "file_name": "image2.jpg"}
            ],
            "annotations": [
                {
                    "id": 1, 
                    "image_id": 1, 
                    "category_id": 1, 
                    "bbox": [10, 20, 100, 200],
                    "area": 20000,
                    "iscrowd": 0
                },
                {
                    "id": 2, 
                    "image_id": 1, 
                    "category_id": 2, 
                    "bbox": [150, 160, 100, 100],
                    "area": 10000,
                    "iscrowd": 0
                },
                {
                    "id": 3, 
                    "image_id": 2, 
                    "category_id": 1, 
                    "bbox": [50, 60, 120, 180],
                    "area": 21600,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "car", "supercategory": "vehicle"}
            ]
        }
        
        # Create a temporary COCO file
        self.coco_file = self.temp_path / "annotations.json"
        with open(self.coco_file, 'w') as f:
            json.dump(self.sample_coco_data, f)
        
        # Initialize parser
        self.parser = CocoParser(self.coco_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_parse(self):
        """Test parsing COCO annotations."""
        result = self.parser.parse()
        
        # Check that all required sections are present
        self.assertIn('images', result)
        self.assertIn('annotations', result)
        self.assertIn('categories', result)
        
        # Check image count
        self.assertEqual(len(result['images']), 2)
        
        # Check annotation count
        self.assertEqual(len(result['annotations']), 3)
        
        # Check category count
        self.assertEqual(len(result['categories']), 2)
    
    def test_get_image_annotations(self):
        """Test getting annotations for a specific image."""
        annotations = self.parser.get_image_annotations(1)
        
        # Should return 2 annotations for image ID 1
        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0]['id'], 1)
        self.assertEqual(annotations[1]['id'], 2)
        
        # Check for image ID 2
        annotations = self.parser.get_image_annotations(2)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]['id'], 3)
        
        # Check for non-existent image ID
        annotations = self.parser.get_image_annotations(999)
        self.assertEqual(len(annotations), 0)
    
    def test_get_category_annotations(self):
        """Test getting annotations for a specific category."""
        annotations = self.parser.get_category_annotations(1)
        
        # Should return 2 annotations for category ID 1
        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0]['id'], 1)
        self.assertEqual(annotations[1]['id'], 3)
        
        # Check for category ID 2
        annotations = self.parser.get_category_annotations(2)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]['id'], 2)
        
        # Check for non-existent category ID
        annotations = self.parser.get_category_annotations(999)
        self.assertEqual(len(annotations), 0)
    
    def test_get_category_name(self):
        """Test getting category name by ID."""
        name = self.parser.get_category_name(1)
        self.assertEqual(name, "person")
        
        name = self.parser.get_category_name(2)
        self.assertEqual(name, "car")
        
        # Check for non-existent category ID
        with self.assertRaises(ValueError):
            self.parser.get_category_name(999)
    
    def test_invalid_json(self):
        """Test handling of invalid JSON file."""
        # Create an invalid JSON file
        invalid_file = self.temp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("This is not valid JSON")
        
        parser = CocoParser(invalid_file)
        with self.assertRaises(ValueError):
            parser.parse()
    
    def test_missing_required_key(self):
        """Test handling of COCO file with missing required key."""
        # Create a COCO file without annotations
        incomplete_data = {
            "images": self.sample_coco_data["images"],
            "categories": self.sample_coco_data["categories"]
            # Missing annotations
        }
        
        incomplete_file = self.temp_path / "incomplete.json"
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_data, f)
        
        parser = CocoParser(incomplete_file)
        with self.assertRaises(ValueError):
            parser.parse()


if __name__ == '__main__':
    unittest.main()