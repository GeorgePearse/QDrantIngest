"""
Parser for COCO annotation format.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class CocoParser:
    """
    Parser for COCO annotation format.
    
    The COCO (Common Objects in Context) format consists of JSON with the following structure:
    {
        "info": {...},
        "licenses": [...],
        "images": [
            {
                "id": int,
                "width": int,
                "height": int,
                "file_name": str,
                ...
            },
            ...
        ],
        "annotations": [
            {
                "id": int,
                "image_id": int,
                "category_id": int,
                "bbox": [x, y, width, height],
                "segmentation": [...],
                "area": float,
                "iscrowd": int,
                ...
            },
            ...
        ],
        "categories": [
            {
                "id": int,
                "name": str,
                "supercategory": str,
                ...
            },
            ...
        ]
    }
    """
    
    def __init__(self, annotations_path: Path):
        """
        Initialize the COCO parser.
        
        Args:
            annotations_path: Path to the COCO annotations JSON file.
        """
        self.annotations_path = annotations_path
    
    def parse(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse the COCO annotations file.
        
        Returns:
            Dict containing the parsed COCO data with keys:
            - 'images': List of image information dictionaries
            - 'annotations': List of annotation dictionaries
            - 'categories': List of category dictionaries
        """
        try:
            with open(self.annotations_path, 'r') as f:
                coco_data = json.load(f)
            
            # Extract and validate the required sections
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in coco_data:
                    raise ValueError(f"Missing required key '{key}' in COCO annotations")
            
            # Create a filtered version with just the required data
            result = {
                'images': coco_data['images'],
                'annotations': coco_data['annotations'],
                'categories': coco_data['categories']
            }
            
            # Validate that all annotations reference valid images and categories
            image_ids = {img['id'] for img in result['images']}
            category_ids = {cat['id'] for cat in result['categories']}
            
            for ann in result['annotations']:
                if ann['image_id'] not in image_ids:
                    raise ValueError(f"Annotation {ann['id']} references non-existent image ID {ann['image_id']}")
                if ann['category_id'] not in category_ids:
                    raise ValueError(f"Annotation {ann['id']} references non-existent category ID {ann['category_id']}")
            
            return result
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in COCO annotations file: {self.annotations_path}")
        except KeyError as e:
            raise ValueError(f"Missing required field in COCO annotations: {e}")
    
    def get_image_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get all annotations for a specific image.
        
        Args:
            image_id: ID of the image to get annotations for
            
        Returns:
            List of annotation dictionaries for the specified image
        """
        coco_data = self.parse()
        return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    def get_category_annotations(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get all annotations for a specific category.
        
        Args:
            category_id: ID of the category to get annotations for
            
        Returns:
            List of annotation dictionaries for the specified category
        """
        coco_data = self.parse()
        return [ann for ann in coco_data['annotations'] if ann['category_id'] == category_id]
    
    def get_category_name(self, category_id: int) -> str:
        """
        Get the name of a category by its ID.
        
        Args:
            category_id: ID of the category
            
        Returns:
            Name of the category
            
        Raises:
            ValueError: If the category ID is not found
        """
        coco_data = self.parse()
        for category in coco_data['categories']:
            if category['id'] == category_id:
                return category['name']
        raise ValueError(f"Category ID {category_id} not found")