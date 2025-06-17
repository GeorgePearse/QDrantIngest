"""
Image processor for cropping objects from COCO annotated images.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from PIL import Image, ImageDraw


class ImageProcessor:
    """
    Processor for cropping objects from images based on COCO annotations.
    
    This class handles loading images and cropping objects based on either
    bounding boxes or segmentation masks from COCO annotations.
    """
    
    def __init__(self, images_dir: Union[str, Path], use_segmentation: bool = False):
        """
        Initialize the image processor.
        
        Args:
            images_dir: Directory containing the source images
            use_segmentation: Whether to use segmentation masks for cropping (if available)
                              instead of bounding boxes
        """
        self.images_dir = Path(images_dir)
        self.use_segmentation = use_segmentation
    
    def load_image(self, image_filename: str) -> Optional[Image.Image]:
        """
        Load an image from the images directory.
        
        Args:
            image_filename: Filename of the image to load
            
        Returns:
            Loaded PIL Image or None if the image could not be loaded
        """
        image_path = self.images_dir / image_filename
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return None
        
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def crop_object(
        self, 
        image_filename: str, 
        bbox: Optional[List[float]] = None,
        segmentation: Optional[List[List[float]]] = None
    ) -> Optional[Image.Image]:
        """
        Crop an object from an image using either bounding box or segmentation mask.
        
        Args:
            image_filename: Filename of the image to crop from
            bbox: Bounding box in COCO format [x, y, width, height]
            segmentation: Segmentation mask in COCO format
                          (list of polygons, each a flattened list of x,y coordinates)
            
        Returns:
            Cropped PIL Image or None if cropping failed
        """
        image = self.load_image(image_filename)
        if image is None:
            return None
        
        # Use segmentation if available and requested
        if self.use_segmentation and segmentation and len(segmentation) > 0:
            return self._crop_by_segmentation(image, segmentation)
        elif bbox and len(bbox) == 4:
            return self._crop_by_bbox(image, bbox)
        else:
            print(f"Warning: No valid bbox or segmentation found for {image_filename}")
            return None
    
    def _crop_by_bbox(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        Crop an object using a bounding box.
        
        Args:
            image: Source image
            bbox: Bounding box in COCO format [x, y, width, height]
            
        Returns:
            Cropped image
        """
        x, y, width, height = bbox
        # Convert to integers and ensure within image bounds
        x = max(0, int(x))
        y = max(0, int(y))
        width = min(int(width), image.width - x)
        height = min(int(height), image.height - y)
        
        # Ensure minimum size
        if width < 1 or height < 1:
            print(f"Warning: Invalid bbox size: {width}x{height}")
            # Return a small portion of the image to avoid errors
            return image.crop((0, 0, min(10, image.width), min(10, image.height)))
        
        return image.crop((x, y, x + width, y + height))
    
    def _crop_by_segmentation(
        self, image: Image.Image, segmentation: List[List[float]]
    ) -> Image.Image:
        """
        Crop an object using a segmentation mask.
        
        Args:
            image: Source image
            segmentation: Segmentation mask in COCO format
                         (list of polygons, each a flattened list of x,y coordinates)
            
        Returns:
            Cropped image with mask applied
        """
        # Create a binary mask from the segmentation
        mask = Image.new('L', (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)
        
        for polygon in segmentation:
            # Convert flat list to list of (x,y) tuples
            points = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    points.append((polygon[i], polygon[i + 1]))
            
            if len(points) >= 3:  # Need at least 3 points for a polygon
                draw.polygon(points, fill=255)
        
        # Find the bounding box of the mask
        bbox = mask.getbbox()
        if not bbox:
            # Fallback to a small section if mask is empty
            return image.crop((0, 0, min(10, image.width), min(10, image.height)))
        
        # Create a cropped version of both the image and mask
        cropped_image = image.crop(bbox)
        cropped_mask = mask.crop(bbox)
        
        # Apply the mask to the cropped image
        # Convert to RGBA to support transparency
        result = Image.new('RGBA', cropped_image.size, (0, 0, 0, 0))
        cropped_image = cropped_image.convert('RGBA')
        
        # Composite the image with the mask
        for x in range(cropped_image.width):
            for y in range(cropped_image.height):
                mask_value = cropped_mask.getpixel((x, y))
                if mask_value > 0:  # If mask is non-zero at this pixel
                    result.putpixel((x, y), cropped_image.getpixel((x, y)))
        
        # Convert back to RGB for compatibility with embedding models
        return result.convert('RGB')
    
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """
        Preprocess an image for embedding generation.
        
        Args:
            image: Source image
            target_size: Size to resize the image to
            
        Returns:
            Preprocessed image
        """
        # Resize the image to the target size
        return image.resize(target_size, Image.Resampling.LANCZOS)