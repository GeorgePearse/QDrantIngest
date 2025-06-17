#!/usr/bin/env python3
"""
QDrantIngest: Convert COCO annotations to QDrant vector database with Jina AI embeddings.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# Will implement these modules
from qdrantingest.coco_parser import CocoParser
from qdrantingest.image_processor import ImageProcessor
from qdrantingest.embedding_generator import EmbeddingGenerator
from qdrantingest.qdrant_uploader import QdrantUploader


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to QDrant vector database with Jina AI embeddings."
    )
    parser.add_argument(
        "--annotations", 
        type=str, 
        required=True,
        help="Path to COCO annotations JSON file"
    )
    parser.add_argument(
        "--images", 
        type=str, 
        required=True,
        help="Path to directory containing source images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./qdrant_db",
        help="Path to store QDrant database (default: ./qdrant_db)"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="coco_objects",
        help="Name of QDrant collection to create (default: coco_objects)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--embedding-model", 
        type=str, 
        default="jina-embeddings-v2-base-en",
        help="Jina AI embedding model to use (default: jina-embeddings-v2-base-en)"
    )
    parser.add_argument(
        "--vector-size", 
        type=int, 
        default=768,
        help="Vector size for the embedding model (default: 768)"
    )
    parser.add_argument(
        "--use-segmentation", 
        action="store_true",
        help="Use segmentation masks for cropping instead of bounding boxes"
    )
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip processing if collection already exists"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for QDrantIngest."""
    args = parse_args()
    
    # Validate input paths
    annotations_path = Path(args.annotations)
    images_path = Path(args.images)
    output_path = Path(args.output)
    
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return 1
    
    if not images_path.exists() or not images_path.is_dir():
        print(f"Error: Images directory not found: {images_path}")
        return 1
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing COCO annotations: {annotations_path}")
    print(f"Images directory: {images_path}")
    print(f"Output path: {output_path}")
    print(f"Collection name: {args.collection}")
    
    try:
        # Parse COCO annotations
        parser = CocoParser(annotations_path)
        coco_data = parser.parse()
        
        print(f"Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
        
        # Initialize image processor
        image_processor = ImageProcessor(
            images_dir=images_path,
            use_segmentation=args.use_segmentation
        )
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(
            model_name=args.embedding_model,
            vector_size=args.vector_size
        )
        
        # Initialize QDrant uploader
        qdrant_uploader = QdrantUploader(
            path=str(output_path),
            collection_name=args.collection,
            vector_size=args.vector_size
        )
        
        # Check if collection already exists
        if args.skip_existing and qdrant_uploader.collection_exists():
            print(f"Collection '{args.collection}' already exists. Skipping processing.")
            return 0
        
        # Process objects in batches
        total_objects = len(coco_data['annotations'])
        batch_size = args.batch_size
        
        with tqdm(total=total_objects, desc="Processing objects") as pbar:
            for i in range(0, total_objects, batch_size):
                batch_annotations = coco_data['annotations'][i:i+batch_size]
                
                # Get image data for this batch
                image_ids = [ann['image_id'] for ann in batch_annotations]
                image_data = {
                    img['id']: img for img in coco_data['images'] 
                    if img['id'] in image_ids
                }
                
                # Process images to get cropped objects
                cropped_objects = []
                for ann in batch_annotations:
                    image_info = image_data[ann['image_id']]
                    category = next(
                        cat for cat in coco_data['categories'] 
                        if cat['id'] == ann['category_id']
                    )
                    
                    # Crop the object
                    cropped_image = image_processor.crop_object(
                        image_filename=image_info['file_name'],
                        bbox=ann.get('bbox'),
                        segmentation=ann.get('segmentation')
                    )
                    
                    if cropped_image is not None:
                        cropped_objects.append({
                            'image': cropped_image,
                            'annotation': ann,
                            'image_info': image_info,
                            'category': category
                        })
                
                # Generate embeddings for cropped objects
                if cropped_objects:
                    images = [obj['image'] for obj in cropped_objects]
                    embeddings = embedding_generator.generate_embeddings(images)
                    
                    # Prepare objects for upload
                    upload_objects = []
                    for i, obj in enumerate(cropped_objects):
                        upload_objects.append({
                            'id': obj['annotation']['id'],
                            'vector': embeddings[i],
                            'payload': {
                                'image_id': obj['image_info']['id'],
                                'file_name': obj['image_info']['file_name'],
                                'category_id': obj['category']['id'],
                                'category_name': obj['category']['name'],
                                'bbox': obj['annotation'].get('bbox'),
                                'segmentation': obj['annotation'].get('segmentation'),
                                'area': obj['annotation'].get('area'),
                                'iscrowd': obj['annotation'].get('iscrowd', 0)
                            }
                        })
                    
                    # Upload to QDrant
                    qdrant_uploader.upload_batch(upload_objects)
                
                pbar.update(len(batch_annotations))
        
        print(f"Successfully processed {total_objects} objects.")
        print(f"QDrant collection '{args.collection}' created at {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())