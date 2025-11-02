#!/usr/bin/env python3
"""
Quick Start Example for QDrantIngest

This script demonstrates how to use QDrantIngest to process a COCO dataset
and create a searchable Qdrant vector database.

Usage:
    python examples/quick_start.py

Make sure to set your JINA_API_KEY environment variable first:
    export JINA_API_KEY="your-api-key-here"
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import qdrantingest modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrantingest.coco_parser import CocoParser
from qdrantingest.image_processor import ImageProcessor
from qdrantingest.embedding_generator import EmbeddingGenerator
from qdrantingest.qdrant_uploader import QdrantUploader
from tqdm import tqdm


def main():
    """Run a quick start example."""

    # Check for API key
    if not os.environ.get("JINA_API_KEY"):
        print("ERROR: JINA_API_KEY environment variable not set!")
        print("Get your API key from https://jina.ai/ and set it:")
        print('  export JINA_API_KEY="your-api-key-here"')
        return 1

    # Configuration
    # NOTE: Update these paths to point to your actual COCO dataset
    annotations_path = "data/annotations/instances_val2017.json"
    images_dir = "data/val2017/"
    output_path = "./qdrant_db_example"
    collection_name = "coco_example"

    print("=" * 60)
    print("QDrantIngest Quick Start Example")
    print("=" * 60)

    # Validate paths
    if not Path(annotations_path).exists():
        print(f"\nERROR: Annotations file not found: {annotations_path}")
        print("\nPlease update the paths in this script to point to your COCO dataset.")
        print("You can download COCO dataset from: https://cocodataset.org/")
        return 1

    if not Path(images_dir).exists():
        print(f"\nERROR: Images directory not found: {images_dir}")
        print("\nPlease update the paths in this script to point to your COCO dataset.")
        return 1

    print(f"\nAnnotations: {annotations_path}")
    print(f"Images: {images_dir}")
    print(f"Output: {output_path}")
    print(f"Collection: {collection_name}")
    print()

    # Step 1: Parse COCO annotations
    print("Step 1: Parsing COCO annotations...")
    parser = CocoParser(annotations_path)
    coco_data = parser.parse()
    print(f"  ✓ Found {len(coco_data['images'])} images")
    print(f"  ✓ Found {len(coco_data['annotations'])} annotations")
    print(f"  ✓ Found {len(coco_data['categories'])} categories")

    # Step 2: Initialize processors
    print("\nStep 2: Initializing processors...")

    image_processor = ImageProcessor(
        images_dir=images_dir,
        use_segmentation=True  # Use segmentation masks for precise cropping
    )
    print("  ✓ Image processor initialized (segmentation mode)")

    embedding_generator = EmbeddingGenerator(
        model_name="jina-embeddings-v2-base-en",
        vector_size=768
    )
    print("  ✓ Embedding generator initialized (Jina AI)")

    qdrant_uploader = QdrantUploader(
        path=output_path,
        collection_name=collection_name,
        vector_size=768
    )
    print("  ✓ QDrant uploader initialized")

    # Step 3: Process a small batch (first 10 annotations for demo)
    print("\nStep 3: Processing objects (limiting to first 10 for demo)...")
    batch_size = 10
    annotations_to_process = coco_data['annotations'][:batch_size]

    processed_count = 0
    with tqdm(total=len(annotations_to_process), desc="Processing") as pbar:
        for ann in annotations_to_process:
            # Get image info
            image_info = next(
                img for img in coco_data['images']
                if img['id'] == ann['image_id']
            )

            # Get category info
            category = next(
                cat for cat in coco_data['categories']
                if cat['id'] == ann['category_id']
            )

            # Crop object from image
            cropped_image = image_processor.crop_object(
                image_filename=image_info['file_name'],
                bbox=ann.get('bbox'),
                segmentation=ann.get('segmentation')
            )

            if cropped_image is not None:
                # Generate embedding
                embedding = embedding_generator.generate_embeddings([cropped_image])[0]

                # Upload to QDrant
                qdrant_uploader.upload_batch([{
                    'id': ann['id'],
                    'vector': embedding,
                    'payload': {
                        'image_id': image_info['id'],
                        'file_name': image_info['file_name'],
                        'category_id': category['id'],
                        'category_name': category['name'],
                        'bbox': ann.get('bbox'),
                        'segmentation': ann.get('segmentation'),
                        'area': ann.get('area'),
                        'iscrowd': ann.get('iscrowd', 0)
                    }
                }])

                processed_count += 1

            pbar.update(1)

    print(f"\n  ✓ Successfully processed {processed_count} objects")

    # Step 4: Verify the database
    print("\nStep 4: Verifying QDrant database...")
    from qdrant_client import QdrantClient

    client = QdrantClient(path=output_path)
    collection_info = client.get_collection(collection_name)
    print(f"  ✓ Collection created: {collection_name}")
    print(f"  ✓ Total vectors: {collection_info.points_count}")
    print(f"  ✓ Vector dimension: {collection_info.config.params.vectors.size}")

    # Step 5: Example search query
    print("\nStep 5: Running example similarity search...")

    # Use the first vector as a query
    first_point = client.scroll(
        collection_name=collection_name,
        limit=1
    )[0][0]

    search_results = client.search(
        collection_name=collection_name,
        query_vector=first_point.vector,
        limit=3
    )

    print(f"  ✓ Found {len(search_results)} similar objects:")
    for i, result in enumerate(search_results, 1):
        print(f"    {i}. {result.payload['category_name']} "
              f"(score: {result.score:.4f}, "
              f"file: {result.payload['file_name']})")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print(f"\nYour QDrant database is ready at: {output_path}")
    print(f"Collection name: {collection_name}")
    print("\nNext steps:")
    print("1. Process the full dataset using main.py")
    print("2. Explore querying and filtering options")
    print("3. Integrate with your application")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
