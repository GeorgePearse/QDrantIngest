#!/usr/bin/env python3
"""
Installation Verification Script for QDrantIngest

This script verifies that all dependencies are installed correctly
and that the basic pipeline works end-to-end using synthetic test data.

Usage:
    python examples/verify_installation.py

This script:
1. Checks all Python dependencies
2. Verifies JINA_API_KEY is set
3. Creates minimal test data
4. Runs the complete pipeline
5. Validates the output
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Dependencies")

    dependencies = {
        'numpy': 'numpy',
        'PIL': 'pillow',
        'qdrant_client': 'qdrant-client',
        'tqdm': 'tqdm',
        'jinaai': 'jinaai'
    }

    all_ok = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} NOT installed")
            all_ok = False

    return all_ok


def check_api_key():
    """Check if JINA_API_KEY is set."""
    print_header("Checking API Key")

    api_key = os.environ.get("JINA_API_KEY")
    if api_key:
        # Mask the key for security
        masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:]
        print_success(f"JINA_API_KEY is set: {masked_key}")
        return True
    else:
        print_error("JINA_API_KEY is NOT set")
        print("  Set it with: export JINA_API_KEY='your-api-key-here'")
        print("  Get a key from: https://jina.ai/")
        return False


def create_test_data(temp_dir):
    """Create minimal test COCO dataset."""
    print_header("Creating Test Data")

    # Create images directory
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)

    # Create a simple test image (100x100 red square)
    from PIL import Image
    import numpy as np

    test_image = Image.fromarray(
        np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
    )
    image_path = images_dir / "test_image.jpg"
    test_image.save(image_path)
    print_success(f"Created test image: {image_path}")

    # Create minimal COCO annotation
    coco_annotation = {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.jpg",
                "width": 100,
                "height": 100
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
                "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]]
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "test_object",
                "supercategory": "test"
            }
        ]
    }

    annotations_path = Path(temp_dir) / "annotations.json"
    with open(annotations_path, 'w') as f:
        json.dump(coco_annotation, f, indent=2)
    print_success(f"Created test annotations: {annotations_path}")

    return str(annotations_path), str(images_dir)


def test_pipeline(annotations_path, images_dir, output_path):
    """Test the complete pipeline."""
    print_header("Testing Pipeline")

    try:
        # Import modules
        from qdrantingest.coco_parser import CocoParser
        from qdrantingest.image_processor import ImageProcessor
        from qdrantingest.embedding_generator import EmbeddingGenerator
        from qdrantingest.qdrant_uploader import QdrantUploader

        # Step 1: Parse annotations
        print("Testing COCO parser...")
        parser = CocoParser(annotations_path)
        coco_data = parser.parse()
        print_success("COCO parser works")

        # Step 2: Process image
        print("Testing image processor...")
        processor = ImageProcessor(images_dir, use_segmentation=True)
        ann = coco_data['annotations'][0]
        img_info = coco_data['images'][0]

        cropped = processor.crop_object(
            image_filename=img_info['file_name'],
            bbox=ann['bbox'],
            segmentation=ann['segmentation']
        )
        if cropped is None:
            print_error("Image cropping failed")
            return False
        print_success("Image processor works")

        # Step 3: Generate embedding
        print("Testing embedding generator (this may take a few seconds)...")
        generator = EmbeddingGenerator(
            model_name="jina-embeddings-v2-base-en",
            vector_size=768
        )
        embeddings = generator.generate_embeddings([cropped])
        if len(embeddings) != 1 or len(embeddings[0]) != 768:
            print_error(f"Embedding generation failed (got {len(embeddings)} embeddings)")
            return False
        print_success("Embedding generator works")

        # Step 4: Upload to QDrant
        print("Testing QDrant uploader...")
        uploader = QdrantUploader(
            path=output_path,
            collection_name="test_collection",
            vector_size=768
        )

        category = coco_data['categories'][0]
        uploader.upload_batch([{
            'id': ann['id'],
            'vector': embeddings[0],
            'payload': {
                'image_id': img_info['id'],
                'file_name': img_info['file_name'],
                'category_id': category['id'],
                'category_name': category['name'],
                'bbox': ann['bbox']
            }
        }])
        print_success("QDrant uploader works")

        # Step 5: Verify database
        print("Verifying QDrant database...")
        from qdrant_client import QdrantClient

        client = QdrantClient(path=output_path)
        collection = client.get_collection("test_collection")

        if collection.points_count != 1:
            print_error(f"Expected 1 point, got {collection.points_count}")
            return False

        print_success("QDrant database verified")

        # Step 6: Test search
        print("Testing similarity search...")
        results = client.search(
            collection_name="test_collection",
            query_vector=embeddings[0],
            limit=1
        )

        if len(results) != 1:
            print_error(f"Search failed: expected 1 result, got {len(results)}")
            return False

        print_success("Similarity search works")

        return True

    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification."""
    print_header("QDrantIngest Installation Verification")

    # Check dependencies
    if not check_dependencies():
        print_error("\nSome dependencies are missing. Install them with:")
        print("  pip install -r requirements.txt")
        return 1

    # Check API key
    if not check_api_key():
        return 1

    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp(prefix="qdrantingest_test_")
    output_dir = tempfile.mkdtemp(prefix="qdrantingest_output_")

    try:
        # Create test data
        annotations_path, images_dir = create_test_data(temp_dir)

        # Test pipeline
        success = test_pipeline(annotations_path, images_dir, output_dir)

        # Print results
        print_header("Verification Results")

        if success:
            print_success("All tests passed!")
            print("\nYour installation is working correctly.")
            print("You can now use QDrantIngest with your own COCO datasets.")
            print("\nNext steps:")
            print("  1. Prepare your COCO dataset")
            print("  2. Run: python main.py --annotations <path> --images <path>")
            print("  3. Check examples/quick_start.py for more details")
            return 0
        else:
            print_error("Some tests failed!")
            print("\nPlease check the error messages above and:")
            print("  1. Ensure all dependencies are installed")
            print("  2. Verify your JINA_API_KEY is valid")
            print("  3. Check your internet connection")
            return 1

    finally:
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        print_success("Cleanup complete")


if __name__ == "__main__":
    sys.exit(main())
