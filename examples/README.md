# QDrantIngest Examples

This directory contains example scripts to help you get started with QDrantIngest.

## Available Examples

### 1. verify_installation.py

**Purpose**: Verify that QDrantIngest is installed correctly and working.

**What it does**:
- Checks all Python dependencies
- Verifies JINA_API_KEY is set
- Creates synthetic test data
- Runs the complete pipeline end-to-end
- Validates all components work correctly

**Usage**:
```bash
# Make sure JINA_API_KEY is set first
export JINA_API_KEY="your-api-key-here"

# Run verification
python examples/verify_installation.py
```

**When to use**:
- After first installation
- After updating dependencies
- To troubleshoot issues
- To verify API key is working

**Expected output**:
```
============================================================
          QDrantIngest Installation Verification
============================================================

============================================================
                   Checking Dependencies
============================================================

✓ numpy installed
✓ pillow installed
✓ qdrant-client installed
✓ tqdm installed
✓ jinaai installed

============================================================
                    Checking API Key
============================================================

✓ JINA_API_KEY is set: jina_ab...xyz

... (continues with test results)

✓ All tests passed!
```

---

### 2. quick_start.py

**Purpose**: Demonstrate how to use QDrantIngest with a real COCO dataset.

**What it does**:
- Shows the complete workflow step-by-step
- Processes a small batch (first 10 annotations) from your dataset
- Creates a QDrant vector database
- Demonstrates similarity search
- Includes detailed comments explaining each step

**Usage**:
```bash
# Edit the script to point to your COCO dataset
# Update these lines in quick_start.py:
#   annotations_path = "data/annotations/instances_val2017.json"
#   images_dir = "data/val2017/"

# Then run
python examples/quick_start.py
```

**Prerequisites**:
- COCO dataset downloaded (or your own COCO-formatted dataset)
- JINA_API_KEY environment variable set

**When to use**:
- Learning how QDrantIngest works
- Understanding the pipeline flow
- Testing with a small sample before processing full dataset
- As a template for custom scripts

**What you'll learn**:
- How to use each QDrantIngest module
- How to process COCO annotations
- How to crop objects from images
- How to generate embeddings
- How to upload to QDrant
- How to query the database

---

## Quick Reference

### Environment Setup

Create a `.env` file in the project root:
```bash
cp ../.env.example ../.env
# Edit .env and add your JINA_API_KEY
```

Or set the environment variable directly:
```bash
export JINA_API_KEY="your-api-key-here"
```

### Getting a COCO Dataset

**Option 1: Official COCO Dataset**
```bash
# Download COCO 2017 validation set (small, ~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip
```

**Option 2: Use Your Own Dataset**

Make sure it follows COCO format:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1000
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

### Common Workflows

**1. First Time Setup**
```bash
# Install and verify
pip install -r requirements.txt
export JINA_API_KEY="your-key"
python examples/verify_installation.py
```

**2. Process Small Sample**
```bash
# Update paths in quick_start.py
python examples/quick_start.py
```

**3. Process Full Dataset**
```bash
# Use main.py for production workloads
python main.py \
  --annotations data/annotations/instances_val2017.json \
  --images data/val2017/ \
  --use-segmentation \
  --batch-size 32
```

## Troubleshooting

### "JINA_API_KEY not found"
- Make sure you've set the environment variable
- Check with: `echo $JINA_API_KEY`
- Set it: `export JINA_API_KEY="your-key"`

### "Annotations file not found"
- Check the path in the script
- Verify file exists: `ls -la path/to/annotations.json`
- Update the path in the script to match your setup

### API Errors
- Verify your API key is valid
- Check internet connection
- Free tier may have rate limits

### Memory Issues
- Reduce batch size in the scripts
- Process smaller datasets first
- Monitor RAM usage with `htop` or Activity Monitor

## Need Help?

- Check the main README: `../README.md`
- Review test files: `../tests/`
- Read module documentation in: `../qdrantingest/`

## Contributing

Have a useful example script? Contributions welcome!
- Fork the repo
- Add your example with documentation
- Submit a pull request
