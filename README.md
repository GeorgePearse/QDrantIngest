# QDrantIngest

A powerful tool for converting COCO-formatted datasets into searchable QDrant vector databases with Jina AI embeddings.

## Overview

QDrantIngest automates the entire pipeline from COCO annotations to a queryable vector database:

1. **Parse** COCO annotation files
2. **Crop** object instances from images using bounding boxes or segmentation masks
3. **Embed** cropped objects using Jina AI's state-of-the-art embedding models
4. **Store** vectors in QDrant for fast similarity search

## Features

- **Complete Pipeline**: Single CLI command processes entire datasets
- **Flexible Cropping**: Support for both bounding box and segmentation mask cropping
- **High-Quality Embeddings**: 768-dimensional vectors via Jina AI (jina-embeddings-v2-base-en)
- **Efficient Processing**: Batch processing with progress tracking
- **Rich Metadata**: Stores category names, bounding boxes, areas, and more
- **Local or Remote**: Works with local QDrant instances or remote servers

## Requirements

- Python 3.8+
- COCO-formatted annotation file (JSON)
- Directory containing source images
- Jina AI API key (free tier available)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/QDrantIngest.git
cd QDrantIngest
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### 4. Set Up Jina AI API Key

Get your free API key from [Jina AI](https://jina.ai/):

1. Sign up at https://jina.ai/
2. Navigate to API keys section
3. Create a new API key

Then set it as an environment variable:

```bash
# Linux/MacOS
export JINA_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set JINA_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:JINA_API_KEY="your-api-key-here"
```

For persistence, create a `.env` file (see `.env.example`):

```bash
cp .env.example .env
# Edit .env and add your API key
```

## Quick Start

Process a COCO dataset with default settings:

```bash
python main.py \
  --annotations data/annotations/instances_train2017.json \
  --images data/train2017/ \
  --output ./qdrant_db
```

Use segmentation masks for precise cropping:

```bash
python main.py \
  --annotations data/annotations/instances_train2017.json \
  --images data/train2017/ \
  --output ./qdrant_db \
  --use-segmentation
```

## Usage

### Basic Command

```bash
python main.py --annotations <path-to-coco-json> --images <path-to-images-dir>
```

### All Available Options

```bash
python main.py \
  --annotations PATH              # (Required) Path to COCO annotations JSON file
  --images PATH                   # (Required) Path to directory containing images
  --output PATH                   # Output path for QDrant DB (default: ./qdrant_db)
  --collection NAME               # Collection name (default: coco_objects)
  --batch-size N                  # Processing batch size (default: 32)
  --embedding-model MODEL         # Jina model name (default: jina-embeddings-v2-base-en)
  --vector-size N                 # Embedding dimension (default: 768)
  --use-segmentation              # Use segmentation masks instead of bounding boxes
  --skip-existing                 # Skip if collection already exists
```

### Common Use Cases

#### 1. Process with Bounding Box Cropping (Fast)

```bash
python main.py \
  --annotations annotations.json \
  --images ./images/ \
  --batch-size 64
```

#### 2. Process with Segmentation Masks (Precise)

```bash
python main.py \
  --annotations annotations.json \
  --images ./images/ \
  --use-segmentation \
  --batch-size 32
```

#### 3. Custom Collection Name and Output Path

```bash
python main.py \
  --annotations annotations.json \
  --images ./images/ \
  --output /path/to/db \
  --collection my_custom_collection
```

#### 4. Skip if Already Processed

```bash
python main.py \
  --annotations annotations.json \
  --images ./images/ \
  --skip-existing
```

## Output

After processing, you'll have:

- **QDrant Database**: Stored at `--output` path (default: `./qdrant_db`)
- **Vector Collection**: Named `--collection` (default: `coco_objects`)
- **Metadata Payload**: Each vector includes:
  - `image_id`: Original COCO image ID
  - `file_name`: Source image filename
  - `category_id`: COCO category ID
  - `category_name`: Human-readable category name
  - `bbox`: Bounding box coordinates [x, y, width, height]
  - `segmentation`: Segmentation polygon (if available)
  - `area`: Object area in pixels
  - `iscrowd`: Whether annotation represents a crowd

## Querying the Database

Use the QDrant client to search:

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")

# Search for similar objects
results = client.search(
    collection_name="coco_objects",
    query_vector=your_embedding_vector,
    limit=10
)

# Filter by category
results = client.search(
    collection_name="coco_objects",
    query_vector=your_embedding_vector,
    query_filter={
        "must": [
            {
                "key": "category_name",
                "match": {"value": "person"}
            }
        ]
    },
    limit=10
)
```

## Architecture

```
main.py (CLI Entry Point)
    ├── CocoParser (qdrantingest/coco_parser.py)
    │   └── Parses COCO JSON and validates structure
    │
    ├── ImageProcessor (qdrantingest/image_processor.py)
    │   └── Crops objects using bbox or segmentation
    │
    ├── EmbeddingGenerator (qdrantingest/embedding_generator.py)
    │   └── Generates embeddings via Jina AI API
    │
    └── QdrantUploader (qdrantingest/qdrant_uploader.py)
        └── Manages QDrant collection and uploads vectors
```

## Troubleshooting

### "JINA_API_KEY not found"

Make sure you've set the environment variable:
```bash
export JINA_API_KEY="your-key-here"
```

### "Annotations file not found"

Verify the path is correct and the file exists:
```bash
ls -la path/to/annotations.json
```

### "Images directory not found"

Check that the images directory exists and contains images:
```bash
ls path/to/images/
```

### Slow Processing

- Increase `--batch-size` (try 64 or 128)
- Use `--use-segmentation` only when precision is critical (bbox cropping is faster)
- Check your internet connection (Jina AI requires API calls)

### Memory Issues

- Decrease `--batch-size` (try 16 or 8)
- Process smaller subsets of your dataset
- Ensure sufficient RAM for image processing

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific test files:

```bash
python -m pytest tests/test_coco_parser.py
python -m pytest tests/test_image_processor.py
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Collaboration Notes

This project was developed through collaboration with multiple AI systems:

- **Claude**: Project structure, core implementation, and integration logic
- **Gemini**: Insights on efficient processing of COCO data and image cropping
- **ChatGPT**: QDrant configuration optimization and CLI design
- **Grok**: Performance optimization strategies

## License

MIT