# QDrantIngest

A tool for converting COCO annotations into a searchable QDrant vector database with Jina AI embeddings.

## Project Overview

QDrantIngest allows users to take any COCO-formatted dataset and automatically:
1. Parse the annotations
2. Crop object instances from images
3. Generate embeddings using Jina AI
4. Store these vectors in a QDrant database for efficient similarity search

## Features

- Single entrypoint via `main.py` for easy CLI usage
- Supports standard COCO annotation format
- Automatic object cropping from source images
- High-quality embeddings via Jina AI
- Persistent vector storage with QDrant
- Simple configuration via command line arguments

## Requirements

- Python 3.8+
- COCO-formatted annotation file
- Source images directory

## Implementation Plan

1. **COCO Annotation Parser**
   - Parse COCO JSON files
   - Extract object instances, categories, and image paths
   - Create a unified data structure for processing

2. **Image Processing**
   - Load source images
   - Crop object instances based on bounding boxes or segmentation
   - Apply necessary preprocessing for embedding generation

3. **Embedding Generation with Jina AI**
   - Initialize Jina AI client
   - Generate embeddings for cropped objects
   - Handle batching for efficiency

4. **QDrant Integration**
   - Set up QDrant client
   - Create appropriate collection with optimal parameters
   - Upload embeddings with associated metadata
   - Implement efficient batch uploading

5. **CLI Interface**
   - Create user-friendly command line interface
   - Support configuration options for all components
   - Implement progress tracking
   - Provide verification and validation steps

## Usage

```bash
python main.py --annotations path/to/coco.json --images path/to/images --output path/to/qdrant
```

## Collaboration Notes

This project was developed through collaboration with multiple AI systems:

- **Claude**: Project structure, core implementation, and integration logic
- **Gemini**: Insights on efficient processing of COCO data and image cropping
- **ChatGPT**: QDrant configuration optimization and CLI design
- **Grok**: Performance optimization strategies

## License

MIT