"""
Embedding generator using Jina AI.
"""

import os
from typing import List, Optional, Union, Any

import numpy as np
from PIL import Image

# Try to import jinaai
try:
    import jinaai
except ImportError:
    jinaai = None


class EmbeddingGenerator:
    """
    Generator for creating embeddings from images using Jina AI.
    
    This class handles connecting to the Jina AI API and generating
    embeddings for images.
    """
    
    def __init__(
        self, 
        model_name: str = "jina-embeddings-v2-base-en",
        vector_size: int = 768,
        api_key: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the Jina AI model to use
            vector_size: Dimensionality of the output embeddings
            api_key: Jina AI API key (if None, will try to load from JINA_API_KEY env var)
        """
        self.model_name = model_name
        self.vector_size = vector_size
        
        # Get API key from env var if not provided
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        
        if not self.api_key:
            print("Warning: No Jina AI API key provided. Please set JINA_API_KEY environment variable.")
        
        # Check if jinaai is installed
        if jinaai is None:
            raise ImportError(
                "jinaai package is not installed. "
                "Please install it using: pip install jinaai"
            )
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize the Jina AI client.
        
        Raises:
            ValueError: If API key is not available
        """
        if not self.api_key:
            raise ValueError(
                "Jina AI API key is required. "
                "Please provide it in the constructor or set JINA_API_KEY environment variable."
            )
        
        # Set the API key for the client
        jinaai.api_key = self.api_key
    
    def generate_embeddings(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Generate embeddings for a list of images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of embedding vectors
        """
        if not images:
            return []
        
        # Convert PIL images to bytes
        image_bytes = []
        for img in images:
            # Convert image to bytes
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_bytes.append(img_byte_arr.getvalue())
        
        try:
            # Generate embeddings using Jina AI
            results = jinaai.embed(
                model=self.model_name,
                inputs=image_bytes,
                input_type="image",
                task_type="retrieval",
            ).embeddings
            
            # Check if the dimensionality matches
            if len(results) > 0 and len(results[0]) != self.vector_size:
                print(f"Warning: Expected embedding dimension {self.vector_size}, "
                      f"but got {len(results[0])}. Continuing anyway.")
            
            return results
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.vector_size for _ in range(len(images))]
    
    def generate_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Embedding vector
        """
        results = self.generate_embeddings([image])
        return results[0] if results else [0.0] * self.vector_size