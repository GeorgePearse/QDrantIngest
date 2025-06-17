"""
Uploader for storing embeddings in QDrant vector database.
"""

from typing import Dict, List, Optional, Union, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantUploader:
    """
    Uploader for storing embeddings in QDrant vector database.
    
    This class handles connecting to QDrant and uploading vectors with
    associated metadata.
    """
    
    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance: str = "Cosine"
    ):
        """
        Initialize the QDrant uploader.
        
        Args:
            collection_name: Name of the collection to create/use
            vector_size: Dimensionality of the embedding vectors
            path: Path to the local QDrant database (for file-based storage)
            url: URL of the QDrant server (for remote storage)
            api_key: API key for QDrant server (if using cloud service)
            distance: Distance function to use (Cosine, Euclid, or Dot)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        
        # Initialize client (either local or remote)
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key
            )
        else:
            self.client = QdrantClient(
                path=path
            )
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """
        Create a QDrant collection if it doesn't exist.
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create a new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            
            # Create indexes for faster filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="image_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            print(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            True if the collection exists, False otherwise
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        return self.collection_name in collection_names
    
    def upload_batch(self, objects: List[Dict[str, Any]]):
        """
        Upload a batch of objects to QDrant.
        
        Args:
            objects: List of dictionaries with the following keys:
                - id: Unique identifier for the object
                - vector: Embedding vector
                - payload: Dictionary of metadata to store with the vector
        """
        if not objects:
            return
        
        # Prepare points for upload
        points = []
        for obj in objects:
            points.append(
                models.PointStruct(
                    id=obj['id'],
                    vector=obj['vector'],
                    payload=obj['payload']
                )
            )
        
        # Upload points in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Embedding vector to search for
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results with distances and payloads
        """
        # Create filter if provided
        filter_obj = None
        if filter_conditions:
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )
        
        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_obj
        )
        
        # Format results
        results = []
        for scored_point in search_result:
            results.append({
                'id': scored_point.id,
                'score': scored_point.score,
                'payload': scored_point.payload
            })
        
        return results