"""Vector store client for Endee vector database."""

import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import msgpack
from config import Config

logger = logging.getLogger(__name__)

class EndeeVectorStore:
    """Client for interacting with Endee vector database."""
    
    def __init__(self, base_url: str = None, auth_token: str = None):
        self.base_url = base_url or Config.ENDEE_BASE_URL
        self.auth_token = auth_token or Config.ENDEE_AUTH_TOKEN
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Endee API."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=data)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            
            # Handle different response types
            if not response.content:
                return {}
            
            content_type = response.headers.get('content-type', '')
            if 'application/msgpack' in content_type:
                # Unpack MessagePack response
                try:
                    return msgpack.unpackb(response.content, raw=False)
                except:
                    return {}
            else:
                # JSON response
                try:
                    return response.json()
                except:
                    return {"text": response.text}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Failed to connect to Endee: {e}")
    
    def health_check(self) -> bool:
        """Check if Endee is running and accessible."""
        try:
            result = self._make_request("GET", "health")
            return result.get("status") == "ok"
        except:
            return False
    
    def create_collection(self, collection_name: str, dimension: int, 
                         metric: str = "cosine") -> bool:
        """Create a new collection/index in Endee."""
        data = {
            "index_name": collection_name,
            "dim": dimension,
            "space_type": metric,
            "M": 16,
            "ef_con": 200,
            "precision": "float32"
        }
        try:
            self._make_request("POST", "index/create", data)
            logger.info(f"Index '{collection_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in Endee."""
        try:
            result = self._make_request("GET", "index/list")
            indexes = result.get("indexes", [])
            return [idx.get("name", "") for idx in indexes]
        except:
            return []
    
    def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                   metadata: List[Dict] = None, ids: List[str] = None) -> bool:
        """Add vectors to a collection."""
        if metadata is None:
            metadata = [{}] * len(vectors)
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]
            
        # Endee expects array of vector objects
        vector_objects = [
            {
                "id": doc_id,
                "vector": vector
                # Note: Endee doesn't support metadata in vectors directly
                # Metadata would need to be stored separately
            }
            for doc_id, vector in zip(ids, vectors)
        ]
        
        try:
            self._make_request("POST", f"index/{collection_name}/vector/insert", vector_objects)
            logger.info(f"Added {len(vectors)} vectors to index '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for similar vectors in a collection."""
        data = {
            "vector": query_vector,
            "k": top_k,
            "include_vectors": False
        }
        
        if filters:
            data["filter"] = str(filters)  # Endee expects filter as string
            
        try:
            # Endee returns MessagePack, but we'll handle that in the client
            result = self._make_request("POST", f"index/{collection_name}/search", data)
            
            # Handle the response - Endee may return different format
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "results" in result:
                return result["results"]
            else:
                return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self._make_request("DELETE", f"index/{collection_name}/delete")
            logger.info(f"Index '{collection_name}' deleted")
            return True
        except:
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection."""
        try:
            return self._make_request("GET", f"index/{collection_name}/info")
        except:
            return {}