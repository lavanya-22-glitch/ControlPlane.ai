import logging
import hashlib
import json
import threading
from typing import Optional, Dict, Any, List

logger = logging.getLogger("controlplane.cache.semantic")

class SemanticCache:
    """
    In-memory semantic cache designed to short-circuit identical prompts and tool calls.
    In a production deployment, this maps to a Redis instance.
    """
    
    def __init__(self):
        # Thread-safe in-memory cache map
        # Key: md5_hash(prompt + context), Value: JSON upstream response payload
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        logger.info("Initialized In-Memory Semantic Cache.")
        
    def _generate_key(self, messages: List[Dict[str, Any]], retrieved_context: Optional[List[str]] = None) -> str:
        """Deterministically hashes the core inputs (messages + RAG context)."""
        raw = ""
        # 1. Flatten messages
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            # Simple stringification for hash stability
            raw += f"[{role}]:{content}"
            if tool_calls:
                raw += json.dumps(tool_calls, sort_keys=True)
                
        # 2. Flatten retrieved context
        if retrieved_context:
            for ctx in retrieved_context:
                raw += f"[ctx]:{ctx}"
                
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_cached_response(self, messages: List[Dict[str, Any]], retrieved_context: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Attempts to find a cached LLM response for the exact input prompt.
        """
        cache_key = self._generate_key(messages, retrieved_context)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                logger.info(f"Semantic Cache HIT. Serving response from memory. Key: {cache_key[:8]}...")
            return cached
            
    def set_cached_response(self, messages: List[Dict[str, Any]], retrieved_context: Optional[List[str]], response_payload: Dict[str, Any]) -> None:
        """
        Stores a known-safe upstream response in the cache.
        """
        cache_key = self._generate_key(messages, retrieved_context)
        with self._lock:
            self._cache[cache_key] = response_payload
            logger.debug(f"Semantic Cache SET. Wrote response to memory. Key: {cache_key[:8]}...")

# Singleton instance
semantic_cache_engine = SemanticCache()
