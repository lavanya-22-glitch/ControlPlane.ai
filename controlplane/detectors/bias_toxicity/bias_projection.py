"""
Bias Space Projections (Word/Sentence Embeddings).

Projects completion text embeddings onto predefined demographic axes (e.g., Gender, Race)
to detect latent bias or skew that simple regex might miss.
Uses sentence-transformers if available.
"""
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("controlplane.detectors.bias.projection")

class BiasProjectionScorer:
    """Computes the cosine similarity projection of text onto demographic axes."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._axes = {}
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            
            # Define demographic anchor words to create the bias axes
            gender_axis = self._compute_axis(["he", "man", "male"], ["she", "woman", "female"])
            race_axis = self._compute_axis(["white", "caucasian"], ["black", "african american", "hispanic", "asian"])
            
            if gender_axis is not None:
                self._axes["Gender"] = gender_axis
            if race_axis is not None:
                self._axes["Race"] = race_axis
                
            logger.info("BiasProjectionScorer loaded successfully.")
        except Exception as e:
            logger.warning("BiasProjectionScorer failed to load (sentence-transformers missing?): %s", e)

    def _compute_axis(self, anchors_a: list, anchors_b: list) -> Optional[np.ndarray]:
        """Compute the vector difference between two sets of anchors to define a subspace axis."""
        if not self._model:
            return None
        emb_a = self._model.encode(anchors_a, normalize_embeddings=True).mean(axis=0)
        emb_b = self._model.encode(anchors_b, normalize_embeddings=True).mean(axis=0)
        
        # The axis is the vector difference
        axis = emb_a - emb_b
        norm = np.linalg.norm(axis)
        if norm > 0:
            return axis / norm
        return None

    def is_available(self) -> bool:
        return self._model is not None and len(self._axes) > 0

    def score(self, text: str) -> Tuple[float, str]:
        """
        Returns (max_skew, reason).
        max_skew is the absolute cosine similarity between the text and any bias axis.
        """
        if not self.is_available() or not text.strip():
            return 0.0, ""
            
        # Get embedding of the text
        text_emb = self._model.encode([text], normalize_embeddings=True)[0]
        
        max_skew = 0.0
        skewed_axis = ""
        
        for axis_name, axis_vector in self._axes.items():
            # Compute projection scalar (cosine similarity)
            projection = float(np.dot(text_emb, axis_vector))
            skew = abs(projection)
            
            if skew > max_skew:
                max_skew = skew
                skewed_axis = axis_name
                
        reason = f"Latent {skewed_axis} bias projection detected." if max_skew > 0 else ""
        return max_skew, reason
