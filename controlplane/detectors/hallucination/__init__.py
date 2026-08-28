from controlplane.detectors.hallucination.nli_verifier import NLIRAGGroundingGuard
from controlplane.detectors.hallucination.nli_backend import (
    NLIBackend,
    NLIResult,
    HeuristicNLIBackend,
    TransformersNLIBackend,
    ONNXNLIBackend,
    auto_select_backend,
)
from controlplane.detectors.hallucination.cross_encoder import CrossEncoderNLIBackend
from controlplane.detectors.hallucination.context_reranker import (
    ContextReranker,
    RankedChunk,
    get_context_reranker,
)
from controlplane.detectors.hallucination.citation_parser import (
    Citation,
    CitationParser,
)
from controlplane.detectors.hallucination.citation_similarity import (
    CitationGroundingGuard,
    CitationSimilarityResult,
    CitationSimilarityScorer,
    NgramSimilarityScorer,
    EmbeddingSimilarityScorer,
)
from controlplane.detectors.hallucination.chatbot_judge import ChatbotHallucinationGuard
from controlplane.detectors.hallucination.claim_splitter import split_claims
from controlplane.detectors.hallucination.context_aligner import align_claims_to_context
from controlplane.detectors.hallucination.types import ClaimContextPair

__all__ = [
    # Core guards
    "NLIRAGGroundingGuard",
    "CitationGroundingGuard",
    "ChatbotHallucinationGuard",
    # NLI backends
    "NLIBackend",
    "NLIResult",
    "HeuristicNLIBackend",
    "CrossEncoderNLIBackend",
    "TransformersNLIBackend",
    "ONNXNLIBackend",
    "auto_select_backend",
    # Context reranker
    "ContextReranker",
    "RankedChunk",
    "get_context_reranker",
    # Citation modules
    "Citation",
    "CitationParser",
    "CitationSimilarityResult",
    "CitationSimilarityScorer",
    "NgramSimilarityScorer",
    "EmbeddingSimilarityScorer",
    # Utilities
    "split_claims",
    "align_claims_to_context",
    "ClaimContextPair",
]
