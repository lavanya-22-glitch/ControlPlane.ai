from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ParameterRule(BaseModel):
    field: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


class ToolGuard(BaseModel):
    parameter_rules: List[ParameterRule] = Field(default_factory=list)
    action_on_violation: str = "block"  # "block" | "rewrite"


class ContextCappingConfig(BaseModel):
    enabled: bool = False
    max_context_length_chars: int = 4000


class SemanticCacheConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = 0.95


class PreExecutionConfig(BaseModel):
    block_prompt_injection: bool = True
    injection_threshold: float = 0.60
    pii_action: str = "mask"  # "mask" | "block" | "none"
    masked_entities: List[str] = Field(
        default_factory=lambda: ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "SSN"]
    )
    context_capping: Optional[ContextCappingConfig] = None
    semantic_cache: Optional[SemanticCacheConfig] = None


class HallucinationConfig(BaseModel):
    enabled: bool = True
    min_grounding_score: float = 0.70
    action_on_low_grounding: str = "rewrite"  # "rewrite" | "block"
    fallback_message: str = (
        "The requested details could not be verified against source documents."
    )
    # NLI backend selection: "auto" | "cross_encoder" | "transformers" | "onnx" | "heuristic"
    # "auto" picks the best available: ONNX > CrossEncoder > Transformers > Heuristic.
    nli_backend: str = "auto"
    # HuggingFace model name used by CrossEncoderNLIBackend / TransformersNLIBackend
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    # Optional path to a pre-exported ONNX model file (enables ONNXNLIBackend)
    onnx_model_path: Optional[str] = None
    # Per-claim grounding score below which the pipeline short-circuits immediately.
    # Set lower (e.g. 0.05) for lenient mode; higher (e.g. 0.30) for strict mode.
    early_exit_threshold: float = 0.20
    # ---- Context Reranker (pre-NLI relevance filter) -----------------------
    # When enabled, filters retrieved context chunks by relevance to the user query
    # using a cross-encoder reranker before running NLI entailment scoring.
    context_reranking_enabled: bool = False
    context_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Chunks with reranker score below this threshold are dropped before NLI.
    context_relevance_threshold: float = 0.30
    # Keep at most this many chunks after reranking (None = keep all above threshold).
    context_reranker_top_k: Optional[int] = None
    # ---- Citation Similarity Check -----------------------------------------
    # When enabled, parses inline citation markers ([1], [Doc A], etc.) from
    # the completion and measures how similar the cited claim is to retrieved chunks.
    citation_check_enabled: bool = False
    # similarity method: "auto" | "ngram" | "embedding"
    # "auto" = embedding if sentence-transformers installed, else ngram.
    citation_similarity_method: str = "auto"
    # Per-citation short-circuit threshold: if any single citation's similarity
    # to the best matching context chunk falls below this, abort immediately.
    citation_fail_threshold: float = 0.35
    # Aggregate threshold: if the average citation similarity is below this,
    # trigger REWRITE/BLOCK even if no single citation short-circuited.
    citation_avg_threshold: float = 0.50



class ConfidenceMismatchConfig(BaseModel):
    """Confidence mismatch detection config for agent tool calls."""
    enabled: bool = True
    # Average log-probability of tool-call tokens below this → flag mismatch
    avg_logprob_threshold: float = -2.0
    # Single token log-probability below this → flag mismatch (worst-case sentinel)
    min_logprob_threshold: float = -4.0
    # Per-token threshold for counting "low confidence" tokens
    per_token_threshold: float = -2.5
    # If more than this fraction of tokens are low-confidence, flag mismatch
    max_low_conf_pct: float = 0.30


class ReasoningSimilarityConfig(BaseModel):
    """Reasoning ↔ tool-output alignment check config."""
    enabled: bool = True
    # Similarity method: "auto" | "ngram" | "embedding"
    # "auto" = embedding if sentence-transformers installed, else ngram.
    similarity_method: str = "auto"
    # Similarity score below this → AGENT_REASONING_MISMATCH violation
    similarity_fail_threshold: float = 0.15


class ChatbotHallucinationConfig(BaseModel):
    """Configuration for LLM-as-a-judge hallucination detection via OpenRouter."""
    enabled: bool = False
    ollama_base_url: str = "https://openrouter.ai/api/v1/chat/completions"  # kept for compat, not used directly
    ollama_model_name: str = "google/gemma-3-27b-it:free"
    # Action if the judge determines it's a hallucination
    action_on_violation: str = "rewrite"
    fallback_message: str = "I'm sorry, I cannot verify the accuracy of this response."

class ToolDeduplicationConfig(BaseModel):
    enabled: bool = False
    max_duplicate_calls: int = 1

class AgentConfig(BaseModel):
    """Umbrella agent-mode safety config attached to PolicyDefinition."""
    confidence_mismatch: ConfidenceMismatchConfig = ConfidenceMismatchConfig()
    reasoning_similarity: ReasoningSimilarityConfig = ReasoningSimilarityConfig()
    tool_deduplication: Optional[ToolDeduplicationConfig] = None


class PostExecutionConfig(BaseModel):
    toxicity_threshold: Optional[float] = 0.70
    bias_subspace_threshold: Optional[float] = 0.65
    
    # Advanced Bias Settings
    bias_projection_enabled: bool = False
    counterfactual_check_enabled: bool = False
    counterfactual_ollama_base_url: str = "https://openrouter.ai/api/v1/chat/completions"  # kept for compat
    counterfactual_ollama_model_name: str = "google/gemma-3-27b-it:free"

    hallucination_engine: Optional[HallucinationConfig] = None
    chatbot_hallucination: Optional[ChatbotHallucinationConfig] = None
    action_on_violation: str = "rewrite"
    fallback_message: Optional[str] = (
        "I am unable to answer this query due to safety policy constraints."
    )


class PolicyDefinition(BaseModel):
    version: str = "1.0.0"
    mode: str = "chatbot"  # "chatbot" | "rag" | "agent"
    fail_mode: str = "fail_closed"  # "fail_closed" | "fail_open"
    streaming_mode: str = "buffered"  # "buffered" | "pass_through" | "deny"
    pre_execution: PreExecutionConfig = Field(default_factory=PreExecutionConfig)
    post_execution: Optional[PostExecutionConfig] = Field(
        default_factory=PostExecutionConfig
    )
    tool_guards: Optional[Dict[str, ToolGuard]] = None
    agent_config: Optional[AgentConfig] = None  # None = agent features disabled
    policy_hash: Optional[str] = None


class PoliciesFile(BaseModel):
    policies: Dict[str, PolicyDefinition]
