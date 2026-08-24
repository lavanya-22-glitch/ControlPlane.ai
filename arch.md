# ControlPlane.ai Architecture & Subsystem Design

```
┌──────────────┐         ┌──────────────────────────────────────────────────────────────┐         ┌───────────────────┐
│              │ Request │  controlplane.proxy (FastAPI Router + SSE Stream Handler)    │ Request │                   │
│  Client App  ├────────►│     └─ Resolves Policy from Registry by App-ID & Hash        ├────────►│   Upstream LLM    │
│  (Chat/RAG/  │         │                                                              │         │ (OpenAI / Local)  │
│    Agent)    │◄────────┤  controlplane.detectors (Pre-Execution Pipeline)              │◄────────┤                   │
│              │ Response│     ├─ injection: Heuristic + Semantic Classifier            │ Response│                   │
└──────────────┘         │     └─ pii: Dual-Tier Scanner (Regex Fast-Path + Presidio)   │         └───────────────────┘
                         │                                                              │
                         │  controlplane.proxy.dispatcher (httpx async pool)            │
                         │                                                              │
                         │  controlplane.detectors (Post-Execution Pipeline)             │
                         │     ├─ tool_safety: JSON Schema & Bounds Validator           │
                         │     ├─ hallucination: Batched ONNX INT8 NLI Grounding Engine │
                         │     ├─ bias_toxicity: Toxicity & Subspace Projection Scorer  │
                         │     └─ pii.detokenizer: In-Memory Vault Egress Lookup        │
                         │                                                              │
                         │  controlplane.pdp (Policy Decision Point: ALLOW/TRANSFORM...)│
                         │                                                              │
                         │  controlplane.telemetry (Async SQLite WAL + Trace Replay)    │
                         └──────────────────────────────────────────────────────────────┘
```

## 1. Modular Execution Pipeline & State Machine

```
                        [Incoming HTTP / SSE Request]
                                      │
                                      ▼
                      [Route & Policy Registry Lookup]
                                      │
                                      ▼
                      [Pre-Execution Guard Pipeline]
                       ├── Injection Check ──(Fail)──► [ACTION: BLOCK (422)]
                       └── Dual-Tier PII Vault Tokenize
                                      │
                                      ▼
                         [Dispatch to Target LLM API]
                                      │
                                      ▼
                         [LLM Output Branch Decision]
                        /                            \
              (Is Tool Call?)                  (Is Text Completion?)
                    /                                    \
                   ▼                                      ▼
        [Tool Guard Pipeline]                 [Post-Execution Content Guard]
         ├── JSON Schema Validation            ├── De-tokenize PII Placeholders
         └── Parameter Range Bounds            ├── Batched NLI Grounding (RAG Context)
                   │                           └── Fast Bias / Toxicity Projection
                   │                                      │
                   └──────────────────┬───────────────────┘
                                      ▼
                         [Policy Decision Point (PDP)]
                         ├── ALLOW     ──► Return Original Payload
                         ├── TRANSFORM ──► Return Masked / Sanitized Payload
                         ├── REWRITE   ──► Return Predefined Safe Fallback
                         └── BLOCK     ──► Return HTTP 422 Error
                                      │
                                      ▼
                       [Async SQLite Audit Sink (WAL)]
```

---

## Module 1: Ingress & Policy Configuration Engine (`controlplane.policy`)

### 1. Responsibilities
- Parse incoming HTTP request headers (`X-ControlPlane-App-ID`).
- Maintain a thread-safe, hot-reloading in-memory registry of compiled policy rules.
- Compute and attach a deterministic `policy_version` and `policy_hash` (SHA-256) to every evaluation context.

### 2. Configuration Schema (`policies.yaml`)

```yaml
policies:
  customer-support:
    version: "1.0.0"
    mode: "chatbot"
    fail_mode: "fail_closed"
    streaming_mode: "buffered" # "buffered" | "pass_through" | "deny"
    pre_execution:
      block_prompt_injection: true
      pii_action: "mask" # "mask" | "block" | "none"
      masked_entities: ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "PERSON"]
    post_execution:
      toxicity_threshold: 0.70
      bias_subspace_threshold: 0.65
      action_on_violation: "rewrite"
      fallback_message: "I am unable to answer this query. Please reach out to customer support directly."

  internal-kb-rag:
    version: "1.0.0"
    mode: "rag"
    fail_mode: "fail_open"
    streaming_mode: "buffered"
    pre_execution:
      pii_action: "mask"
      masked_entities: ["ALL"]
    post_execution:
      hallucination_engine:
        enabled: true
        min_grounding_score: 0.70
        action_on_low_grounding: "rewrite"
        fallback_message: "The requested details could not be verified from the provided source documents."

  action-agent:
    version: "1.0.0"
    mode: "agent"
    fail_mode: "fail_closed"
    streaming_mode: "pass_through"
    tool_guards:
      execute_command:
        parameter_rules:
          - field: "command_type"
            type: "string"
            allowed_values: ["READ", "QUERY", "UPDATE", "NOTIFY"]
          - field: "timeout_seconds"
            type: "int"
            max_value: 300
            min_value: 1
          - field: "batch_size"
            type: "int"
            max_value: 1000
            min_value: 1
        action_on_violation: "block"
```

---

## Module 2: Pre-Execution Guard (`controlplane.detectors.pii` & `injection`)

### 1. Dual-Tier Reversible PII Masking Vault
- **Tier 1 (Fast-Path Regex, <5ms):** Compiled regex patterns for structural, high-cardinality entities (Email, Credit Card, SSN, Phone Numbers, IP Addresses).
- **Tier 2 (Presidio Contextual Fallback):** Invoked for NLP-dependent entities (Person names, Locations) when specified by policy.
- **Vault Tokenization:** Replaces matched values with session-scoped deterministic tokens (`Alice Smith` $\to$ `<PERSON_1>`, `4111-2222-3333-4444` $\to$ `<CREDIT_CARD_1>`) stored in an in-memory session vault keyed by `trace_id`.

### 2. Prompt Injection Scanner
- **Heuristic Engine:** Fast regex scanning for delimiter hijacking, system prompt override phrases, and token smuggling.
- **Semantic Evaluator:** Lightweight scoring for instruction boundary breaks.
- If score $> \theta_{\text{injection}}$, PDP triggers `BLOCK` (HTTP 422).

---

## Module 3: Upstream Dispatcher & Streaming Engine (`controlplane.proxy`)

### 1. Responsibilities
- Non-blocking connection pooling via `httpx.AsyncClient`.
- Forward sanitized JSON payload to target LLM API (OpenAI or mock upstream).
- **Buffered SSE Stream Handler:** When `stream=true` is requested on high-risk or RAG routes, buffer SSE chunks into sentence units, execute post-guards, and flush verified chunks, maintaining wire compatibility.

---

## Module 4: Post-Execution Content & Hallucination Guard (`controlplane.detectors`)

### 1. Batched NLI RAG Grounding Engine (`hallucination`)
1. **Claim Extraction:** Parse completion into discrete claims/sentences.
2. **Context Alignment:** Pair each claim with relevant chunks from `retrieved_context`.
3. **Batched ONNX Inference:** Feed all `(Context, Claim)` pairs as a single tensor batch into quantized INT8 `cross-encoder/nli-deberta-v3-small`.
4. **Scoring:** Compute Entailment vs. Contradiction probability:
   $$S = P(\text{Entailment}) - P(\text{Contradiction})$$
5. Aggregate claim scores. If $\text{score} < \text{min\_grounding\_score}$, trigger PDP `REWRITE` or `BLOCK`.

### 2. Fast Bias & Toxicity Filter (`bias_toxicity`)
- Toxicity classifier & demographic subspace cosine similarity evaluation.

### 3. PII De-tokenization (`pii.detokenizer`)
- Egress scan restores original sensitive values from the session vault (`<PERSON_1>` $\to$ `Alice Smith`).

---

## Module 5: Tool Call Safety Matrix (`controlplane.detectors.tool_safety`)

- Inspects `tool_calls` payload before client receipt.
- Verifies JSON schema types (`float`, `int`, `string`, `bool`), numeric boundaries (`min_value`, `max_value`), and categorical allowlists (`allowed_values`).
- Rejects malformed or out-of-bounds agent actions with HTTP 422.

---

## Module 6: Policy Decision Point (`controlplane.pdp`)

Deterministic mapping matrix:

| **Rule Trigger** | **Action** | **HTTP Status** | **Returned Payload Body** |
| :--- | :--- | :--- | :--- |
| **All Checks Clean** | `ALLOW` | 200 OK | Original Model Response (PII detokenized). |
| **PII Detected (Mask Mode)** | `TRANSFORM` | 200 OK | Masked upstream, detokenized on egress. |
| **Hallucination / Low Grounding** | `REWRITE` | 200 OK | Safe fallback completion string. |
| **Injection / Tool Out-of-Bounds** | `BLOCK` | 422 Unprocessable | Structured policy violation JSON. |
| **Detector Timeout/Exception** | `FAIL_OPEN` / `FAIL_CLOSED` | 200 OK / 500 Error | Determined by route `fail_mode`. |

---

## Module 7: Local Audit Logger & Trace Replay (`controlplane.telemetry`)

### SQLite Schema (`audit_logs.db`)

```sql
CREATE TABLE IF NOT EXISTS audit_traces (
    trace_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    app_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    use_case_mode TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms REAL,
    pre_execution_latency_ms REAL,
    upstream_latency_ms REAL,
    post_execution_latency_ms REAL,
    pii_entities_found TEXT, -- JSON Array: ["EMAIL_ADDRESS", "PERSON"]
    injection_score REAL,
    grounding_score REAL,
    bias_score REAL,
    pdp_action TEXT NOT NULL, -- 'ALLOW', 'TRANSFORM', 'REWRITE', 'BLOCK'
    violation_reason TEXT,
    raw_request_payload TEXT, -- Stored for deterministic trace replay/simulation
    final_response_payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_app_id ON audit_traces (app_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_traces (timestamp);
```

---

## Wire Format Contracts

### Error Response (HTTP 422 on Violation)
```json
{
  "error": {
    "type": "controlplane_policy_violation",
    "code": "PROMPT_INJECTION_DETECTED",
    "message": "Transaction rejected by policy 'customer-support'.",
    "trace_id": "cp_trace_8f9a2b1c",
    "policy_version": "1.0.0",
    "pdp_action": "BLOCK"
  }
}
```