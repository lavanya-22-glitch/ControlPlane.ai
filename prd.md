
**1. Executive Summary & MVP Scope**
`ControlPlane.ai` is a lightweight, configurable inline reverse proxy designed to inspect, validate, sanitize, and guard LLM requests and completions in real time.
For the MVP, all distributed external infrastructure (Kafka, ClickHouse, distributed vector stores) is replaced with **in-memory and local file/SQLite primitives** to maximize simplicity, reproducibility, and local runnability while preserving strict wire-level compatibility with standard LLM endpoints.

**2. Core Functional Requirements (FR)**

**FR-1: Drop-In API Compatibility & Streaming Management**
- The proxy must expose standard OpenAI-compatible endpoints:
  - `POST /v1/chat/completions` (JSON and Server-Sent Events `stream=true`).
- **Streaming Policy:** Supports **buffered streaming** (buffers tokens into sentences/tool-call blocks before running post-execution guards and flushing) or enforces non-streaming based on route risk classification (`fail_closed` high-risk routes).
- Routing context via HTTP header:
  - `X-ControlPlane-App-ID`: Unique string identifying the active policy (e.g., `customer-support`, `internal-kb-rag`, `financial-action-agent`).

**FR-2: Request & Response Safety Pipelines**
- **Pre-Execution (Prompt-level):**
  - Prompt injection detection via compiled heuristics + semantic scoring.
  - **Dual-Tier Reversible PII Masking:** Fast-path compiled regex (<5ms) for structural entities (Email, Credit Card, SSN, Phone) + Presidio contextual fallback (Person, Location) mapping to session vault tokens (`Alice Smith` $\to$ `<PERSON_1>`).
- **Post-Execution (Completion-level):**
  - **Batched NLI Grounding/Hallucination Engine:** Multi-claim sentence splitting evaluated in a single batched ONNX INT8 pass against supplied context chunks.
  - Fast lexical and embedding-based bias/toxicity scanning.
  - Reversible PII de-tokenization (`<PERSON_1>` $\to$ `Alice Smith`).
- **Agent Tool-Call Interception:**
  - Strict JSON Schema validation, argument typing, and numerical/categorical bounds checking.

**FR-3: Policy Decision Point (PDP) Actions**
Every request evaluation yields one of four deterministic actions:
1. **`ALLOW`**: Forward payload unmodified.
2. **`TRANSFORM`**: Reversibly mask PII or strip specific disallowed tags.
3. **`REWRITE`**: Replace model completion with a predefined safe fallback message.
4. **`BLOCK`**: Return HTTP `422 Unprocessable Content` with a structured JSON error body detailing the violated policy.

**FR-4: Local Auditing, Traceability & Replayability**
- Every transaction generates a structured audit record persisted to local SQLite (WAL mode):
  - Captures trace ID, policy version/hash, latency breakdown, detector scores, and final PDP action.
  - Supports deterministic trace replay and offline policy simulation.

**3. Non-Functional Requirements (NFR)**
- **Zero External Dependencies (MVP):** Deployable via Docker / Docker Compose with embedded local ONNX runtime, Presidio/regex, and SQLite.
- **Overhead Budget:** Proxy overhead $\le 50\text{ ms}$ for heuristic/regex checks; $\le 250\text{ ms}$ for batched ONNX NLI cross-encoder verification.
- **Fail-Safe Behavior:** Configurable `fail_mode` (`fail_open` vs. `fail_closed`) per route if an internal detector times out.