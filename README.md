# ControlPlane.ai

> **High-Performance Inline AI Safety, Governance & Observability Reverse Proxy**

`ControlPlane.ai` is a lightweight, drop-in reverse proxy (wire-compatible with OpenAI `/v1/chat/completions`) that inspects, validates, sanitizes, and guards LLM requests and completions in real time with zero cloud lock-in.

---

## 🚀 What's New
- **Agentic Workflows Demo (`agent_demo.py`)**: A dedicated testing script that pits an unguarded Direct LLM against the ControlPlane Proxy across 5 critical attack vectors (Tool Schema Violations, Prompt Injections, Command Injections, Tool Loops, and Safe Baselines).
- **Dual External API Keys in Dashboard**: The Interactive Playground now securely accepts external Gemini (Upstream LLM) and OpenRouter (Cloud Judge) API keys, persisting them locally to let anyone run real API requests through the proxy safely and for free.
- **Cloud LLM Judges via OpenRouter**: Seamlessly shifted heavy detectors (like Counterfactual Bias and Chatbot Hallucination) to run on cloud `google/gemma-3-27b-it:free` via OpenRouter. Get enterprise-grade safety evaluations without needing a local GPU.
- **FAISS-powered RAG Demo (`rag_demo.py`)**: A complete Retrieval-Augmented Generation pipeline using local FAISS vector stores and lightweight embeddings to test the `internal-kb-rag` policy end-to-end.
- **Hot-Reloading Docker Environment**: The `docker-compose.yml` now mounts the dashboard and proxy source code directly into the container. Any UI or Python code changes update instantly in the browser—no rebuilds required.
- **Gemini 3.6 Support**: Full integration with the latest `gemini-3.6-flash` and `gemini-3.6-pro` via OpenAI-compatible endpoints.

---

## 🛡️ Core Features (Old & Gold)

- **Drop-In OpenAI Compatibility**: Exposes `POST /v1/chat/completions` supporting both synchronous JSON and streaming Server-Sent Events (`stream=true`).
- **Policy Enforcement Routing**: Dynamic policy selection via `X-ControlPlane-App-ID` header with hot-reloading YAML configuration.
- **Model-Specific Guard Pipelines**:
  - **RAG (`internal-kb-rag`)**:
    - **Pre-Execution**: Prompt Injection Defense, PII Masking Vault, Context Capping (Token Optimization), and Semantic Caching.
    - **Post-Execution**: NLI Grounding Verification (Hallucination), Citation Similarity, Toxicity Check, Subspace Bias Drift, and PII De-tokenization (Recovery).
  - **Chatbot (`customer-support`)**:
    - **Pre-Execution**: Prompt Injection Defense, PII Masking Vault, and Semantic Caching.
    - **Post-Execution**: SLM-as-a-judge Hallucination (via OpenRouter), Toxicity Check, Counterfactual Fairness (Identity Swapping), Subspace Bias Drift, and PII De-tokenization.
  - **Agent (`action-agent`)**:
    - **Pre-Execution**: Prompt Injection Defense, PII Masking Vault, and Semantic Caching.
    - **Post-Execution**: Tool Call Deduplication (Token Loop Optimization), Tool Parameter Validation (Bounds/Schema), Confidence Mismatch (Logprobs check), Reasoning Similarity (Thinking ↔ Output alignment), and PII De-tokenization.
- **Extreme Latency Optimizations**: Concurrent execution of all inline guards (`asyncio.gather`), sub-2ms deterministic Semantic Caching, and Token Context Capping to minimize downstream LLM costs.
- **Deterministic PDP Decision Matrix**: `ALLOW`, `TRANSFORM` (PII masked), `REWRITE` (fallback completion / injected cache output), `BLOCK` (HTTP 422 Unprocessable Content, 429 Too Many Requests).
- **Zero-Overhead Auditing**: Async SQLite WAL persistence recording trace IDs, latency breakdowns, risk scores, and full replayable payloads.
- **Live Observability Dashboard**: Interactive playground and real-time telemetry stream at `http://localhost:8000/dashboard`.

---

## ⚡ Quickstart

### Option 1: Docker Compose (Recommended)

```bash
docker compose up -d
```
Open **`http://localhost:8000/dashboard`** in your browser.
*Note: The proxy code and dashboard are mounted as volumes. UI/Python changes will hot-reload instantly.*

---

### Option 2: Local Python Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests:**
   ```bash
   python -m pytest -v
   ```

3. **Start the ControlPlane Proxy:**
   ```bash
   python -m controlplane.main
   ```
   Or via uvicorn:
   ```bash
   uvicorn controlplane.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🧪 Demo Scripts

`ControlPlane.ai` ships with end-to-end Python demo scripts to pit direct LLMs against our Proxy in real-time.

**Agentic Safety Demo:**
```bash
python agent_demo.py
```
*Simulates Tool Schema Violations, Jailbreaks, SQL Injections, and Tool Loops.*

**RAG Grounding Demo:**
```bash
python rag_demo.py
```
*Creates a local FAISS vector store and simulates retrieving context chunks to test grounding and context capping.*

---

## 📁 Modular Repository Architecture

```text
ControlPlane/
├── config/
│   └── policies.yaml                     # Policy rule definitions & threshold matrix
├── controlplane/
│   ├── proxy/                            # OpenAI wire-compatible ingress & SSE streaming
│   ├── policy/                           # Hot-reloading YAML registry & Pydantic models
│   ├── pdp/                              # Deterministic Policy Decision Point (ALLOW/BLOCK/REWRITE/TRANSFORM)
│   ├── detectors/                        # Isolated modular guard & detection subsystems
│   │   ├── pii/                          # Dual-tier regex/Presidio masking vault & detokenizer
│   │   ├── injection/                    # Heuristic & semantic prompt injection defense
│   │   ├── hallucination/                # Claim splitter, context aligner & NLI grounding (OpenRouter)
│   │   ├── bias_toxicity/                # Toxicity & demographic subspace classifier (OpenRouter)
│   │   └── tool_safety/                  # Function call schema & bounds interception
│   ├── telemetry/                        # Async SQLite WAL audit sink & metrics query
│   └── mock_upstream/                    # Embedded scenario-aware test bed
├── dashboard/                            # Real-time dark-mode observability interface
├── tests/                                # Isolated unit & end-to-end integration tests
├── Dockerfile
└── docker-compose.yml
```
