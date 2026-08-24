import time
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, Response, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from controlplane.policy.registry import policy_registry
from controlplane.proxy.schemas import ChatCompletionRequest
from controlplane.proxy.dispatcher import dispatcher
from controlplane.proxy.stream_handler import handle_buffered_stream
from controlplane.mock_upstream.server import mock_llm_provider
from controlplane.detectors.base import GuardContext, DetectionResult
from controlplane.detectors.pii.scanner import PIIScannerGuard
from controlplane.detectors.pii.detokenizer import PIIDetokenizerGuard
from controlplane.detectors.injection.classifier import PromptInjectionGuard
from controlplane.detectors.hallucination.nli_verifier import NLIRAGGroundingGuard
from controlplane.detectors.bias_toxicity.toxicity_scorer import ToxicityGuard
from controlplane.detectors.bias_toxicity.bias_subspace import BiasSubspaceGuard
from controlplane.detectors.tool_safety.validator import AgentToolGuard
from controlplane.pdp.engine import pdp_engine
from controlplane.pdp.types import PDPAction, ViolationCode
from controlplane.pdp.actions import build_error_response
from controlplane.telemetry.schema import AuditTraceRecord
from controlplane.telemetry.sink import telemetry_sink
from controlplane.telemetry.query import telemetry_query

logger = logging.getLogger("controlplane.proxy.router")
router = APIRouter()

# Instantiate active detector instances
pii_scanner = PIIScannerGuard()
pii_detokenizer = PIIDetokenizerGuard()
injection_guard = PromptInjectionGuard()
rag_grounding_guard = NLIRAGGroundingGuard()
toxicity_guard = ToxicityGuard()
bias_guard = BiasSubspaceGuard()
tool_guard = AgentToolGuard()


@router.post("/v1/chat/completions")
async def chat_completions(
    req_body: ChatCompletionRequest,
    request: Request,
    x_controlplane_app_id: Optional[str] = Header(default=None, alias="X-ControlPlane-App-ID"),
):
    """Drop-in OpenAI-compatible chat completions proxy endpoint."""
    t_start = time.perf_counter()
    trace_id = f"cp_trace_{uuid.uuid4().hex[:12]}"
    app_id = x_controlplane_app_id or "default"
    policy = policy_registry.get_policy(app_id)

    raw_payload_dict = req_body.model_dump(exclude_unset=True)
    messages_payload = [m.model_dump(exclude_unset=True) for m in req_body.messages]

    ctx = GuardContext(
        trace_id=trace_id,
        app_id=app_id,
        policy=policy,
        messages=messages_payload,
        retrieved_context=req_body.retrieved_context,
        tool_calls=None,
    )

    # -------------------------------------------------------------
    # 1. PRE-EXECUTION PIPELINE (Injection Check -> PII Masking)
    # -------------------------------------------------------------
    t_pre_start = time.perf_counter()
    pre_results: List[DetectionResult] = []

    # Prompt Injection Guard
    inj_res = await injection_guard.evaluate(ctx)
    pre_results.append(inj_res)

    # PII Scanner & Vault Masking
    pii_res = await pii_scanner.evaluate(ctx)
    pre_results.append(pii_res)

    t_pre_end = time.perf_counter()
    pre_latency_ms = (t_pre_end - t_pre_start) * 1000

    # Evaluate Pre-Execution Decision
    pre_decision = pdp_engine.evaluate(ctx, pre_results, upstream_model=req_body.model)
    if pre_decision.action == PDPAction.BLOCK:
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        error_body = build_error_response(
            code=pre_decision.violation_code,
            message=pre_decision.reason or "Pre-execution policy violation.",
            trace_id=trace_id,
            policy_version=policy.version,
            action=PDPAction.BLOCK,
        )
        # Record trace
        await telemetry_sink.record_trace(
            AuditTraceRecord(
                trace_id=trace_id,
                app_id=app_id,
                policy_version=policy.version,
                policy_hash=policy.policy_hash or "default",
                use_case_mode=policy.mode,
                latency_ms=total_latency_ms,
                pre_execution_latency_ms=pre_latency_ms,
                injection_score=pre_decision.scores.get("prompt_injection"),
                pii_entities_found=pre_decision.entities_found,
                pdp_action=PDPAction.BLOCK.value,
                violation_reason=pre_decision.reason,
                raw_request_payload=json.dumps(raw_payload_dict),
                final_response_payload=json.dumps(error_body),
            )
        )
        return JSONResponse(status_code=422, content=error_body)

    # Update request messages with masked versions
    upstream_payload = raw_payload_dict.copy()
    upstream_payload["messages"] = ctx.messages

    # -------------------------------------------------------------
    # 2. UPSTREAM DISPATCH (Live API or Mock Provider)
    # -------------------------------------------------------------
    t_upstream_start = time.perf_counter()

    # Handle streaming request
    if req_body.stream:
        post_guards = [tool_guard, rag_grounding_guard, toxicity_guard, bias_guard, pii_detokenizer]
        stream_gen = mock_llm_provider.stream_complete(upstream_payload)
        return StreamingResponse(
            handle_buffered_stream(stream_gen, ctx, post_guards, model=req_body.model),
            media_type="text/event-stream",
            headers={"X-ControlPlane-Trace-ID": trace_id},
        )

    # Synchronous Request
    raw_headers = dict(request.headers)
    upstream_response = await dispatcher.dispatch(upstream_payload, raw_headers)
    t_upstream_end = time.perf_counter()
    upstream_latency_ms = (t_upstream_end - t_upstream_start) * 1000

    # -------------------------------------------------------------
    # 3. POST-EXECUTION PIPELINE (Tool Guard, RAG Grounding, Toxicity, De-tokenization)
    # -------------------------------------------------------------
    t_post_start = time.perf_counter()
    choice = upstream_response.get("choices", [{}])[0]
    message_obj = choice.get("message", {})
    ctx.completion_text = message_obj.get("content") or ""
    ctx.tool_calls = message_obj.get("tool_calls")

    post_results: List[DetectionResult] = []

    # Tool Call Safety Guard
    if ctx.tool_calls:
        tool_res = await tool_guard.evaluate(ctx)
        post_results.append(tool_res)

    # RAG Grounding & Hallucination Guard
    if ctx.retrieved_context and ctx.completion_text:
        rag_res = await rag_grounding_guard.evaluate(ctx)
        post_results.append(rag_res)

    # Toxicity & Bias Guards
    if ctx.completion_text:
        tox_res = await toxicity_guard.evaluate(ctx)
        post_results.append(tox_res)
        bias_res = await bias_guard.evaluate(ctx)
        post_results.append(bias_res)

    # Egress PII De-tokenization
    detok_res = await pii_detokenizer.evaluate(ctx)
    post_results.append(detok_res)

    t_post_end = time.perf_counter()
    post_latency_ms = (t_post_end - t_post_start) * 1000

    # -------------------------------------------------------------
    # 4. FINAL POLICY DECISION POINT (PDP)
    # -------------------------------------------------------------
    all_results = pre_results + post_results
    final_decision = pdp_engine.evaluate(ctx, all_results, upstream_model=req_body.model)
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    final_response_body: Dict[str, Any] = upstream_response
    http_status_code = 200

    if final_decision.action == PDPAction.BLOCK:
        http_status_code = 422
        final_response_body = build_error_response(
            code=final_decision.violation_code,
            message=final_decision.reason or "Post-execution policy violation.",
            trace_id=trace_id,
            policy_version=policy.version,
            action=PDPAction.BLOCK,
        )
    elif final_decision.action == PDPAction.REWRITE and final_decision.fallback_payload:
        final_response_body = final_decision.fallback_payload
    else:
        # ALLOW or TRANSFORM: update completion with detokenized text
        if ctx.completion_text is not None and "choices" in final_response_body:
            final_response_body["choices"][0]["message"]["content"] = ctx.completion_text

    # Attach ControlPlane trace metadata to response header
    response_headers = {
        "X-ControlPlane-Trace-ID": trace_id,
        "X-ControlPlane-Policy-Action": final_decision.action.value,
        "X-ControlPlane-Latency-MS": f"{total_latency_ms:.2f}",
    }

    # -------------------------------------------------------------
    # 5. ASYNC AUDIT TELEMETRY SINK
    # -------------------------------------------------------------
    usage = upstream_response.get("usage", {})
    await telemetry_sink.record_trace(
        AuditTraceRecord(
            trace_id=trace_id,
            app_id=app_id,
            policy_version=policy.version,
            policy_hash=policy.policy_hash or "default",
            use_case_mode=policy.mode,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=total_latency_ms,
            pre_execution_latency_ms=pre_latency_ms,
            upstream_latency_ms=upstream_latency_ms,
            post_execution_latency_ms=post_latency_ms,
            pii_entities_found=final_decision.entities_found,
            injection_score=final_decision.scores.get("prompt_injection"),
            grounding_score=final_decision.scores.get("rag_hallucination_engine"),
            bias_score=final_decision.scores.get("bias_subspace_detector"),
            pdp_action=final_decision.action.value,
            violation_reason=final_decision.reason if final_decision.action != PDPAction.ALLOW else None,
            raw_request_payload=json.dumps(raw_payload_dict),
            final_response_payload=json.dumps(final_response_body),
        )
    )

    return JSONResponse(
        status_code=http_status_code,
        content=final_response_body,
        headers=response_headers,
    )


# -------------------------------------------------------------
# OBSERVABILITY & MANAGEMENT ENDPOINTS
# -------------------------------------------------------------

@router.get("/v1/traces")
async def list_traces(limit: int = 50, app_id: Optional[str] = None):
    """Retrieve recent transaction audit logs."""
    traces = await telemetry_query.get_recent_traces(limit=limit, app_id=app_id)
    return {"traces": traces, "count": len(traces)}


@router.get("/v1/metrics")
async def get_metrics():
    """Retrieve system aggregated telemetry metrics."""
    return await telemetry_query.get_metrics_summary()


@router.get("/policies")
async def list_policies():
    """List loaded policies and their configurations."""
    return {"policies": policy_registry.list_policies()}


@router.post("/policies/reload")
async def reload_policies():
    """Hot-reload policy configurations from disk."""
    policy_registry.reload()
    return {"status": "reloaded", "policies_count": len(policy_registry.list_policies())}


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ControlPlane.ai", "version": "1.0.0"}
