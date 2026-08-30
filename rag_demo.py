#!/usr/bin/env python3
"""
rag_demo.py
───────────────────────────────────────────────────────────────────────────
End-to-end RAG comparison demo: Direct Gemini  vs  ControlPlane Proxy

Shows side-by-side differences across 5 dimensions:
  1. Hallucination   — does the proxy catch ungrounded answers?
  2. Bias            — does the proxy block out-of-scope/biased responses?
  3. PII Leakage     — does the proxy redact/block sensitive data?
  4. Token Usage     — how many tokens does each path consume?
  5. Latency         — what overhead does the proxy add?

Usage:
  python rag_demo.py                  # run all 5 preset demo queries
  python rag_demo.py --interactive    # drop into a free-form Q&A loop
  python rag_demo.py --rebuild        # force rebuild FAISS index, then run

Requirements (install once):
  pip install faiss-cpu rank_bm25 langchain langchain-community langchain-openai sentence-transformers httpx python-dotenv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ── Terminal colour helpers ────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"

def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"

def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"

def hr(char: str = "─", width: int = 72, colour: str = DIM) -> str:
    return c(char * width, colour)


# ── Pretty printing helpers ────────────────────────────────────────────────

def _wrap(text: str, width: int = 68, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_header() -> None:
    print()
    print(hr("═", 72, CYAN))
    print(c(bold("  🚀  FAISS RAG  ·  HYBRID RETRIEVAL + CROSS-ENCODER  "), CYAN))
    print(c(bold("       Direct Gemini  vs  ControlPlane Proxy"), CYAN))
    print(hr("═", 72, CYAN))
    print()


def print_section(title: str) -> None:
    print()
    print(hr("─", 72, BLUE))
    print(c(f"  ▶  {title}", BLUE + BOLD))
    print(hr("─", 72, BLUE))


def print_retrieved_chunks(chunks, scores) -> None:
    print(c("  📄  Top Re-ranked Chunks:", YELLOW + BOLD))
    for i, (doc, score) in enumerate(zip(chunks, scores), 1):
        snippet = doc.page_content[:120].replace("\n", " ")
        print(c(f"    [{i}] score={score:+.3f}  ", GREEN) + c(f'"{snippet}…"', DIM))


def print_comparison(label: str, direct_val: str, proxy_val: str, highlight: bool = False) -> None:
    tag  = c(f"  {label:<22}", WHITE + BOLD)
    dval = c(f"DIRECT: {direct_val}", RED if highlight else DIM)
    pval = c(f"PROXY: {proxy_val}",  GREEN)
    print(f"{tag}  {dval}   {pval}")


def print_response_block(tag: str, colour: str, response) -> None:
    from rag.pipeline import RAGResponse
    r: RAGResponse = response

    print()
    print(c(f"  ── {tag} ──", colour + BOLD))

    if r.error:
        print(c(f"    ❌  ERROR: {r.error}", RED))
        return

    if r.http_status != 200:
        print(c(f"    🚫  HTTP {r.http_status} — Request blocked/rejected by proxy", RED + BOLD))
        if r.violation_reason:
            print(c(f"    Reason: {r.violation_reason}", RED))
        return

    answer_preview = r.answer[:400].replace("\n", " ") if r.answer else "(empty)"
    print(c("    Answer:", colour))
    print(_wrap(answer_preview))
    if len(r.answer) > 400:
        print(c("    … (truncated)", DIM))


def print_metrics_table(direct, proxy) -> None:
    from rag.pipeline import RAGResponse
    d: RAGResponse = direct
    p: RAGResponse = proxy

    print()
    print(c("  📊  Metrics Comparison", CYAN + BOLD))
    print()

    # Latency
    lat_diff = p.latency_ms - d.latency_ms
    lat_tag  = c(f"({lat_diff:+.0f} ms proxy overhead)", YELLOW)
    print(c(f"    {'Latency':<22}", WHITE + BOLD) +
          c(f"Direct: {d.latency_ms:7.0f} ms   ", RED) +
          c(f"Proxy: {p.latency_ms:7.0f} ms  ", GREEN) + lat_tag)

    # Tokens
    print(c(f"    {'Prompt Tokens':<22}", WHITE + BOLD) +
          c(f"Direct: {d.prompt_tokens:>6}         ", RED) +
          c(f"Proxy: {p.prompt_tokens:>6}", GREEN))
    print(c(f"    {'Completion Tokens':<22}", WHITE + BOLD) +
          c(f"Direct: {d.completion_tokens:>6}         ", RED) +
          c(f"Proxy: {p.completion_tokens:>6}", GREEN))
    print(c(f"    {'Total Tokens':<22}", WHITE + BOLD) +
          c(f"Direct: {d.total_tokens:>6}         ", RED) +
          c(f"Proxy: {p.total_tokens:>6}", GREEN))

    # Proxy-specific guard signals
    if p.proxy_action is not None:
        action_colour = GREEN if p.proxy_action == "ALLOW" else RED + BOLD
        print()
        print(c(f"    {'Proxy Action':<22}", WHITE + BOLD) + c(p.proxy_action, action_colour))

    if p.grounding_score is not None:
        print(c(f"    {'NLI Grounding Score':<22}", WHITE + BOLD) +
              c(f"{p.grounding_score}", YELLOW))

    if p.violation_reason:
        print(c(f"    {'Violation Reason':<22}", WHITE + BOLD) +
              c(p.violation_reason, RED + BOLD))

    if p.pii_detected:
        print(c(f"    {'PII Detected':<22}", WHITE + BOLD) +
              c(p.pii_detected, MAGENTA + BOLD))


def run_query(
    label: str,
    question: str,
    direct_pipeline,
    proxy_pipeline,
    verbose: bool = True,
) -> None:
    print_section(label)
    print(c(f"  ❓  Question: ", WHITE + BOLD) + c(question, CYAN))
    print()

    # Retrieve & rerank (shared — same result for both paths)
    print(c("  🔍  Running hybrid retrieval + cross-encoder rerank ...", DIM))
    # We call direct first (it triggers the shared stack internally)
    
    # ── Direct call ────────────────────────────────────────────────────────
    print(c("\n  ⚡  Calling DIRECT Gemini (no guardrails) ...", RED))
    t0 = time.perf_counter()
    direct_resp = direct_pipeline.ask(question)

    # ── Proxy call ─────────────────────────────────────────────────────────
    print(c("  🛡️   Calling ControlPlane PROXY (with guardrails) ...", GREEN))
    proxy_resp = proxy_pipeline.ask(question)

    # ── Show retrieved chunks (from whichever has them) ────────────────────
    if verbose and direct_resp.reranked_chunks:
        print()
        print_retrieved_chunks(direct_resp.reranked_chunks, direct_resp.rerank_scores)

    # ── Response blocks ────────────────────────────────────────────────────
    print_response_block("DIRECT GEMINI (no guardrails)", RED, direct_resp)
    print_response_block("CONTROLPLANE PROXY (with guardrails)", GREEN, proxy_resp)

    # ── Metrics table ──────────────────────────────────────────────────────
    print_metrics_table(direct_resp, proxy_resp)

    print()
    print(hr())


def interactive_loop(direct_pipeline, proxy_pipeline) -> None:
    print()
    print(hr("═", 72, CYAN))
    print(c(bold("  💬  INTERACTIVE MODE"), CYAN))
    print(hr("═", 72, CYAN))
    print()
    print(c("  Ask any question about the TechFlow platform.", WHITE))
    print(c("  Every question runs through BOTH paths simultaneously:", WHITE))
    print(c("    🔴 Direct Gemini  — raw LLM, no guardrails", RED))
    print(c("    🟢 ControlPlane   — hallucination guard + PII redaction + bias check", GREEN))
    print()
    print(c("  Try asking things like:", YELLOW + BOLD))
    print(c("    • 'What is the refund policy?'             (grounded)", DIM))
    print(c("    • 'Who invented the internet?'             (hallucination trap)", DIM))
    print(c("    • 'What is the CEO\'s personal email?'     (PII)", DIM))
    print(c("    • 'Which political party does TechFlow support?' (bias)", DIM))
    print()
    print(c("  Type 'quit' or press Ctrl+C to exit.", DIM))
    print()
    while True:
        try:
            question = input(c("  You ❯ ", WHITE + BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        run_query("CUSTOM QUERY", question, direct_pipeline, proxy_pipeline)
        print(c("  ⏳  Cooling down 3s ...", DIM))
        time.sleep(3)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAISS RAG side-by-side comparison demo"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the FAISS index from the corpus",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide retrieved chunk details",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive chat loop at the end of the script",
    )
    args = parser.parse_args()

    # Optionally rebuild index
    if args.rebuild:
        import shutil
        from rag.config import FAISS_INDEX_DIR
        index_path = FAISS_INDEX_DIR
        if index_path.exists():
            shutil.rmtree(index_path)
            print(c(f"[Setup] Deleted existing index at {index_path}", YELLOW))

    print_header()

    # ── Bootstrap retrieval stack (shared between both pipelines) ──────────
    print(c("  ⚙️   Initialising retrieval stack ...", DIM))
    from rag.pipeline import RetrievalStack, DirectPipeline, ProxyPipeline
    stack = RetrievalStack.get()
    direct_pipeline = DirectPipeline(stack)
    proxy_pipeline  = ProxyPipeline(stack)
    print(c("  ✅  Stack ready.\n", GREEN))

    # ── Preset demo queries ────────────────────────────────────────────────
    from rag.config import DEMO_QUERIES
    for label, question, _ in DEMO_QUERIES:
        run_query(
            label,
            question,
            direct_pipeline,
            proxy_pipeline,
            verbose=not args.quiet,
        )
        print(c("  ⏳  Sleeping 4 seconds to avoid API rate limits...", DIM))
        time.sleep(4)   # pause between queries to avoid rate limits

    # ── Optional interactive mode (default ON, skip with --no-interactive) ──
    if not args.no_interactive:
        interactive_loop(direct_pipeline, proxy_pipeline)

    print()
    print(c(bold("  Demo complete. "), CYAN))
    print(hr("═", 72, CYAN))
    print()


if __name__ == "__main__":
    main()
