"""Quick end-to-end retrieval smoke test — no LLM calls."""
from rag.pipeline import RetrievalStack

print("Building retrieval stack (ONNX model downloads on first run) ...")
stack = RetrievalStack.get()

query = "What is the refund policy?"
print(f"\nQuery: {query}")
retrieved, reranked, scores = stack.retrieve_and_rerank(query)

print(f"\nHybrid retrieved : {len(retrieved)} chunks")
print(f"Reranked to top  : {len(reranked)} chunks\n")
print("--- Top reranked chunks ---")
for i, (doc, score) in enumerate(zip(reranked, scores), 1):
    snippet = doc.page_content[:120].replace("\n", " ")
    print(f"  [{i}] score={score:.4f}  \"{snippet}...\"")
print("\nRetrieval pipeline OK!")
