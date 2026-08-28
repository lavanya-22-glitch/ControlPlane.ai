from asyncio import taskgroups
import os
import httpx
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
PROXY_URL = "http://localhost:8000/v1/chat/completions"
DIRECT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# 1. The RAG Knowledge Base (Hardcoded for demo)
# In a real RAG app, you would fetch these from a Vector DB (Pinecone, Weaviate, etc.) based on the user's question.
KNOWLEDGE_BASE = [
    "Our store policy states that we only sell red apples and yellow bananas.",
    "We do not sell oranges, and we have a strict no-refund policy.",
    "The CEO of the company is Mr. John Smith."
]

def ask_rag(user_question: str):
    print(f"\n[RAG Client] User asks: '{user_question}'")
    print("[RAG Client] Retrieving context from Vector DB...")
    
    # Simulate retrieving context (we just send the whole KB for the demo)
    retrieved_context = KNOWLEDGE_BASE
    
    # 2. Construct the Proxy Payload
    # Notice how we send the 'retrieved_context' explicitly so the Proxy can run Grounding checks!
    payload = {
        "model": "gemini-3.6-flash",
        "messages": [
            {"role": "user", "content": user_question}
        ],
        "retrieved_context": retrieved_context
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-ControlPlane-App-ID": "internal-kb-rag" # This triggers the NLI Hallucination Guard
    }
    
    print("[RAG Client] Sending payload DIRECTLY to Gemini API (No Proxy)...")
    try:
        start_time = time.time()
        # Direct API doesn't accept 'retrieved_context', so we must strip it just like the proxy does!
        direct_payload = {k: v for k, v in payload.items() if k != "retrieved_context"}
        response = httpx.post(DIRECT_URL, json=direct_payload, headers=headers, timeout=30.0)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"\n--- DIRECT GEMINI RESPONSE ({latency:.2f} ms) ---")
            print(f"Final LLM Output: {content}")
        else:
            print(f"\n--- DIRECT GEMINI RESPONSE FAILED ---")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n[RAG Client] Sending payload to ControlPlane.ai Proxy...")
    try:
        start_time = time.time()
        response = httpx.post(PROXY_URL, json=payload, headers=headers, timeout=30.0)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        print(f"\n--- PROXY RESPONSE ({latency:.2f} ms) ---")
        print(f"HTTP Status: {response.status_code}")
        
        # Read the telemetry headers injected by the proxy
        action = response.headers.get("x-controlplane-policy-action", "UNKNOWN")
        grounding_score = response.headers.get("x-controlplane-score-grounding", "N/A")
        violation_reason = response.headers.get("x-controlplane-violation-reason", "None")
        
        print(f"Action Taken: {action}")
        if grounding_score != "N/A":
            print(f"NLI Grounding Score: {grounding_score}")
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"\nFinal LLM Output: {content}")
        else:
            print(f"\nBLOCKED BY PROXY: {violation_reason}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("WELCOME TO THE RAG HALLUCINATION DEMO")
    print("="*60)
    print("Context loaded into Vector DB:")
    for doc in KNOWLEDGE_BASE:
        print(f" - {doc}")
    print("\n")
    
    # Test 1: Grounded question
    ask_rag("Do you sell oranges?")
    
    time.sleep(2)
    
    # Test 2: Out-of-bounds / Hallucination question
    # Gemini will use its worldly knowledge to answer this, but because it contradicts the context,
    # the proxy's Hallucination guard should flag it!
    ask_rag("Who was the 16th president of the United States?")

    time.sleep(2)
    ask_rag("Do you sell apples?")