"""
Memory system — two-tier architecture:

1. Short-term (scratchpad): string appended to every LLM prompt.
   Holds tool results, intermediate conclusions, failed attempts.
   Resets per agent run.

2. Long-term (ChromaDB): persisted vector store.
   Stores key facts, company profiles, past analyses.
   Recalled at the start of each run via semantic search.
   Grows across sessions — the agent gets smarter over time.
"""
from __future__ import annotations
import json
import uuid
import hashlib
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings


# ── ChromaDB setup ────────────────────────────────────────────────────────────

_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "fin_agent_memory"


class _SimpleEmbeddingFunction:
    """
    Lightweight TF-IDF-style embedding that works fully offline.
    Not as accurate as a neural embedder but good enough for semantic recall
    without requiring a model download or API call.

    On your own machine you can swap this for:
      from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
      ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    """
    DIM = 256

    def name(self) -> str:
        return "simple_hash_embedding"

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return self._embed(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def _embed(self, input: list[str]) -> list[list[float]]:
        import hashlib, math
        embeddings = []
        for text in input:
            vec = [0.0] * self.DIM
            words = text.lower().split()
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self.DIM
                vec[idx] += 1.0 / (1 + math.log(1 + words.count(word)))
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_SimpleEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Long-term memory ──────────────────────────────────────────────────────────

def store_memory(content: str, metadata: dict | None = None) -> str:
    """
    Store a fact or analysis result in long-term vector memory.
    Uses a simple hash embedding (ChromaDB default) for free-tier compatibility.

    Returns the memory ID.
    """
    collection = _get_collection()
    mem_id = str(uuid.uuid4())
    meta = {
        "timestamp": datetime.now().isoformat(),
        "content_hash": hashlib.md5(content.encode()).hexdigest()[:8],
        **(metadata or {}),
    }
    # Truncate content for embedding (ChromaDB limit)
    content_trunc = content[:8000]
    collection.add(
        documents=[content_trunc],
        metadatas=[meta],
        ids=[mem_id],
    )
    return mem_id


def recall_memories(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve relevant past memories via semantic similarity.
    Returns list of {id, content, metadata, distance}.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    n = min(n_results, count)
    results = collection.query(
        query_texts=[query],
        n_results=n,
    )

    memories = []
    for i, doc in enumerate(results["documents"][0]):
        memories.append({
            "id": results["ids"][0][i],
            "content": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return memories


def delete_memory(mem_id: str) -> bool:
    """Delete a specific memory by ID."""
    try:
        _get_collection().delete(ids=[mem_id])
        return True
    except Exception:
        return False


def get_memory_count() -> int:
    """Return total number of stored memories."""
    return _get_collection().count()


# ── Short-term scratchpad ─────────────────────────────────────────────────────

class Scratchpad:
    """
    In-memory working scratchpad for a single agent run.
    Appended to LLM context so the agent can reference earlier steps.
    """

    def __init__(self):
        self._entries: list[dict] = []

    def add(self, step: str, content: str, tool: str | None = None) -> None:
        self._entries.append({
            "step": step,
            "tool": tool,
            "content": content[:2000],  # cap individual entries
            "timestamp": datetime.now().isoformat(),
        })

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self.add(f"Tool: {tool_name}", result, tool=tool_name)

    def add_error(self, step: str, error: str) -> None:
        self.add(f"ERROR in {step}", error)

    def add_plan(self, plan_text: str) -> None:
        self.add("PLAN", plan_text)

    def to_string(self, max_chars: int = 6000) -> str:
        """
        Render scratchpad as a string for LLM context injection.
        Truncates oldest entries first if over limit.
        """
        if not self._entries:
            return ""

        lines = ["=== WORKING MEMORY (scratchpad) ==="]
        for e in self._entries:
            header = f"[{e['step']}]"
            if e.get("tool"):
                header += f" via {e['tool']}"
            lines.append(header)
            lines.append(e["content"])
            lines.append("")

        full = "\n".join(lines)
        if len(full) > max_chars:
            # Trim from the middle, keep first plan + last N entries
            keep_first = self._entries[:1]
            keep_last = self._entries[-2:]
            trimmed_lines = ["=== WORKING MEMORY (truncated) ==="]
            for e in keep_first + [{"step": "...", "tool": None, "content": "(older entries trimmed)"}] + keep_last:
                trimmed_lines.append(f"[{e['step']}]")
                trimmed_lines.append(e["content"])
                trimmed_lines.append("")
            full = "\n".join(trimmed_lines)

        return full

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
