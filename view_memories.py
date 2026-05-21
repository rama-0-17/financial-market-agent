#!/usr/bin/env python3
"""
View all stored analyses in the long-term memory (ChromaDB).
Run this from the repo root: python view_memories.py
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.store import _get_collection, recall_memories

def view_all_memories():
    """Display all stored memories."""
    try:
        collection = _get_collection()
        count = collection.count()
        
        if count == 0:
            print("No analyses stored yet.")
            return
        
        print(f"\nTotal stored analyses: {count}\n")
        print("=" * 80)
        
        # Get all memories (query with empty string returns all)
        all_data = collection.get()
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            all_data["ids"],
            all_data["documents"],
            all_data["metadatas"]
        ), 1):
            print(f"\nAnalysis #{i}")
            print(f"   ID: {doc_id}")
            print(f"   Stored: {metadata.get('timestamp', 'N/A')}")
            print(f"   Hash: {metadata.get('content_hash', 'N/A')}")
            print(f"   Preview: {document[:150]}...")
            print("-" * 80)
    except FileNotFoundError:
        print("Memory database not found. Run some analyses first.")
    except Exception as e:
        print(f"Error reading memories: {e}")

def search_memories(query: str):
    """Search memories by query."""
    try:
        print(f"\nSearching for: '{query}'\n")
        results = recall_memories(query, n_results=10)
        
        if not results:
            print("   No matches found.")
            return
        
        for i, mem in enumerate(results, 1):
            print(f"\nMatch {i} (relevance: {1 - mem['distance']:.1%})")
            print(f"   ID: {mem['id']}")
            print(f"   Content: {mem['content'][:200]}...")
            if mem['metadata']:
                print(f"   Timestamp: {mem['metadata'].get('timestamp', 'N/A')}")
    except FileNotFoundError:
        print("Memory database not found. Run some analyses first.")
    except Exception as e:
        print(f"Error searching memories: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Search mode: python view_memories.py "query string"
        query = " ".join(sys.argv[1:])
        search_memories(query)
    else:
        # View all mode
        view_all_memories()
