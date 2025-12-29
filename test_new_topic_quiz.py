#!/usr/bin/env python3
"""
Test script to verify new topic quiz detection works correctly
"""

import sys
import os
from pathlib import Path

# Add personalized_rag to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'personalized_rag'))

from personalized_rag.local_llm_rag import LocalLLMRAGSystem

def test_new_topic_detection():
    """Test if system correctly detects new topics"""
    
    print("=" * 60)
    print("Testing New Topic Detection")
    print("=" * 60)
    
    # Create test user
    test_user = "TestUser"
    
    # Initialize RAG system
    print(f"\n1. Initializing RAG system for user: {test_user}")
    rag = LocalLLMRAGSystem(user_name=test_user, use_external_sources=False)
    
    # Test topics
    test_topics = [
        "quantum computing",
        "machine learning",
        "blockchain technology",
        "neural networks"
    ]
    
    print(f"\n2. Testing topic detection for {len(test_topics)} topics:")
    print("-" * 60)
    
    for topic in test_topics:
        is_new = not rag.is_topic_in_user_kb(topic)
        status = "✅ NEW" if is_new else "❌ KNOWN"
        print(f"   {topic:30} → {status}")
    
    # Simulate saving a topic
    print(f"\n3. Simulating learning 'quantum computing'...")
    rag.save_to_user_kb(
        query="What is quantum computing?",
        topic="quantum computing",
        response="Quantum computing uses quantum mechanics...",
        entry_type='query'
    )
    
    # Check again
    print(f"\n4. Re-checking 'quantum computing' after saving:")
    is_new = not rag.is_topic_in_user_kb("quantum computing")
    status = "✅ NEW" if is_new else "❌ KNOWN (correctly detected!)"
    print(f"   quantum computing → {status}")
    
    # Test fuzzy matching
    print(f"\n5. Testing fuzzy matching:")
    fuzzy_tests = [
        ("quantum", "Should match 'quantum computing'"),
        ("computing", "Should match 'quantum computing'"),
        ("quantum mechanics", "Should match 'quantum computing'"),
        ("totally different topic", "Should NOT match")
    ]
    
    for test_query, expected in fuzzy_tests:
        is_new = not rag.is_topic_in_user_kb(test_query)
        status = "NEW" if is_new else "KNOWN"
        print(f"   '{test_query:25}' → {status:6} ({expected})")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    # Cleanup
    kb_file = Path(f"personalized_rag/user_profiles/{test_user}_knowledge_base.csv")
    if kb_file.exists():
        print(f"\n📁 Knowledge base created at: {kb_file}")
        print(f"   You can delete it with: rm {kb_file}")

if __name__ == "__main__":
    try:
        test_new_topic_detection()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
