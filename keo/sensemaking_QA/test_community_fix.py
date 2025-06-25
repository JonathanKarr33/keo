#!/usr/bin/env python3
"""
Test the fixed community detection in answer generator
"""

import os
from answer_generator import SensemakingAnswerGenerator

def test_community_detection():
    """Test that community detection works with directed graphs"""
    
    print("Testing Community Detection Fix")
    print("=" * 40)
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return False
    
    # Initialize answer generator
    print("Initializing answer generator...")
    answer_generator = SensemakingAnswerGenerator(openai_api_key)
    
    # Load knowledge graph
    kg_path = "../kg/knowledge_graph.gml"
    print(f"Loading knowledge graph: {kg_path}")
    
    kg_loaded = answer_generator.load_knowledge_graph(kg_path)
    if not kg_loaded:
        print("ERROR: Could not load knowledge graph")
        return False
    
    print(f"✓ Knowledge graph loaded successfully")
    print(f"  - Nodes: {answer_generator.knowledge_graph.number_of_nodes()}")
    print(f"  - Edges: {answer_generator.knowledge_graph.number_of_edges()}")
    print(f"  - Directed: {answer_generator.knowledge_graph.is_directed()}")
    
    # Test community detection
    print("\nTesting community detection...")
    try:
        test_question = "What should be done when engine components are leaking?"
        community_summary = answer_generator._get_community_summaries(test_question)
        
        print("✓ Community detection successful!")
        print("\nCommunity Summary (first 500 chars):")
        print("-" * 40)
        print(community_summary[:500] + "..." if len(community_summary) > 500 else community_summary)
        
        return True
        
    except Exception as e:
        print(f"ERROR in community detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_community_detection()
    if success:
        print("\n✅ Community detection test PASSED!")
    else:
        print("\n❌ Community detection test FAILED!")
