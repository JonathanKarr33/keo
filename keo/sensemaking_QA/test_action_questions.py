#!/usr/bin/env python3
"""
Test Action-Specific Question Generation
"""

import os
import sys
import pandas as pd
from data_analyzer import AviationDataAnalyzer
from question_generator import SensemakingQuestionGenerator

def test_action_specific_questions():
    """Test the new action-specific question generation"""
    
    print("Testing Action-Specific Question Generation")
    print("=" * 50)
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    print("✓ OpenAI API key found")
    
    # Load data
    data_paths = {
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    
    # Check data structure
    df = analyzer.datasets['aircraft_annotation']
    print(f"✓ Loaded aircraft annotation data: {df.shape[0]} records")
    
    # Show sample data
    print("\nSample Problem-Action Pairs:")
    print("-" * 50)
    for i, row in df.head(5).iterrows():
        print(f"Problem: {row['PROBLEM']}")
        print(f"Action: {row['ACTION']}")
        print()
    
    # Initialize question generator
    generator = SensemakingQuestionGenerator(openai_api_key)
    
    # Test action-specific question generation
    print("Generating action-specific questions...")
    print("-" * 50)
    
    try:
        action_questions = generator.generate_action_specific_questions(
            analyzer, 
            num_questions=3  # Small number for testing
        )
        
        print(f"✓ Successfully generated {len(action_questions)} action-specific questions")
        
        # Display generated questions
        print("\nGenerated Action-Specific Questions:")
        print("=" * 50)
        
        for i, q in enumerate(action_questions, 1):
            print(f"Question {i}:")
            print(f"  Q: {q['question']}")
            print(f"  A: {q['ground_truth_answer']}")
            print(f"  Original Problem: {q['original_problem']}")
            print(f"  Category: {q['category']}")
            print(f"  Type: {q['type']}")
            print()
            
        return action_questions
        
    except Exception as e:
        print(f"ERROR in question generation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_action_specific_questions()
