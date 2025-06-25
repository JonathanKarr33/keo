#!/usr/bin/env python3
"""
Generate RAG-based Answers for Aviation Maintenance Sensemaking Questions
Uses both vanilla LLM and GraphRAG approaches with the knowledge graph
"""

import os
import json
import pandas as pd
from datetime import datetime
from data_analyzer import AviationDataAnalyzer
from question_generator import SensemakingQuestionGenerator
from answer_generator import SensemakingAnswerGenerator

def generate_aviation_answers():
    """Generate comprehensive answers for aviation maintenance questions using RAG"""
    
    print("=" * 70)
    print("AVIATION MAINTENANCE SENSEMAKING ANSWER GENERATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verify OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    print("✓ OpenAI API key verified")
    
    # Data paths - using only question generation datasets (excludes KG files)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    # Knowledge graph path
    kg_path = "../kg/knowledge_graph.gml"
    
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA AND KNOWLEDGE GRAPH")
    print("=" * 70)
    
    # Load datasets
    print("Loading datasets for answer generation...")
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    datasets = analyzer.datasets
    
    # Initialize answer generator
    print("Initializing answer generator...")
    answer_generator = SensemakingAnswerGenerator(openai_api_key)
    
    # Load knowledge graph
    print(f"Loading knowledge graph from: {kg_path}")
    kg_loaded = answer_generator.load_knowledge_graph(kg_path)
    if not kg_loaded:
        print("WARNING: Knowledge graph could not be loaded. GraphRAG answers will be skipped.")
    
    print("\n" + "=" * 70)
    print("STEP 2: LOADING OR GENERATING QUESTIONS")
    print("=" * 70)
    
    # Check if we have existing questions
    question_files = [
        "output/aviation_sensemaking_questions_20250625_173716.json"
    ]
    
    questions = None
    questions_source = None
    
    # Try to load existing questions
    for question_file in question_files:
        if os.path.exists(question_file):
            try:
                with open(question_file, 'r') as f:
                    questions = json.load(f)
                questions_source = question_file
                print(f"✓ Loaded {len(questions)} questions from {question_file}")
                break
            except Exception as e:
                print(f"Failed to load {question_file}: {e}")
                continue
    
    # If no existing questions, generate new ones (focusing on action-specific)
    if questions is None:
        print("No existing questions found. Generating new questions...")
        analyzer.analyze_failure_patterns()  # Run basic analysis
        
        question_generator = SensemakingQuestionGenerator(openai_api_key)
        
        # Generate action-specific questions (practical and fast)
        print("Generating action-specific questions...")
        action_questions = question_generator.generate_action_specific_questions(
            analyzer, num_questions=20
        )
        
        # Generate a few global questions
        print("Generating global sensemaking questions...")
        global_questions = question_generator.generate_global_sensemaking_questions(
            analyzer, num_questions=5
        )
        
        questions = action_questions + global_questions
        questions_source = "newly_generated"
        print(f"✓ Generated {len(questions)} new questions")
    
    # Use a sample for testing (to make it faster)
    sample_size = min(10, len(questions))
    sample_questions = questions[:sample_size]
    print(f"Using sample of {len(sample_questions)} questions for answer generation")
    
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING ANSWERS")
    print("=" * 70)
    
    all_results = []
    
    # Generate vanilla LLM answers
    print("Generating vanilla LLM answers...")
    vanilla_answers = answer_generator.generate_vanilla_answers(
        sample_questions, 
        datasets
    )
    print(f"✓ Generated {len(vanilla_answers)} vanilla answers")
    
    # Generate GraphRAG answers (if knowledge graph is loaded)
    graphrag_answers = []
    if kg_loaded:
        print("Generating GraphRAG answers...")
        try:
            graphrag_answers = answer_generator.generate_graphrag_answers(
                sample_questions, 
                datasets
            )
            print(f"✓ Generated {len(graphrag_answers)} GraphRAG answers")
        except Exception as e:
            print(f"ERROR generating GraphRAG answers: {e}")
            print("Continuing without GraphRAG answers...")
    else:
        print("Skipping GraphRAG answers (knowledge graph not loaded)")
    
    print("\n" + "=" * 70)
    print("STEP 4: COMBINING AND SAVING RESULTS")
    print("=" * 70)
    
    # Combine results
    for i, question in enumerate(sample_questions):
        result = {
            'question_id': question.get('id', f'q_{i+1:03d}'),
            'question': question['question'],
            'question_category': question.get('category', 'unknown'),
            'question_type': question.get('type', 'unknown'),
            'vanilla_answer': vanilla_answers[i] if i < len(vanilla_answers) else None,
            'graphrag_answer': graphrag_answers[i] if i < len(graphrag_answers) else None,
            'ground_truth': question.get('ground_truth_answer', None),
            'original_problem': question.get('original_problem', None)
        }
        all_results.append(result)
    
    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/aviation_answers_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    # Save summary CSV for easy review
    summary_data = []
    for result in all_results:
        summary_data.append({
            'question_id': result['question_id'],
            'question': result['question'],
            'category': result['question_category'],
            'vanilla_answer': result['vanilla_answer']['answer'] if result['vanilla_answer'] else '',
            'graphrag_answer': result['graphrag_answer']['answer'] if result['graphrag_answer'] else '',
            'ground_truth': result['ground_truth'] or '',
            'has_ground_truth': bool(result['ground_truth'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_dir}/aviation_answers_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to {summary_file}")
    
    print("\n" + "=" * 70)
    print("STEP 5: SAMPLE RESULTS")
    print("=" * 70)
    
    # Display sample results
    for i, result in enumerate(all_results[:3]):
        print(f"\nQuestion {i+1}:")
        print(f"Q: {result['question']}")
        print(f"Category: {result['question_category']}")
        
        if result['ground_truth']:
            print(f"Ground Truth: {result['ground_truth']}")
        
        if result['vanilla_answer']:
            print(f"Vanilla Answer: {result['vanilla_answer']['answer'][:200]}...")
        
        if result['graphrag_answer']:
            print(f"GraphRAG Answer: {result['graphrag_answer']['answer'][:200]}...")
        
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Total questions processed: {len(all_results)}")
    print(f"Vanilla answers: {len([r for r in all_results if r['vanilla_answer']])}")
    print(f"GraphRAG answers: {len([r for r in all_results if r['graphrag_answer']])}")
    print(f"Questions with ground truth: {len([r for r in all_results if r['ground_truth']])}")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    return all_results

if __name__ == "__main__":
    try:
        results = generate_aviation_answers()
        print("\n✅ Answer generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in answer generation: {e}")
        import traceback
        traceback.print_exc()
