#!/usr/bin/env python3
"""
Generate RAG-based Answers for Aviation Maintenance Sensemaking Questions
Uses both vanilla LLM and GraphRAG approaches with the knowledge graph

Providers for answer generation:
- huggingface / local: preferred open-weight answerers (DoD story)
- anthropic: Claude Messages API (optional closed)
- openai: GPT-5.6 (legacy)

Usage examples:
    # HuggingFace open answerer (preferred for paper)
    python generate_answers.py --question-files questions.json --output-file answers.json \
        --provider huggingface --answer-model Qwen/Qwen3.8-27B --API-provider nebius

    # Legacy OpenAI
    python generate_answers.py --question-files questions.json --output-file answers.json \
        --provider openai --answer-model gpt-5.6-luna
"""

import os
import json
import pandas as pd
import argparse
from datetime import datetime
from data_analyzer import AviationDataAnalyzer
from question_generator import SensemakingQuestionGenerator
from answer_generator import SensemakingAnswerGenerator
import random

def generate_aviation_answers(question_files, output_file, sample_size=None, answer_model=None, kg_path="../kg/output/knowledge_graph.gml", provider="openai", API_provider="nebius", use_mst=True, edge_weight="freq", base_url=None, embedding_backend="local"):
    """Generate comprehensive answers for aviation maintenance questions using RAG"""
    
    print("=" * 70)
    print("AVIATION MAINTENANCE SENSEMAKING ANSWER GENERATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"use_mst={use_mst} edge_weight={edge_weight}")
    print()
    
    # Set default model based on provider if not specified
    if answer_model is None:
        if provider == "openai":
            answer_model = "gpt-5.6-luna"  # legacy
        elif provider == "anthropic":
            from anthropic_client import DEFAULT_ANSWER_MODEL
            answer_model = DEFAULT_ANSWER_MODEL
        elif provider == "local":
            answer_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        elif provider == "huggingface":
            answer_model = "Qwen/Qwen3.8-27B"
    
    # Verify API key based on provider
    if provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        api_key = openai_api_key
        print("✓ OpenAI API key verified (legacy)")
    elif provider == "anthropic":
        anth_key = os.getenv("ANTHROPIC_API_KEY")
        if not anth_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        api_key = anth_key
        print("✓ Anthropic API key verified")
    elif provider == "huggingface":
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        api_key = hf_token
        print("✓ HuggingFace token verified")
    elif provider == "local":
        api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
        print(f"✓ Local LLM → {base_url or os.getenv('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Supported: 'huggingface', 'local', 'anthropic', 'openai' (legacy)"
        )
    
    # Data paths - using only question generation datasets (excludes KG files)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA AND KNOWLEDGE GRAPH")
    print("=" * 70)
    
    # Load datasets
    print("Loading datasets for answer generation...")
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    datasets = analyzer.datasets
    
    # Initialize answer generator
    print(f"Initializing answer generator with provider: {provider}")
    if provider == "huggingface":
        print(f"Using HuggingFace API provider: {API_provider}")
    answer_generator = SensemakingAnswerGenerator(
        api_key,
        model=answer_model,
        provider=provider,
        API_provider=API_provider,
        use_mst=use_mst,
        edge_weight=edge_weight,
        embedding_backend=embedding_backend,
        base_url=base_url,
        use_completion_cache=True,
        max_answer_tokens=600,
    )
    
    # Load knowledge graph
    print(f"Loading knowledge graph from: {kg_path}")
    kg_loaded = False
    if os.path.exists(kg_path):
        kg_loaded = answer_generator.load_knowledge_graph(kg_path)
    if not kg_loaded:
        print("WARNING: Knowledge graph could not be loaded. GraphRAG answers will be skipped.")
    
    print("\n" + "=" * 70)
    print("STEP 2: LOADING OR GENERATING QUESTIONS")
    print("=" * 70)
    
    # Check if we have existing questions
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
        
        question_generator = SensemakingQuestionGenerator(api_key)
        
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
    if sample_size is None:
        sample_questions = questions
        print(f"Using all {len(sample_questions)} questions for answer generation")
    else:
        if sample_size > len(questions):
            print(f"Requested sample size {sample_size} exceeds available questions ({len(questions)}). Using all questions instead.")
            sample_size = len(questions)
        elif sample_size <= 0:
            print(f"Invalid sample size {sample_size}. Using all questions instead.")
            sample_size = len(questions)
        sample_questions = random.sample(questions, sample_size)
        print(f"Using sample of {len(sample_questions)} questions for answer generation")
    
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING ANSWERS")
    print("=" * 70)
    
    all_results = []
    
    # Generate vanilla LLM answers
    print("Generating vanilla LLM answers...")
    vanilla_answers = answer_generator.generate_vanilla_answers(
        sample_questions
    )
    print(f"✓ Generated {len(vanilla_answers)} vanilla answers")
    
    # Generate text-chunk RAG answers
    print("Generating text-chunk RAG answers...")
    textchunkrag_answers = answer_generator.generate_textchunkrag_answers(
        sample_questions, 
        datasets
    )
    print(f"✓ Generated {len(textchunkrag_answers)} text-chunk RAG answers")
    
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
            'textchunkrag_answer': textchunkrag_answers[i] if i < len(textchunkrag_answers) else None,
            'graphrag_answer': graphrag_answers[i] if i < len(graphrag_answers) else None,
            'ground_truth': question.get('ground_truth_answer', None),
            'original_problem': question.get('original_problem', None)
        }
        all_results.append(result)
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")

    # Exact model provenance for reproducibility
    meta_path = os.path.splitext(output_file)[0] + "_meta.json"
    provenance = answer_generator.get_model_provenance()
    with open(meta_path, "w") as f:
        json.dump(
            {
                "provider": provider,
                "API_provider": API_provider if provider == "huggingface" else None,
                "base_url": base_url if provider == "local" else None,
                "answer_model": answer_model,
                "answer_model_exact": answer_generator.exact_model_id(),
                "embedding_backend": embedding_backend,
                "embedding_model": getattr(answer_generator.embedder, "model", None),
                "kg_path": kg_path,
                "use_mst": use_mst,
                "edge_weight": edge_weight,
                "n_questions": len(all_results),
                "timestamp": datetime.now().isoformat(),
                "model_provenance": provenance,
            },
            f,
            indent=2,
        )
    print(
        f"✓ Model provenance: requested={answer_model} "
        f"exact={answer_generator.exact_model_id()} → {meta_path}"
    )
    
    # Save summary CSV for easy review
    summary_data = []
    for result in all_results:
        summary_data.append({
            'question_id': result['question_id'],
            'question': result['question'],
            'category': result['question_category'],
            'type': result['question_type'],
            'vanilla_answer': result['vanilla_answer']['answer'] if result['vanilla_answer'] else '',
            'textchunkrag_answer': result['textchunkrag_answer']['answer'] if result['textchunkrag_answer'] else '',
            'graphrag_answer': result['graphrag_answer']['answer'] if result['graphrag_answer'] else '',
            'ground_truth': result['ground_truth'] or '',
            'has_ground_truth': bool(result['ground_truth'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    # Create summary file in same directory as output file
    base_name = os.path.splitext(output_file)[0]
    summary_file = f"{base_name}_summary.csv"
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
        print(f"Type: {result['question_type']}")
        
        if result['ground_truth']:
            print(f"Ground Truth: {result['ground_truth']}")
        
        if result['vanilla_answer']:
            print(f"Vanilla Answer: {result['vanilla_answer']['answer'][:200]}...")
        
        if result['textchunkrag_answer']:
            print(f"TextChunk RAG Answer: {result['textchunkrag_answer']['answer'][:200]}...")
        
        if result['graphrag_answer']:
            print(f"GraphRAG Answer: {result['graphrag_answer']['answer'][:200]}...")
        
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Total questions processed: {len(all_results)}")
    print(f"Vanilla answers: {len([r for r in all_results if r['vanilla_answer']])}")
    print(f"TextChunk RAG answers: {len([r for r in all_results if r['textchunkrag_answer']])}")
    print(f"GraphRAG answers: {len([r for r in all_results if r['graphrag_answer']])}")
    print(f"Questions with ground truth: {len([r for r in all_results if r['ground_truth']])}")
    print(f"Results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    
    # Clean up and save cache
    answer_generator.cleanup()
    
    return all_results

def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate RAG-based answers for aviation maintenance sensemaking questions"
    )
    parser.add_argument(
        '--question-files', 
        nargs='+', 
        required=True,
        help='Path(s) to question files (JSON format). Required.'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output file for generated answers (JSON format). Required.'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of questions to sample for answer generation. If not provided, uses all questions.'
    )
    parser.add_argument(
        '--answer-model',
        help='Answer model (default: Qwen/Qwen3.8-27B HF; gpt-5.6-luna openai legacy; claude-haiku-4-5 anthropic)'
    )
    parser.add_argument(
        '--kg-path',
        default="../kg/output/knowledge_graph.gml",
        help='Path to knowledge graph file (default: ../kg/output/knowledge_graph.gml)'
    )
    parser.add_argument(
        '--provider',
        default="huggingface",
        choices=["huggingface", "local", "anthropic", "openai"],
        help='Answer provider (default: huggingface; openai=legacy GPT)'
    )
    parser.add_argument(
        '--API-provider',
        default="nebius",
        help='API provider for HuggingFace InferenceClient (only used when provider=huggingface, default: nebius)'
    )
    parser.add_argument(
        '--base-url',
        default=None,
        help='OpenAI-compatible base URL for --provider local',
    )
    parser.add_argument(
        '--embedding-backend',
        default='local',
        choices=['openai', 'local', 'hash'],
    )
    parser.add_argument('--use-mst', dest='use_mst', action='store_true', default=True)
    parser.add_argument('--no-mst', dest='use_mst', action='store_false')
    parser.add_argument('--edge-weight', default='freq', choices=['freq', 'tfidf', 'pmi', 'query', 'rarity'])
    
    args = parser.parse_args()
    
    try:
        results = generate_aviation_answers(
            question_files=args.question_files,
            output_file=args.output_file,
            sample_size=args.sample_size,
            answer_model=args.answer_model,
            kg_path=args.kg_path,
            provider=args.provider,
            API_provider=args.API_provider,
            use_mst=args.use_mst,
            edge_weight=args.edge_weight,
            base_url=args.base_url,
            embedding_backend=args.embedding_backend,
        )
        print("\n✅ Answer generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in answer generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
