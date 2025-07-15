#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Aviation Sensemaking QA
Includes action-specific evaluation with ground truth metrics

Supports both OpenAI and HuggingFace providers for evaluation:
- OpenAI: Uses OpenAI's GPT models (requires OPENAI_API_KEY)
- HuggingFace: Uses HuggingFace's InferenceClient (requires HF_TOKEN)

Usage examples:
    # Using OpenAI (default)
    python run_evaluation.py --questions-file questions.json --answers-file answers.json --output-dir results

    # Using HuggingFace
    python run_evaluation.py --questions-file questions.json --answers-file answers.json --output-dir results \
        --provider huggingface --evaluation-model google/gemma-3-4b-it
"""

import os
import json
import pandas as pd
import argparse
from datetime import datetime
from evaluator import SensemakingEvaluator

def run_comprehensive_evaluation(questions_file, answers_file, output_dir, evaluation_model=None, provider="openai", API_provider="featherless-ai"):
    """Run comprehensive evaluation including action-specific ground truth evaluation"""
    
    print("=" * 70)
    print("COMPREHENSIVE AVIATION SENSEMAKING QA EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set default model based on provider if not specified
    if evaluation_model is None:
        if provider == "openai":
            evaluation_model = "gpt-4o"
        elif provider == "huggingface":
            evaluation_model = "google/gemma-3-4b-it"
    
    # Verify API key based on provider
    if provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        api_key = openai_api_key
        print("✓ OpenAI API key verified")
    elif provider == "huggingface":
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        api_key = hf_token
        print("✓ HuggingFace token verified")
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'openai', 'huggingface'")
    
    # Initialize evaluator
    print(f"Initializing evaluator with provider: {provider}")
    if provider == "huggingface":
        print(f"Using HuggingFace API provider: {API_provider}")
    evaluator = SensemakingEvaluator(api_key, model=evaluation_model, provider=provider, API_provider=API_provider)
    print("✓ Evaluator initialized")
    
    # Load questions and answers
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    # Load questions
    if not os.path.exists(questions_file):
        print(f"ERROR: Questions file not found: {questions_file}")
        return
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"✓ Loaded {len(questions)} questions from {questions_file}")
    
    # Categorize questions
    question_categories = {}
    for q in questions:
        category = q.get('category', 'unknown')
        question_categories[category] = question_categories.get(category, 0) + 1
    
    print("Question categories:")
    for category, count in question_categories.items():
        print(f"  - {category}: {count}")
    
    # Load answers
    if not os.path.exists(answers_file):
        print(f"ERROR: Answers file not found: {answers_file}")
        return
    
    with open(answers_file, 'r') as f:
        answers_data = json.load(f)
    
    # Extract vanilla, text-chunk RAG, and GraphRAG answers
    vanilla_answers = []
    textchunkrag_answers = []
    graphrag_answers = []
    
    for answer_result in answers_data:
        if answer_result.get('vanilla_answer'):
            vanilla_answers.append({
                'question_id': answer_result['question_id'],
                'answer': answer_result['vanilla_answer']['answer'],
                'method': 'vanilla'
            })
        
        if answer_result.get('textchunkrag_answer'):
            textchunkrag_answers.append({
                'question_id': answer_result['question_id'],
                'answer': answer_result['textchunkrag_answer']['answer'],
                'method': 'textchunkrag'
            })
        
        if answer_result.get('graphrag_answer'):
            graphrag_answers.append({
                'question_id': answer_result['question_id'],
                'answer': answer_result['graphrag_answer']['answer'],
                'method': 'graphrag'
            })
    
    print(f"✓ Loaded {len(vanilla_answers)} vanilla answers, {len(textchunkrag_answers)} text-chunk RAG answers, and {len(graphrag_answers)} GraphRAG answers")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize file variables
    action_eval_file = None
    global_eval_file = None
    standard_eval_file = None
    
    # print("\n" + "=" * 70)
    # print("STEP 2: QUESTION QUALITY EVALUATION")
    # print("=" * 70)
    
    # Get unique question IDs from answers file
    answered_question_ids = set()
    for answer_result in answers_data:
        answered_question_ids.add(answer_result['question_id'])
    
    # Filter questions to only those that have been answered
    answered_questions = [q for q in questions if q.get('id') in answered_question_ids]
    print(f"Found {len(answered_questions)} questions that have been answered out of {len(questions)} total questions")
    
    # # Evaluate question quality for answered questions only
    # question_evaluations = evaluator.evaluate_questions(answered_questions)
    
    # question_eval_file = f"{output_dir}/question_evaluations_{timestamp}.json"
    # evaluator.save_evaluation_results(
    #     {'question_evaluations': question_evaluations}, 
    #     question_eval_file
    # )
    # print(f"✓ Question evaluations saved to {question_eval_file}")
    
    print("\n" + "=" * 70)
    print("STEP 3: ACTION-SPECIFIC EVALUATION WITH GROUND TRUTH")
    print("=" * 70)
    
    # Filter action-specific questions that have been answered
    action_questions = [q for q in questions if q.get('category') == 'action_specific' and q.get('id') in answered_question_ids]
    
    if action_questions:
        print(f"Found {len(action_questions)} answered action-specific questions with ground truth")
        
        # Evaluate action-specific answers with ground truth metrics - Three-way comparison
        print("Performing three-way comparison: Vanilla vs Text-chunk RAG vs GraphRAG")
        
        # Vanilla vs Text-chunk RAG
        vanilla_vs_textchunk = evaluator.compare_action_methods_flexible(
            vanilla_answers, textchunkrag_answers, action_questions,
            "vanilla", "textchunkrag"
        )
        
        # Vanilla vs GraphRAG
        vanilla_vs_graphrag = evaluator.compare_action_methods_flexible(
            vanilla_answers, graphrag_answers, action_questions,
            "vanilla", "graphrag"
        )
        
        # Text-chunk RAG vs GraphRAG
        textchunk_vs_graphrag = evaluator.compare_action_methods_flexible(
            textchunkrag_answers, graphrag_answers, action_questions,
            "textchunkrag", "graphrag"
        )
        
        # Combine results
        action_evaluation = {
            'vanilla_vs_textchunk': vanilla_vs_textchunk,
            'vanilla_vs_graphrag': vanilla_vs_graphrag,
            'textchunk_vs_graphrag': textchunk_vs_graphrag,
            'timestamp': timestamp
        }
        # Debug: Check which comparison failed
        print("Checking comparison results...")
        for name, comparison in [('vanilla_vs_textchunk', vanilla_vs_textchunk), 
                                ('vanilla_vs_graphrag', vanilla_vs_graphrag), 
                                ('textchunk_vs_graphrag', textchunk_vs_graphrag)]:
            if comparison.get('error'):
                print(f"❌ {name} failed: {comparison.get('error')}")
            else:
                print(f"✓ {name} succeeded")
        if not any(comparison.get('error') for comparison in [vanilla_vs_textchunk, vanilla_vs_graphrag, textchunk_vs_graphrag]):
            action_eval_file = f"{output_dir}/action_specific_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(action_evaluation, action_eval_file)
            print(f"✓ Action-specific evaluation saved to {action_eval_file}")
            
            # Display comprehensive action-specific results
            print("\nAction-Specific Evaluation Summary:")
            print("=" * 50)
            
            # Extract metrics for each method
            vanilla_metrics = vanilla_vs_textchunk['vanilla_results']['aggregate_metrics']
            textchunk_metrics = vanilla_vs_textchunk['textchunkrag_results']['aggregate_metrics']
            graphrag_metrics = vanilla_vs_graphrag['graphrag_results']['aggregate_metrics']
            
            print(f"Questions Evaluated: {vanilla_metrics.get('total_evaluated', 0)}")
            print("\nThree-Way NLP Metrics Comparison:")
            print(f"                    Vanilla    TextChunk  GraphRAG")
            print(f"BLEU Score:         {vanilla_metrics.get('bleu_scores_mean', 0):.3f}      {textchunk_metrics.get('bleu_scores_mean', 0):.3f}      {graphrag_metrics.get('bleu_scores_mean', 0):.3f}")
            print(f"METEOR Score:       {vanilla_metrics.get('meteor_scores_mean', 0):.3f}      {textchunk_metrics.get('meteor_scores_mean', 0):.3f}      {graphrag_metrics.get('meteor_scores_mean', 0):.3f}")
            print(f"ROUGE-L F1:         {vanilla_metrics.get('rouge_l_f1_scores_mean', 0):.3f}      {textchunk_metrics.get('rouge_l_f1_scores_mean', 0):.3f}      {graphrag_metrics.get('rouge_l_f1_scores_mean', 0):.3f}")
            print(f"Exact Match Rate:   {vanilla_metrics.get('exact_matches_rate', 0):.3f}      {textchunk_metrics.get('exact_matches_rate', 0):.3f}      {graphrag_metrics.get('exact_matches_rate', 0):.3f}")
            print(f"LLM Evaluation:     {vanilla_metrics.get('llm_evaluation_scores_mean', 0):.3f}      {textchunk_metrics.get('llm_evaluation_scores_mean', 0):.3f}      {graphrag_metrics.get('llm_evaluation_scores_mean', 0):.3f}")
            
            # Determine overall best method
            methods = {
                'Vanilla': vanilla_metrics,
                'TextChunk RAG': textchunk_metrics, 
                'GraphRAG': graphrag_metrics
            }
            
            best_method = max(methods.items(), key=lambda x: x[1].get('llm_evaluation_scores_mean', 0))
            print(f"\nBest Performing Method: {best_method[0]} (LLM Score: {best_method[1].get('llm_evaluation_scores_mean', 0):.3f})")
            
            # Show pairwise winners
            print("\nPairwise Comparisons:")
            print(f"Vanilla vs TextChunk: {vanilla_vs_textchunk.get('winner', {}).get('overall_winner', 'N/A')}")
            print(f"Vanilla vs GraphRAG: {vanilla_vs_graphrag.get('winner', {}).get('overall_winner', 'N/A')}")
            print(f"TextChunk vs GraphRAG: {textchunk_vs_graphrag.get('winner', {}).get('overall_winner', 'N/A')}")
            
        else:
            print("Error in action-specific evaluation - some comparisons failed")
    
    else:
        print("No action-specific questions found")
    
    print("\n" + "=" * 70)
    print("STEP 4: OTHER SENSEMAKING QUESTION EVALUATION")
    print("=" * 70)
    
    # Filter global questions that have been answered
    global_questions = [q for q in questions if q.get('id') in answered_question_ids and q.get('category') != 'action_specific']
    
    if global_questions:
        print(f"Found {len(global_questions)} answered global sensemaking questions")
        
        # Evaluate global sensemaking capability for all three methods
        print(f"Evaluating all {len(global_questions)} global questions")
        
        # Evaluate Vanilla LLM
        vanilla_global = evaluator.evaluate_global_sensemaking_capability(
            global_questions, vanilla_answers
        )
        
        # Evaluate Text-chunk RAG
        textchunk_global = evaluator.evaluate_global_sensemaking_capability(
            global_questions, textchunkrag_answers
        )
        
        # Evaluate GraphRAG
        graphrag_global = evaluator.evaluate_global_sensemaking_capability(
            global_questions, graphrag_answers
        )
        
        # Combine global evaluation results
        global_evaluation = {
            'vanilla_global': vanilla_global,
            'textchunk_global': textchunk_global,
            'graphrag_global': graphrag_global,
            'timestamp': timestamp
        }
        
        if not any(eval_result.get('error') for eval_result in [vanilla_global, textchunk_global, graphrag_global]):
            print(f"\nGlobal Sensemaking Results (3-way comparison):")
            print("=" * 60)
            print(f"Questions Evaluated: {len(global_questions)}")
            print()
            
            # Compare metrics across all three methods
            methods_global = {
                'Vanilla': vanilla_global.get('global_metrics', {}),
                'TextChunk RAG': textchunk_global.get('global_metrics', {}),
                'GraphRAG': graphrag_global.get('global_metrics', {})
            }

            # Calculate overall average scores
            vanilla_score_sum = 0
            textchunk_score_sum = 0
            graphrag_score_sum = 0
            for metric in ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition']:
                vanilla_score_sum += methods_global['Vanilla'].get(metric, 0)
                textchunk_score_sum += methods_global['TextChunk RAG'].get(metric, 0)
                graphrag_score_sum += methods_global['GraphRAG'].get(metric, 0)
            methods_global['Vanilla']['overall_avg_score'] = vanilla_score_sum / 5
            methods_global['TextChunk RAG']['overall_avg_score'] = textchunk_score_sum / 5
            methods_global['GraphRAG']['overall_avg_score'] = graphrag_score_sum / 5
            
            print(f"                     Vanilla    TextChunk  GraphRAG")
            for metric in ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition', 'overall_avg_score']:
                vanilla_score = methods_global['Vanilla'].get(metric, 0)
                textchunk_score = methods_global['TextChunk RAG'].get(metric, 0)
                graphrag_score = methods_global['GraphRAG'].get(metric, 0)
                
                metric_name = metric.replace('_', ' ').title()[:19]
                print(f"{metric_name:<20} {vanilla_score:.2f}       {textchunk_score:.2f}       {graphrag_score:.2f}")
            
            # Perform pairwise comparison for global sensemaking
            print("\nPairwise Comparison Analysis:")
            print("=" * 50)
            
            # Get individual question results for pairwise comparison
            vanilla_results = vanilla_global.get("individual_evaluations", [])
            textchunk_results = textchunk_global.get("individual_evaluations", [])
            graphrag_results = graphrag_global.get("individual_evaluations", [])
            
            # Create dictionaries for easy lookup by question_id
            vanilla_dict = {result['question_id']: result for result in vanilla_results}
            textchunk_dict = {result['question_id']: result for result in textchunk_results}
            graphrag_dict = {result['question_id']: result for result in graphrag_results}
            
            # Define method pairs for comparison
            method_pairs = [
                ('Vanilla', 'TextChunk RAG', vanilla_dict, textchunk_dict),
                ('Vanilla', 'GraphRAG', vanilla_dict, graphrag_dict),
                ('TextChunk RAG', 'GraphRAG', textchunk_dict, graphrag_dict)
            ]
            
            comparison_results = {}
            
            for method1_name, method2_name, method1_dict, method2_dict in method_pairs:
                # Initialize counters for total score comparison
                wins_method1_total = 0
                wins_method2_total = 0
                ties_total = 0
                total_compared = 0
                
                # Initialize counters for individual metric comparisons
                metric_comparisons = {}
                for metric in ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition']:
                    metric_comparisons[metric] = {
                        'method1_wins': 0,
                        'method2_wins': 0,
                        'ties': 0
                    }
                
                # Compare each question across the two methods
                for question_id in set(method1_dict.keys()) & set(method2_dict.keys()):
                    method1_result = method1_dict[question_id]
                    method2_result = method2_dict[question_id]
                    
                    # Compare across all metrics (metrics are nested under 'global_metrics')
                    method1_total_score = 0
                    method2_total_score = 0
                    
                    for metric in ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition']:
                        method1_metric_score = method1_result.get('global_metrics', {}).get(metric, 0)
                        method2_metric_score = method2_result.get('global_metrics', {}).get(metric, 0)
                        
                        # Add to total score
                        method1_total_score += method1_metric_score
                        method2_total_score += method2_metric_score
                        
                        # Individual metric comparison
                        if method1_metric_score > method2_metric_score:
                            metric_comparisons[metric]['method1_wins'] += 1
                        elif method2_metric_score > method1_metric_score:
                            metric_comparisons[metric]['method2_wins'] += 1
                        else:
                            metric_comparisons[metric]['ties'] += 1
                    
                    # Determine winner for this question based on total score
                    if method1_total_score > method2_total_score:
                        wins_method1_total += 1
                    elif method2_total_score > method1_total_score:
                        wins_method2_total += 1
                    else:
                        ties_total += 1
                    
                    total_compared += 1
                
                # Calculate win rates for total score (excluding ties)
                total_wins = wins_method1_total + wins_method2_total
                if total_wins > 0:
                    winrate_method1_total = (wins_method1_total / total_wins) * 100
                    winrate_method2_total = (wins_method2_total / total_wins) * 100
                else:
                    winrate_method1_total = winrate_method2_total = 0
                
                # Calculate win rates for individual metrics
                for metric in metric_comparisons:
                    metric_total_wins = metric_comparisons[metric]['method1_wins'] + metric_comparisons[metric]['method2_wins']
                    if metric_total_wins > 0:
                        metric_comparisons[metric]['method1_winrate'] = (metric_comparisons[metric]['method1_wins'] / metric_total_wins) * 100
                        metric_comparisons[metric]['method2_winrate'] = (metric_comparisons[metric]['method2_wins'] / metric_total_wins) * 100
                    else:
                        metric_comparisons[metric]['method1_winrate'] = 0
                        metric_comparisons[metric]['method2_winrate'] = 0
                
                # Store results (including both total score and individual metric comparisons)
                comparison_results[f"{method1_name}_vs_{method2_name}"] = {
                    'total_score_comparison': {
                        'method1_wins': wins_method1_total,
                        'method2_wins': wins_method2_total,
                        'ties': ties_total,
                        'total_compared': total_compared,
                        'method1_winrate': winrate_method1_total,
                        'method2_winrate': winrate_method2_total
                    },
                    'individual_metric_comparisons': metric_comparisons
                }
                
                # Display results (only show total score comparison)
                print(f"\n{method1_name} vs {method2_name}:")
                print(f"  Questions Compared: {total_compared}")
                print(f"  {method1_name} Wins: {wins_method1_total} ({winrate_method1_total:.1f}%)")
                print(f"  {method2_name} Wins: {wins_method2_total} ({winrate_method2_total:.1f}%)")
                print(f"  Ties: {ties_total}")
            
            # Add pairwise comparison results to global evaluation
            global_evaluation['pairwise_comparison'] = comparison_results
            
            # Determine best method for global sensemaking
            best_global = max(methods_global.items(), 
                            key=lambda x: x[1].get('overall_avg_score', 0))
            print(f"\nBest Global Sensemaking Method: {best_global[0]} (Avg Score: {best_global[1].get('overall_avg_score', 0):.2f}/5.0)")
            
            # Save global evaluation results (including pairwise comparison)
            global_eval_file = f"{output_dir}/global_sensemaking_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(global_evaluation, global_eval_file)
            print(f"✓ Global sensemaking evaluation saved to {global_eval_file}")
            
        else:
            print("Error in global evaluation: Some methods failed")
    
    else:
        print("No global sensemaking questions found")
    
    print("\n" + "=" * 70)
    print("STEP 5: STANDARD METHOD COMPARISON")
    print("=" * 70)
    
    # Standard comparison for non-action questions that have been answered
    non_action_questions = [q for q in questions 
                           if q.get('category') != 'action_specific' and q.get('type') != 'global' and q.get('question_id') in answered_question_ids]
    
    if non_action_questions:
        print(f"Comparing methods on all {len(non_action_questions)} answered standard questions")
        
        # Three-way standard comparison
        vanilla_vs_textchunk_std = evaluator.compare_answer_methods(
            vanilla_answers, textchunkrag_answers, non_action_questions,
            "vanilla", "textchunkrag"
        )
        
        vanilla_vs_graphrag_std = evaluator.compare_answer_methods(
            vanilla_answers, graphrag_answers, non_action_questions,
            "vanilla", "graphrag"
        )
        
        textchunk_vs_graphrag_std = evaluator.compare_answer_methods(
            textchunkrag_answers, graphrag_answers, non_action_questions,
            "textchunkrag", "graphrag"
        )
        
        standard_comparison = {
            'vanilla_vs_textchunk': vanilla_vs_textchunk_std,
            'vanilla_vs_graphrag': vanilla_vs_graphrag_std,
            'textchunk_vs_graphrag': textchunk_vs_graphrag_std,
            'timestamp': timestamp,
            'questions_evaluated': len(non_action_questions)
        }
        
        standard_eval_file = f"{output_dir}/standard_method_comparison_{timestamp}.json"
        evaluator.save_evaluation_results(standard_comparison, standard_eval_file)
        print(f"✓ Standard method comparison saved to {standard_eval_file}")
        
        print("\nStandard Evaluation Summary:")
        print("=" * 50)
        print(f"Questions Evaluated: {len(non_action_questions)}")
        print("\nPairwise Standard Comparisons:")
        print(f"Vanilla vs TextChunk: {vanilla_vs_textchunk_std.get('evaluation_summary', 'N/A')}")
        print(f"Vanilla vs GraphRAG: {vanilla_vs_graphrag_std.get('evaluation_summary', 'N/A')}")
        print(f"TextChunk vs GraphRAG: {textchunk_vs_graphrag_std.get('evaluation_summary', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    
    print("Generated Files:")
    #print(f"  - {question_eval_file}")
    if action_eval_file:
        print(f"  - {action_eval_file}")
    if global_eval_file:
        print(f"  - {global_eval_file}")
    if standard_eval_file:
        print(f"  - {standard_eval_file}")
    
    print("\nEvaluation Summary:")
    print(f"  Total Questions: {len(questions)}")
    print(f"  Action-Specific (w/ Ground Truth): {len(action_questions) if 'action_questions' in locals() else 0}")
    print(f"  Global Sensemaking: {len(global_questions) if 'global_questions' in locals() else 0}")
    print(f"  Standard Questions: {len(non_action_questions) if 'non_action_questions' in locals() else 0}")
    print(f"  Vanilla Answers: {len(vanilla_answers)}")
    print(f"  TextChunk RAG Answers: {len(textchunkrag_answers)}")
    print(f"  GraphRAG Answers: {len(graphrag_answers)}")
    print("\nThree-way evaluation completed across all question types!")

def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation for aviation sensemaking QA"
    )
    parser.add_argument(
        '--questions-file',
        required=True,
        help='Path to questions file (JSON format). Required.'
    )
    parser.add_argument(
        '--answers-file',
        required=True,
        help='Path to answers file (JSON format). Required.'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for evaluation results. Required.'
    )
    parser.add_argument(
        '--evaluation-model',
        help='Model to use for evaluation (default: gpt-4o for OpenAI, google/gemma-3-4b-it for HuggingFace)'
    )
    parser.add_argument(
        '--provider',
        default="openai",
        choices=["openai", "huggingface"],
        help='Provider for evaluation (default: openai)'
    )
    parser.add_argument(
        '--API-provider',
        default="featherless-ai",
        help='API provider for HuggingFace InferenceClient (only used when provider=huggingface, default: featherless-ai)'
    )
    
    args = parser.parse_args()
    
    try:
        run_comprehensive_evaluation(
            questions_file=args.questions_file,
            answers_file=args.answers_file,
            output_dir=args.output_dir,
            evaluation_model=args.evaluation_model,
            provider=args.provider,
            API_provider=args.API_provider
        )
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
