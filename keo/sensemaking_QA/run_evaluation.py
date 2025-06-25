#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Aviation Sensemaking QA
Includes action-specific evaluation with ground truth metrics
"""

import os
import json
import pandas as pd
from datetime import datetime
from evaluator import SensemakingEvaluator

def run_comprehensive_evaluation():
    """Run comprehensive evaluation including action-specific ground truth evaluation"""
    
    print("=" * 70)
    print("COMPREHENSIVE AVIATION SENSEMAKING QA EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verify OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    print("✓ OpenAI API key verified")
    
    # Initialize evaluator
    evaluator = SensemakingEvaluator(openai_api_key)
    print("✓ Evaluator initialized")
    
    # Load questions and answers
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    # Load questions
    questions_file = None
    for filename in ["./output/aviation_sensemaking_questions_20250625_173716.json"]:
        try:
            import glob
            files = glob.glob(filename) if '*' in filename else [filename]
            if files:
                questions_file = files[-1]  # Get most recent
                break
        except:
            continue
    
    if not questions_file:
        print("ERROR: No questions file found")
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
    answers_file = None
    for filename in ["./output/aviation_answers_20250625_181025.json"]:
        try:
            files = glob.glob(filename) if '*' in filename else [filename]
            if files:
                answers_file = files[-1]  # Get most recent
                break
        except:
            continue
    
    if not answers_file:
        print("ERROR: No answers file found. Run generate_answers.py first.")
        return
    
    with open(answers_file, 'r') as f:
        answers_data = json.load(f)
    
    # Extract vanilla and GraphRAG answers
    vanilla_answers = []
    graphrag_answers = []
    
    for answer_result in answers_data:
        if answer_result.get('vanilla_answer'):
            vanilla_answers.append({
                'question_id': answer_result['question_id'],
                'answer': answer_result['vanilla_answer']['answer'],
                'method': 'vanilla'
            })
        
        if answer_result.get('graphrag_answer'):
            graphrag_answers.append({
                'question_id': answer_result['question_id'],
                'answer': answer_result['graphrag_answer']['answer'],
                'method': 'graphrag'
            })
    
    print(f"✓ Loaded {len(vanilla_answers)} vanilla answers and {len(graphrag_answers)} GraphRAG answers")
    
    # Create output directory
    output_dir = "./evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize file variables
    action_eval_file = None
    global_eval_file = None
    standard_eval_file = None
    
    print("\n" + "=" * 70)
    print("STEP 2: QUESTION QUALITY EVALUATION")
    print("=" * 70)
    
    # Evaluate question quality (sample)
    sample_questions = questions[:10]  # Evaluate sample for demo
    question_evaluations = evaluator.evaluate_questions(sample_questions)
    
    question_eval_file = f"{output_dir}/question_evaluations_{timestamp}.json"
    evaluator.save_evaluation_results(
        {'question_evaluations': question_evaluations}, 
        question_eval_file
    )
    print(f"✓ Question evaluations saved to {question_eval_file}")
    
    print("\n" + "=" * 70)
    print("STEP 3: ACTION-SPECIFIC EVALUATION WITH GROUND TRUTH")
    print("=" * 70)
    
    # Filter action-specific questions
    action_questions = [q for q in questions if q.get('category') == 'action_specific']
    
    if action_questions:
        print(f"Found {len(action_questions)} action-specific questions with ground truth")
        
        # Evaluate action-specific answers with ground truth metrics
        action_evaluation = evaluator.compare_action_specific_methods(
            vanilla_answers, graphrag_answers, action_questions
        )
        
        if not action_evaluation.get('error'):
            action_eval_file = f"{output_dir}/action_specific_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(action_evaluation, action_eval_file)
            print(f"✓ Action-specific evaluation saved to {action_eval_file}")
            
            # Display action-specific results
            print("\nAction-Specific Evaluation Summary:")
            print("-" * 50)
            
            vanilla_metrics = action_evaluation['vanilla_results']['aggregate_metrics']
            graphrag_metrics = action_evaluation['graphrag_results']['aggregate_metrics']
            winner = action_evaluation['winner']
            
            print(f"Questions Evaluated: {vanilla_metrics.get('total_evaluated', 0)}")
            print("\nNLP Metrics Comparison:")
            print(f"                    Vanilla    GraphRAG   Winner")
            print(f"BLEU Score:         {vanilla_metrics.get('bleu_scores_mean', 0):.3f}      {graphrag_metrics.get('bleu_scores_mean', 0):.3f}      {action_evaluation['comparison'].get('bleu_scores_mean', {}).get('winner', 'N/A')}")
            print(f"METEOR Score:       {vanilla_metrics.get('meteor_scores_mean', 0):.3f}      {graphrag_metrics.get('meteor_scores_mean', 0):.3f}      {action_evaluation['comparison'].get('meteor_scores_mean', {}).get('winner', 'N/A')}")
            print(f"ROUGE-L F1:         {vanilla_metrics.get('rouge_l_f1_scores_mean', 0):.3f}      {graphrag_metrics.get('rouge_l_f1_scores_mean', 0):.3f}      {action_evaluation['comparison'].get('rouge_l_f1_scores_mean', {}).get('winner', 'N/A')}")
            print(f"Exact Match Rate:   {vanilla_metrics.get('exact_matches_rate', 0):.3f}      {graphrag_metrics.get('exact_matches_rate', 0):.3f}      {action_evaluation['comparison'].get('exact_matches_rate', {}).get('winner', 'N/A')}")
            print(f"LLM Evaluation:     {vanilla_metrics.get('llm_evaluation_scores_mean', 0):.3f}      {graphrag_metrics.get('llm_evaluation_scores_mean', 0):.3f}      {action_evaluation['comparison'].get('llm_evaluation_scores_mean', {}).get('winner', 'N/A')}")
            
            print(f"\nOverall Winner: {winner.get('overall_winner', 'Unknown')}")
            print(f"GraphRAG Win Rate: {winner.get('graphrag_win_rate', 0):.1%}")
            print(f"Vanilla Win Rate: {winner.get('vanilla_win_rate', 0):.1%}")
            
        else:
            print("Error in action-specific evaluation:", action_evaluation.get('error'))
    
    else:
        print("No action-specific questions found")
    
    print("\n" + "=" * 70)
    print("STEP 4: GLOBAL SENSEMAKING EVALUATION")
    print("=" * 70)
    
    # Filter global questions
    global_questions = [q for q in questions if q.get('type') == 'global']
    
    if global_questions:
        print(f"Found {len(global_questions)} global sensemaking questions")
        
        # Evaluate global sensemaking capability
        global_evaluation = evaluator.evaluate_global_sensemaking_capability(
            global_questions[:5], graphrag_answers  # Sample for demo
        )
        
        if not global_evaluation.get('error'):
            global_eval_file = f"{output_dir}/global_sensemaking_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(global_evaluation, global_eval_file)
            print(f"✓ Global sensemaking evaluation saved to {global_eval_file}")
            
            global_metrics = global_evaluation.get('global_metrics', {})
            print(f"\nGlobal Sensemaking Results:")
            print(f"Questions Evaluated: {global_evaluation.get('global_questions_evaluated', 0)}")
            if global_metrics:
                print(f"Average Scores:")
                print(f"  Global Perspective: {global_metrics.get('global_perspective', 0):.2f}/5.0")
                print(f"  Theme Identification: {global_metrics.get('theme_identification', 0):.2f}/5.0")
                print(f"  Synthesis Quality: {global_metrics.get('synthesis_quality', 0):.2f}/5.0")
                print(f"  Overall Global Score: {global_metrics.get('overall_global_score', 0):.2f}/5.0")
        else:
            print("Error in global evaluation:", global_evaluation.get('error'))
    
    else:
        print("No global sensemaking questions found")
    
    print("\n" + "=" * 70)
    print("STEP 5: STANDARD METHOD COMPARISON")
    print("=" * 70)
    
    # Standard comparison for non-action questions
    non_action_questions = [q for q in questions 
                           if q.get('category') != 'action_specific' and q.get('type') != 'global']
    
    if non_action_questions:
        print(f"Comparing methods on {len(non_action_questions[:10])} standard questions")
        
        standard_comparison = evaluator.compare_answer_methods(
            vanilla_answers, graphrag_answers, non_action_questions[:10]  # Sample
        )
        
        standard_eval_file = f"{output_dir}/standard_method_comparison_{timestamp}.json"
        evaluator.save_evaluation_results(standard_comparison, standard_eval_file)
        print(f"✓ Standard method comparison saved to {standard_eval_file}")
        
        print("\nStandard Evaluation Summary:")
        print(standard_comparison.get('evaluation_summary', 'No summary available'))
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    
    print("Generated Files:")
    print(f"  - {question_eval_file}")
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
    print(f"  GraphRAG Answers: {len(graphrag_answers)}")


if __name__ == "__main__":
    run_comprehensive_evaluation()
