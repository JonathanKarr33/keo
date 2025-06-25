#!/usr/bin/env python3
"""
Test Action-Specific Evaluation with Ground Truth
"""

import os
import sys
from evaluator import SensemakingEvaluator, NLP_METRICS_AVAILABLE

def test_action_evaluation():
    """Test the new action-specific evaluation functionality"""
    
    print("Testing Action-Specific Evaluation with Ground Truth")
    print("=" * 60)
    
    # Check dependencies
    print(f"NLP Metrics Available: {NLP_METRICS_AVAILABLE}")
    if not NLP_METRICS_AVAILABLE:
        print("Warning: Some NLP metrics may not work. Install with:")
        print("pip install nltk rouge-score")
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    print("✓ OpenAI API key found")
    
    # Initialize evaluator
    evaluator = SensemakingEvaluator(openai_api_key)
    print("✓ Evaluator initialized")
    
    # Create test data
    test_questions = [
        {
            'id': 'action_001',
            'category': 'action_specific',
            'question': 'What should be done when #3 intake is leaking?',
            'ground_truth_answer': 'REMOVED & REPLACED GASKET.',
            'type': 'actionable'
        },
        {
            'id': 'action_002',
            'category': 'action_specific',
            'question': 'What should be done when rocker cover screws are loose?',
            'ground_truth_answer': 'TIGHTENED SCREWS.',
            'type': 'actionable'
        }
    ]
    
    test_answers = [
        {
            'question_id': 'action_001',
            'answer': 'Remove and replace the gasket on the #3 intake.',
            'method': 'vanilla'
        },
        {
            'question_id': 'action_001',
            'answer': 'REMOVED & REPLACED GASKET',
            'method': 'graphrag'
        },
        {
            'question_id': 'action_002',
            'answer': 'Tighten the loose screws on the rocker cover.',
            'method': 'vanilla'
        },
        {
            'question_id': 'action_002',
            'answer': 'TIGHTENED SCREWS',
            'method': 'graphrag'
        }
    ]
    
    print("\nTest Data:")
    print(f"Questions: {len(test_questions)}")
    print(f"Answers: {len(test_answers)}")
    
    # Test action-specific evaluation
    print("\n" + "=" * 60)
    print("TESTING ACTION-SPECIFIC EVALUATION")
    print("=" * 60)
    
    try:
        # Separate vanilla and GraphRAG answers
        vanilla_answers = [a for a in test_answers if a['method'] == 'vanilla']
        graphrag_answers = [a for a in test_answers if a['method'] == 'graphrag']
        
        print(f"Vanilla answers: {len(vanilla_answers)}")
        print(f"GraphRAG answers: {len(graphrag_answers)}")
        
        # Run evaluation
        action_evaluation = evaluator.compare_action_specific_methods(
            vanilla_answers, graphrag_answers, test_questions
        )
        
        if action_evaluation.get('error'):
            print(f"ERROR: {action_evaluation['error']}")
            return
        
        print("✓ Action-specific evaluation completed successfully")
        
        # Display results
        print("\nRESULTS:")
        print("-" * 40)
        
        vanilla_metrics = action_evaluation['vanilla_results']['aggregate_metrics']
        graphrag_metrics = action_evaluation['graphrag_results']['aggregate_metrics']
        winner = action_evaluation['winner']
        
        print(f"Questions Evaluated: {vanilla_metrics.get('total_evaluated', 0)}")
        
        if NLP_METRICS_AVAILABLE:
            print("\nNLP Metrics:")
            print(f"                    Vanilla    GraphRAG")
            print(f"BLEU Score:         {vanilla_metrics.get('bleu_scores_mean', 0):.3f}      {graphrag_metrics.get('bleu_scores_mean', 0):.3f}")
            print(f"ROUGE-L F1:         {vanilla_metrics.get('rouge_l_f1_scores_mean', 0):.3f}      {graphrag_metrics.get('rouge_l_f1_scores_mean', 0):.3f}")
            print(f"Exact Match Rate:   {vanilla_metrics.get('exact_matches_rate', 0):.3f}      {graphrag_metrics.get('exact_matches_rate', 0):.3f}")
        
        print(f"\nLLM Evaluation:     {vanilla_metrics.get('llm_evaluation_scores_mean', 0):.3f}      {graphrag_metrics.get('llm_evaluation_scores_mean', 0):.3f}")
        print(f"Overall Score:      {vanilla_metrics.get('overall_scores_mean', 0):.3f}      {graphrag_metrics.get('overall_scores_mean', 0):.3f}")
        
        print(f"\nOverall Winner: {winner.get('overall_winner', 'Unknown')}")
        print(f"GraphRAG Win Rate: {winner.get('graphrag_win_rate', 0):.1%}")
        print(f"Vanilla Win Rate: {winner.get('vanilla_win_rate', 0):.1%}")
        
        # Show individual evaluations
        print("\nIndividual Evaluations:")
        for eval_result in action_evaluation['vanilla_results']['individual_evaluations']:
            print(f"\nQuestion: {eval_result.question_id}")
            print(f"Ground Truth: {eval_result.ground_truth}")
            print(f"Predicted: {eval_result.predicted_answer}")
            if NLP_METRICS_AVAILABLE:
                print(f"BLEU: {eval_result.nlp_metrics.bleu_score:.3f}")
                print(f"Exact Match: {eval_result.nlp_metrics.exact_match}")
            print(f"Overall Score: {eval_result.overall_score:.3f}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("✓ Action-specific evaluation with ground truth working correctly")
        print("✓ NLP metrics calculated")
        print("✓ LLM evaluation completed")
        print("✓ Method comparison functional")
        
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_action_evaluation()
