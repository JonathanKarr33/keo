#!/usr/bin/env python3
"""
Quantitative Evaluation Script for Action-Specific Questions
This script performs comprehensive quantitative evaluation of action-specific answers
using NLP metrics (BLEU, METEOR, ROUGE) and ground truth comparisons.
"""

import os
import json
import re
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import subprocess
import sys

# Install required packages if not available
def install_required_packages():
    """Install required packages for NLP metrics"""
    packages = ['nltk', 'rouge-score', 'pandas', 'numpy', 'tqdm']
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages first
install_required_packages()

# Now import the packages
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading NLTK omw-1.4...")
        nltk.download('omw-1.4')

download_nltk_data()

@dataclass
class ActionSpecificMetrics:
    """Structure for action-specific evaluation metrics with ground truth"""
    bleu_score: float
    meteor_score: float
    rouge_l_f1: float
    rouge_1_f1: float
    rouge_2_f1: float
    semantic_similarity: float
    exact_match: bool
    explanation: str

@dataclass
class MethodEvaluation:
    """Evaluation results for a specific method"""
    method_name: str
    total_questions: int
    avg_bleu: float
    avg_meteor: float
    avg_rouge_l: float
    avg_rouge_1: float
    avg_rouge_2: float
    avg_semantic_sim: float
    exact_match_rate: float
    individual_results: List[Dict]

class QuantitativeActionEvaluator:
    """Quantitative evaluator for action-specific questions"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_func = SmoothingFunction().method1
        
    def load_questions(self, questions_file: str) -> Dict[str, Dict]:
        """Load questions and filter for action-specific ones with ground truth"""
        print(f"Loading questions from {questions_file}...")
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Filter for action-specific questions with ground truth
        action_questions = {}
        for question in questions:
            if (question.get('category') == 'action_specific' and 
                question.get('ground_truth_answer')):
                action_questions[question['id']] = question
        
        print(f"Found {len(action_questions)} action-specific questions with ground truth")
        return action_questions
    
    def load_answer_files(self, base_dir: str, kg_dir: str) -> Dict[str, List[Dict]]:
        """Load all answer files from both directories"""
        print("Loading answer files...")
        
        all_files = {}
        
        # Load from base directory
        base_files = glob.glob(os.path.join(base_dir, "answers_*.json"))
        for file_path in base_files:
            filename = os.path.basename(file_path)
            method_name = self._extract_method_name(filename, "base")
            all_files[method_name] = self._load_json_file(file_path)
        
        # Load from kg directory
        kg_files = glob.glob(os.path.join(kg_dir, "answers_*.json"))
        for file_path in kg_files:
            filename = os.path.basename(file_path)
            method_name = self._extract_method_name(filename, "kg")
            all_files[method_name] = self._load_json_file(file_path)
        
        print(f"Loaded {len(all_files)} answer files")
        return all_files
    
    def _extract_method_name(self, filename: str, prefix: str) -> str:
        """Extract method name from filename"""
        # Remove .json extension
        name = filename.replace('.json', '')
        # Remove answers_ prefix
        name = name.replace('answers_', '')
        # Add prefix to distinguish base vs kg methods
        return f"{prefix}_{name}"
    
    def _load_json_file(self, file_path: str) -> List[Dict]:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP metrics"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common maintenance text artifacts but keep meaningful punctuation
        text = re.sub(r'[^\w\s&.,;:()-]', ' ', text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def calculate_word_overlap_f1(self, predicted_tokens: List[str], ground_truth_tokens: List[str]) -> float:
        """Calculate F1 score based on word overlap"""
        if not predicted_tokens or not ground_truth_tokens:
            return 0.0
        
        predicted_set = set(token.lower() for token in predicted_tokens)
        ground_truth_set = set(token.lower() for token in ground_truth_tokens)
        
        intersection = predicted_set & ground_truth_set
        
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(predicted_set)
        recall = len(intersection) / len(ground_truth_set)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def calculate_nlp_metrics(self, predicted: str, ground_truth: str) -> ActionSpecificMetrics:
        """Calculate comprehensive NLP metrics for action-specific answers"""
        
        # Preprocess text
        predicted_clean = self.preprocess_text(predicted)
        ground_truth_clean = self.preprocess_text(ground_truth)
        
        # Handle empty strings
        if not predicted_clean or not ground_truth_clean:
            return ActionSpecificMetrics(
                bleu_score=0.0,
                meteor_score=0.0,
                rouge_l_f1=0.0,
                rouge_1_f1=0.0,
                rouge_2_f1=0.0,
                semantic_similarity=0.0,
                exact_match=False,
                explanation="Empty predicted or ground truth text"
            )
        
        # Exact match (case-insensitive)
        exact_match = predicted_clean.lower() == ground_truth_clean.lower()
        
        # Tokenize
        predicted_tokens = predicted_clean.split()
        ground_truth_tokens = ground_truth_clean.split()
        
        # BLEU score
        try:
            bleu = sentence_bleu([ground_truth_tokens], predicted_tokens, 
                               smoothing_function=self.smoothing_func)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            bleu = 0.0
        
        # METEOR score
        try:
            meteor = meteor_score([ground_truth_clean], predicted_clean)
        except Exception as e:
            print(f"METEOR calculation error: {e}")
            meteor = 0.0
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge_scorer.score(ground_truth_clean, predicted_clean)
            rouge_1_f1 = rouge_scores['rouge1'].fmeasure
            rouge_2_f1 = rouge_scores['rouge2'].fmeasure
            rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            rouge_1_f1 = rouge_2_f1 = rouge_l_f1 = 0.0
        
        # Semantic similarity (word overlap F1)
        semantic_sim = self.calculate_word_overlap_f1(predicted_tokens, ground_truth_tokens)
        
        return ActionSpecificMetrics(
            bleu_score=bleu,
            meteor_score=meteor,
            rouge_l_f1=rouge_l_f1,
            rouge_1_f1=rouge_1_f1,
            rouge_2_f1=rouge_2_f1,
            semantic_similarity=semantic_sim,
            exact_match=exact_match,
            explanation=f"BLEU: {bleu:.3f}, METEOR: {meteor:.3f}, ROUGE-L: {rouge_l_f1:.3f}"
        )
    
    def extract_answer_from_response(self, answer_data: Dict, method_type: str) -> str:
        """Extract answer text from the answer data structure"""
        try:
            if method_type == "vanilla":
                return answer_data.get('vanilla_answer', {}).get('answer', '')
            elif method_type == "textchunkrag":
                return answer_data.get('textchunkrag_answer', {}).get('answer', '')
            elif method_type == "graphrag":
                return answer_data.get('graphrag_answer', {}).get('answer', '')
            else:
                # Fallback - try to find any answer field
                if 'answer' in answer_data:
                    return answer_data['answer']
                elif 'vanilla_answer' in answer_data:
                    return answer_data['vanilla_answer'].get('answer', '')
                elif 'textchunkrag_answer' in answer_data:
                    return answer_data['textchunkrag_answer'].get('answer', '')
                elif 'graphrag_answer' in answer_data:
                    return answer_data['graphrag_answer'].get('answer', '')
                else:
                    return ''
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return ''
    
    def evaluate_method(self, answers: List[Dict], action_questions: Dict[str, Dict], 
                       method_name: str) -> MethodEvaluation:
        """Evaluate a single method on action-specific questions"""
        
        print(f"Evaluating method: {method_name}")
        
        # Create question lookup
        answers_by_question = {answer.get('question_id', ''): answer for answer in answers}
        
        individual_results = []
        all_metrics = []
        
        # Process each action-specific question
        for question_id, question_data in tqdm(action_questions.items(), 
                                             desc=f"Evaluating {method_name}"):
            if question_id not in answers_by_question:
                continue
            
            answer_data = answers_by_question[question_id]
            ground_truth = question_data.get('ground_truth_answer', '').strip()
            
            # Extract answers for different methods
            methods_to_eval = []
            
            # Check which answer types are available
            if 'vanilla_answer' in answer_data:
                methods_to_eval.append(('vanilla', answer_data['vanilla_answer'].get('answer', '')))
            if 'textchunkrag_answer' in answer_data:
                methods_to_eval.append(('textchunkrag', answer_data['textchunkrag_answer'].get('answer', '')))
            if 'graphrag_answer' in answer_data:
                methods_to_eval.append(('graphrag', answer_data['graphrag_answer'].get('answer', '')))
            
            # If no specific methods found, try generic answer field
            if not methods_to_eval and 'answer' in answer_data:
                methods_to_eval.append(('generic', answer_data['answer']))
            
            # Evaluate each method type
            for method_type, predicted_answer in methods_to_eval:
                if not predicted_answer:
                    continue
                
                # Calculate metrics
                metrics = self.calculate_nlp_metrics(predicted_answer, ground_truth)
                all_metrics.append(metrics)
                
                # Store individual result
                result = {
                    'question_id': question_id,
                    'method_type': method_type,
                    'question': question_data.get('question', ''),
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'bleu_score': metrics.bleu_score,
                    'meteor_score': metrics.meteor_score,
                    'rouge_l_f1': metrics.rouge_l_f1,
                    'rouge_1_f1': metrics.rouge_1_f1,
                    'rouge_2_f1': metrics.rouge_2_f1,
                    'semantic_similarity': metrics.semantic_similarity,
                    'exact_match': metrics.exact_match
                }
                individual_results.append(result)
        
        # Calculate aggregate metrics
        if all_metrics:
            avg_bleu = np.mean([m.bleu_score for m in all_metrics])
            avg_meteor = np.mean([m.meteor_score for m in all_metrics])
            avg_rouge_l = np.mean([m.rouge_l_f1 for m in all_metrics])
            avg_rouge_1 = np.mean([m.rouge_1_f1 for m in all_metrics])
            avg_rouge_2 = np.mean([m.rouge_2_f1 for m in all_metrics])
            avg_semantic_sim = np.mean([m.semantic_similarity for m in all_metrics])
            exact_match_rate = np.mean([m.exact_match for m in all_metrics])
        else:
            avg_bleu = avg_meteor = avg_rouge_l = avg_rouge_1 = avg_rouge_2 = avg_semantic_sim = exact_match_rate = 0.0
        
        return MethodEvaluation(
            method_name=method_name,
            total_questions=len(individual_results),
            avg_bleu=avg_bleu,
            avg_meteor=avg_meteor,
            avg_rouge_l=avg_rouge_l,
            avg_rouge_1=avg_rouge_1,
            avg_rouge_2=avg_rouge_2,
            avg_semantic_sim=avg_semantic_sim,
            exact_match_rate=exact_match_rate,
            individual_results=individual_results
        )
    
    def evaluate_all_methods(self, answer_files: Dict[str, List[Dict]], 
                           action_questions: Dict[str, Dict]) -> Dict[str, MethodEvaluation]:
        """Evaluate all methods"""
        
        print("Starting evaluation of all methods...")
        evaluations = {}
        
        for method_name, answers in answer_files.items():
            try:
                evaluation = self.evaluate_method(answers, action_questions, method_name)
                evaluations[method_name] = evaluation
                print(f"Completed evaluation for {method_name}: {evaluation.total_questions} questions evaluated")
            except Exception as e:
                print(f"Error evaluating {method_name}: {e}")
        
        return evaluations
    
    def save_detailed_results(self, evaluations: Dict[str, MethodEvaluation], 
                            output_dir: str) -> None:
        """Save detailed results to CSV files"""
        
        print("Saving detailed results...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary results
        summary_data = []
        for method_name, evaluation in evaluations.items():
            summary_data.append({
                'method': method_name,
                'total_questions': evaluation.total_questions,
                'avg_bleu': evaluation.avg_bleu,
                'avg_meteor': evaluation.avg_meteor,
                'avg_rouge_l': evaluation.avg_rouge_l,
                'avg_rouge_1': evaluation.avg_rouge_1,
                'avg_rouge_2': evaluation.avg_rouge_2,
                'avg_semantic_similarity': evaluation.avg_semantic_sim,
                'exact_match_rate': evaluation.exact_match_rate
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'quantitative_evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Save detailed results for each method
        all_detailed_results = []
        for method_name, evaluation in evaluations.items():
            for result in evaluation.individual_results:
                result['method'] = method_name
                all_detailed_results.append(result)
        
        if all_detailed_results:
            detailed_df = pd.DataFrame(all_detailed_results)
            detailed_path = os.path.join(output_dir, 'quantitative_evaluation_detailed.csv')
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to {detailed_path}")
        
        # Save results as JSON for further analysis
        json_results = {}
        for method_name, evaluation in evaluations.items():
            json_results[method_name] = {
                'summary': {
                    'total_questions': evaluation.total_questions,
                    'avg_bleu': evaluation.avg_bleu,
                    'avg_meteor': evaluation.avg_meteor,
                    'avg_rouge_l': evaluation.avg_rouge_l,
                    'avg_rouge_1': evaluation.avg_rouge_1,
                    'avg_rouge_2': evaluation.avg_rouge_2,
                    'avg_semantic_similarity': evaluation.avg_semantic_sim,
                    'exact_match_rate': evaluation.exact_match_rate
                },
                'individual_results': evaluation.individual_results
            }
        
        json_path = os.path.join(output_dir, 'quantitative_evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to {json_path}")
    
    def print_summary(self, evaluations: Dict[str, MethodEvaluation]) -> None:
        """Print summary of evaluation results"""
        
        print("\n" + "="*80)
        print("QUANTITATIVE EVALUATION SUMMARY FOR ACTION-SPECIFIC QUESTIONS")
        print("="*80)
        
        if not evaluations:
            print("No evaluation results available.")
            return
        
        # Sort methods by average BLEU score for ranking
        sorted_methods = sorted(evaluations.items(), 
                              key=lambda x: x[1].avg_bleu, reverse=True)
        
        print(f"\nEvaluated {len(evaluations)} methods:")
        print("-" * 80)
        
        # Header
        print(f"{'Method':<25} {'Questions':<10} {'BLEU':<8} {'METEOR':<8} {'ROUGE-L':<8} {'Exact Match':<12}")
        print("-" * 80)
        
        # Results
        for method_name, evaluation in sorted_methods:
            print(f"{method_name:<25} {evaluation.total_questions:<10} "
                  f"{evaluation.avg_bleu:<8.3f} {evaluation.avg_meteor:<8.3f} "
                  f"{evaluation.avg_rouge_l:<8.3f} {evaluation.exact_match_rate:<12.3f}")
        
        print("-" * 80)
        
        # Best performing method
        if sorted_methods:
            best_method = sorted_methods[0]
            print(f"\nBest performing method (by BLEU): {best_method[0]}")
            print(f"  BLEU Score: {best_method[1].avg_bleu:.3f}")
            print(f"  METEOR Score: {best_method[1].avg_meteor:.3f}")
            print(f"  ROUGE-L F1: {best_method[1].avg_rouge_l:.3f}")
            print(f"  Exact Match Rate: {best_method[1].exact_match_rate:.3f}")
        
        print("\n" + "="*80)

def main():
    """Main evaluation function"""
    
    # Configuration
    base_dir = "/home/kuangshiai/Desktop/AAAI2026/Code/keo/keo/sensemaking_QA/output"
    kg_dir = "/home/kuangshiai/Desktop/AAAI2026/Code/keo/keo/sensemaking_QA/output/kg_gpt-4o"
    questions_file = "/home/kuangshiai/Desktop/AAAI2026/Code/keo/keo/sensemaking_QA/output/aviation_sensemaking_questions.json"
    output_dir = "/home/kuangshiai/Desktop/AAAI2026/Code/keo/keo/sensemaking_QA/evaluation_results/quantitative_action_specific"
    
    print("Starting Quantitative Action-Specific Question Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = QuantitativeActionEvaluator()
    
    # Load questions
    action_questions = evaluator.load_questions(questions_file)
    if not action_questions:
        print("No action-specific questions with ground truth found!")
        return
    
    # Load answer files
    answer_files = evaluator.load_answer_files(base_dir, kg_dir)
    if not answer_files:
        print("No answer files found!")
        return
    
    # Evaluate all methods
    evaluations = evaluator.evaluate_all_methods(answer_files, action_questions)
    
    # Print summary
    evaluator.print_summary(evaluations)
    
    # Save results
    evaluator.save_detailed_results(evaluations, output_dir)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
