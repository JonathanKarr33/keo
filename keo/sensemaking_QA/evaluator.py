"""
Evaluation Framework for Sensemaking Questions and Answers
Based on GraphRAG evaluation methodology with qualitative metrics
Enhanced with ground truth evaluation for action-specific questions
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import time
from dataclasses import dataclass
import re
from collections import Counter

# NLP evaluation metrics imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    NLP_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some NLP metrics libraries not available: {e}")
    print("Install with: pip install nltk rouge-score")
    NLP_METRICS_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Structure for evaluation metrics"""
    comprehensiveness: float
    human_enfranchisement: float
    diversity: float
    faithfulness: float
    overall_score: float
    explanation: str


@dataclass
class ActionSpecificMetrics:
    """Structure for action-specific evaluation metrics with ground truth"""
    bleu_score: float
    meteor_score: float
    rouge_l_f1: float
    rouge_1_f1: float
    rouge_2_f1: float
    semantic_similarity: float
    llm_evaluation_score: float
    exact_match: bool
    explanation: str


@dataclass
class GroundTruthEvaluation:
    """Structure for ground truth evaluation results"""
    question_id: str
    ground_truth: str
    predicted_answer: str
    nlp_metrics: ActionSpecificMetrics
    llm_evaluation: Dict
    overall_score: float


class SensemakingEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the evaluator with GraphRAG-style metrics
        
        Args:
            api_key: OpenAI API key
            model: Model to use for evaluation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # GraphRAG evaluation criteria
        self.evaluation_criteria = {
            'comprehensiveness': {
                'description': 'Completeness within the framing of the implied context of the question',
                'scale': '1-5 where 5 = thoroughly comprehensive, 1 = incomplete or superficial'
            },
            'human_enfranchisement': {
                'description': 'Provision of supporting source material or contextual information',
                'scale': '1-5 where 5 = excellent provenance and citations, 1 = no supporting evidence'
            },
            'diversity': {
                'description': 'Provision of differing viewpoints or angles on the question posed',
                'scale': '1-5 where 5 = multiple perspectives considered, 1 = single narrow view'
            },
            'faithfulness': {
                'description': 'Factual accuracy and grounding in source material',
                'scale': '1-5 where 5 = fully accurate and grounded, 1 = inaccurate or hallucinated'
            }
        }
    
    def evaluate_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Evaluate the quality of generated sensemaking questions
        
        Args:
            questions: List of question dictionaries
        
        Returns:
            List of question evaluations
        """
        print("Evaluating sensemaking questions...")
        evaluations = []
        
        for question_data in tqdm(questions, desc="Evaluating questions"):
            try:
                evaluation = self._evaluate_single_question(question_data)
                evaluations.append(evaluation)
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error evaluating question {question_data.get('id', '')}: {e}")
                evaluations.append({
                    'question_id': question_data.get('id', ''),
                    'error': True,
                    'error_message': str(e)
                })
        
        return evaluations
    
    def evaluate_answers(self, 
                        answers: List[Dict], 
                        questions: List[Dict],
                        reference_data: Optional[Dict] = None) -> List[Dict]:
        """
        Evaluate answers using GraphRAG-style qualitative metrics
        
        Args:
            answers: List of answer dictionaries
            questions: Original questions for context
            reference_data: Optional reference data for faithfulness checking
        
        Returns:
            List of answer evaluations
        """
        print("Evaluating answers with GraphRAG-style metrics...")
        evaluations = []
        
        # Create question lookup
        question_lookup = {q.get('id', ''): q for q in questions}
        
        for answer_data in tqdm(answers, desc="Evaluating answers"):
            try:
                if answer_data.get('error'):
                    continue
                
                question_id = answer_data.get('question_id', '')
                question_data = question_lookup.get(question_id, {})
                
                evaluation = self._evaluate_single_answer(
                    answer_data, question_data, reference_data
                )
                evaluations.append(evaluation)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error evaluating answer {answer_data.get('question_id', '')}: {e}")
                evaluations.append({
                    'question_id': answer_data.get('question_id', ''),
                    'answer_id': answer_data.get('id', ''),
                    'error': True,
                    'error_message': str(e)
                })
        
        return evaluations
    
    def compare_answer_methods(self, 
                             vanilla_answers: List[Dict],
                             graphrag_answers: List[Dict],
                             questions: List[Dict]) -> Dict:
        """
        Compare vanilla LLM vs GraphRAG answer quality
        
        Args:
            vanilla_answers: Answers from vanilla LLM
            graphrag_answers: Answers from GraphRAG
            questions: Original questions
        
        Returns:
            Comparative evaluation results
        """
        print("Comparing vanilla LLM vs GraphRAG answers...")
        
        # Evaluate both sets of answers
        vanilla_evaluations = self.evaluate_answers(vanilla_answers, questions)
        graphrag_evaluations = self.evaluate_answers(graphrag_answers, questions)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self._perform_pairwise_comparisons(
            vanilla_answers, graphrag_answers, questions
        )
        
        # Calculate aggregate metrics
        comparison_results = self._calculate_comparison_metrics(
            vanilla_evaluations, graphrag_evaluations, pairwise_comparisons
        )
        
        return comparison_results
    
    def evaluate_global_sensemaking_capability(self,
                                             questions: List[Dict],
                                             answers: List[Dict]) -> Dict:
        """
        Evaluate capability on global sensemaking questions specifically
        
        Args:
            questions: Global sensemaking questions
            answers: Corresponding answers
        
        Returns:
            Global sensemaking evaluation results
        """
        print("Evaluating global sensemaking capability...")
        
        # Filter for global questions
        global_questions = [q for q in questions if q.get('type') == 'global']
        global_answers = [a for a in answers if any(
            a.get('question_id') == q.get('id') for q in global_questions
        )]
        
        if not global_questions or not global_answers:
            return {"error": "No global sensemaking questions/answers found"}
        
        # Evaluate global questions specifically
        global_evaluations = []
        
        for answer in global_answers:
            try:
                question_id = answer.get('question_id', '')
                question = next((q for q in global_questions if q.get('id') == question_id), {})
                
                if question:
                    evaluation = self._evaluate_global_sensemaking_answer(answer, question)
                    global_evaluations.append(evaluation)
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error evaluating global answer {answer.get('question_id', '')}: {e}")
        
        # Calculate global sensemaking metrics
        global_metrics = self._calculate_global_metrics(global_evaluations)
        
        return {
            'global_questions_evaluated': len(global_evaluations),
            'global_metrics': global_metrics,
            'individual_evaluations': global_evaluations
        }
    
    def evaluate_action_specific_answers(self,
                                       answers: List[Dict],
                                       questions: List[Dict]) -> Dict:
        """
        Evaluate action-specific answers using ground truth and NLP metrics
        
        Args:
            answers: List of answer dictionaries
            questions: List of questions with ground truth for action-specific questions
        
        Returns:
            Comprehensive evaluation results for action-specific questions
        """
        print("Evaluating action-specific answers with ground truth metrics...")
        
        # Filter for action-specific questions with ground truth
        action_questions = {q.get('id', ''): q for q in questions 
                          if q.get('category') == 'action_specific' and q.get('ground_truth_answer')}
        
        action_answers = [a for a in answers 
                         if a.get('question_id') in action_questions and not a.get('error')]
        
        if not action_questions or not action_answers:
            return {"error": "No action-specific questions with ground truth found"}
        
        print(f"Found {len(action_questions)} action-specific questions with ground truth")
        print(f"Found {len(action_answers)} corresponding answers to evaluate")
        
        # Evaluate each answer
        evaluations = []
        for answer in tqdm(action_answers, desc="Evaluating action-specific answers"):
            try:
                question_id = answer.get('question_id', '')
                question = action_questions.get(question_id)
                
                if question:
                    evaluation = self._evaluate_single_action_answer(answer, question)
                    evaluations.append(evaluation)
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error evaluating action answer {answer.get('question_id', '')}: {e}")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_action_aggregate_metrics(evaluations)
        
        return {
            'total_action_questions': len(action_questions),
            'evaluated_answers': len(evaluations),
            'individual_evaluations': evaluations,
            'aggregate_metrics': aggregate_metrics,
            'evaluation_summary': self._generate_action_evaluation_summary(aggregate_metrics)
        }
    
    def compare_action_specific_methods(self,
                                      vanilla_answers: List[Dict],
                                      graphrag_answers: List[Dict],
                                      questions: List[Dict]) -> Dict:
        """
        Compare vanilla LLM vs GraphRAG for action-specific questions with ground truth
        
        Args:
            vanilla_answers: Vanilla LLM answers
            graphrag_answers: GraphRAG answers
            questions: Questions with ground truth
        
        Returns:
            Comparison results for action-specific questions
        """
        print("Comparing methods on action-specific questions with ground truth...")
        
        # Evaluate both methods
        vanilla_results = self.evaluate_action_specific_answers(vanilla_answers, questions)
        graphrag_results = self.evaluate_action_specific_answers(graphrag_answers, questions)
        
        if vanilla_results.get('error') or graphrag_results.get('error'):
            return {"error": "Failed to evaluate one or both methods"}
        
        # Compare aggregate metrics
        comparison = self._compare_action_metrics(
            vanilla_results['aggregate_metrics'],
            graphrag_results['aggregate_metrics']
        )
        
        return {
            'vanilla_results': vanilla_results,
            'graphrag_results': graphrag_results,
            'comparison': comparison,
            'winner': self._determine_action_winner(comparison)
        }
    
    def _evaluate_single_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question for quality"""
        
        prompt = f"""
Evaluate this aviation maintenance sensemaking question for quality:

Question: {question_data['question']}
Category: {question_data.get('category', 'unknown')}
Type: {question_data.get('type', 'unknown')}

Rate the question on these criteria (1-5 scale):

1. Clarity: Is the question clear and well-formulated?
2. Relevance: How relevant is it to aviation maintenance safety?
3. Complexity: Does it require synthesis across multiple data points?
4. Actionability: Could answers lead to actionable insights?
5. Sensemaking Value: Does it support understanding of broader patterns?

For each criterion, provide:
- Score (1-5)
- Brief explanation

Then provide an overall assessment and suggestions for improvement if any.

Format your response as:
Clarity: [score] - [explanation]
Relevance: [score] - [explanation]
Complexity: [score] - [explanation]
Actionability: [score] - [explanation]
Sensemaking Value: [score] - [explanation]

Overall Assessment: [overall score 1-5] - [explanation]
Suggestions: [any improvements needed]
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of research questions in aviation safety and maintenance, with deep knowledge of what makes effective sensemaking questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        
        # Parse scores (simplified parsing)
        scores = self._parse_question_scores(evaluation_text)
        
        return {
            'question_id': question_data.get('id', ''),
            'question': question_data['question'],
            'category': question_data.get('category', ''),
            'evaluation_scores': scores,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model
        }
    
    def _evaluate_single_answer(self, 
                              answer_data: Dict, 
                              question_data: Dict,
                              reference_data: Optional[Dict] = None) -> Dict:
        """Evaluate a single answer using GraphRAG criteria"""
        
        prompt = f"""
Evaluate this answer to an aviation maintenance sensemaking question using the following criteria:

Question: {question_data.get('question', '')}
Answer: {answer_data.get('answer', '')}
Method: {answer_data.get('method', 'unknown')}

Rate the answer on these GraphRAG-style criteria (1-5 scale):

1. COMPREHENSIVENESS: Completeness within the framing of the implied context of the question
   - Does the answer thoroughly address all aspects of the question?
   - Is it complete within the expected scope?

2. HUMAN ENFRANCHISEMENT: Provision of supporting source material or contextual information
   - Does it provide evidence, examples, or source references?
   - Can a human verify and build upon the information?

3. DIVERSITY: Provision of differing viewpoints or angles on the question posed
   - Does it consider multiple perspectives or approaches?
   - Are different factors or scenarios addressed?

4. FAITHFULNESS: Factual accuracy and grounding in source material
   - Is the information accurate and well-grounded?
   - Are there any apparent hallucinations or inaccuracies?

For each criterion, provide:
- Score (1-5)
- Detailed explanation

Then provide an overall assessment.

Format your response as:
Comprehensiveness: [score] - [explanation]
Human Enfranchisement: [score] - [explanation]
Diversity: [score] - [explanation]
Faithfulness: [score] - [explanation]

Overall Score: [average score] - [overall assessment]
Strengths: [key strengths]
Weaknesses: [areas for improvement]
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of analytical responses in aviation safety, using GraphRAG evaluation methodology to assess answer quality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        
        # Parse scores
        metrics = self._parse_answer_scores(evaluation_text)
        
        return {
            'question_id': answer_data.get('question_id', ''),
            'answer_method': answer_data.get('method', ''),
            'evaluation_metrics': metrics,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model
        }
    
    def _evaluate_global_sensemaking_answer(self, answer_data: Dict, question_data: Dict) -> Dict:
        """Evaluate answers specifically for global sensemaking capability"""
        
        prompt = f"""
Evaluate this answer to a GLOBAL SENSEMAKING question about aviation maintenance data:

Question: {question_data.get('question', '')}
Answer: {answer_data.get('answer', '')}

Global sensemaking questions require synthesis across entire datasets to identify overarching themes and patterns.

Rate this answer on these specialized criteria (1-5 scale):

1. GLOBAL PERSPECTIVE: Does the answer demonstrate understanding of dataset-wide patterns?
2. THEME IDENTIFICATION: Are major themes and patterns clearly identified?
3. SYNTHESIS QUALITY: How well does it synthesize information across multiple sources?
4. STRATEGIC VALUE: Does it provide insights useful for high-level decision making?
5. PATTERN RECOGNITION: Are complex relationships and dependencies identified?

For each criterion, provide score and explanation.

Then assess whether this answer successfully demonstrates global sensemaking capability.

Format as:
Global Perspective: [score] - [explanation]
Theme Identification: [score] - [explanation]
Synthesis Quality: [score] - [explanation]
Strategic Value: [score] - [explanation]
Pattern Recognition: [score] - [explanation]

Global Sensemaking Assessment: [overall evaluation of global capability]
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in evaluating global sensemaking capabilities in data analysis systems, particularly for aviation safety applications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.2
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        
        # Parse global sensemaking scores
        global_scores = self._parse_global_scores(evaluation_text)
        
        return {
            'question_id': question_data.get('id', ''),
            'global_metrics': global_scores,
            'evaluation_text': evaluation_text,
            'evaluator_model': self.model
        }
    
    def evaluate_action_specific_answers(self,
                                       answers: List[Dict],
                                       questions: List[Dict]) -> Dict:
        """
        Evaluate action-specific answers using ground truth and NLP metrics
        
        Args:
            answers: List of answer dictionaries
            questions: List of questions with ground truth for action-specific questions
        
        Returns:
            Comprehensive evaluation results for action-specific questions
        """
        print("Evaluating action-specific answers with ground truth metrics...")
        
        # Filter for action-specific questions with ground truth
        action_questions = {q.get('id', ''): q for q in questions 
                          if q.get('category') == 'action_specific' and q.get('ground_truth_answer')}
        
        action_answers = [a for a in answers 
                         if a.get('question_id') in action_questions and not a.get('error')]
        
        if not action_questions or not action_answers:
            return {"error": "No action-specific questions with ground truth found"}
        
        print(f"Found {len(action_questions)} action-specific questions with ground truth")
        print(f"Found {len(action_answers)} corresponding answers to evaluate")
        
        # Evaluate each answer
        evaluations = []
        for answer in tqdm(action_answers, desc="Evaluating action-specific answers"):
            try:
                question_id = answer.get('question_id', '')
                question = action_questions.get(question_id)
                
                if question:
                    evaluation = self._evaluate_single_action_answer(answer, question)
                    evaluations.append(evaluation)
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error evaluating action answer {answer.get('question_id', '')}: {e}")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_action_aggregate_metrics(evaluations)
        
        return {
            'total_action_questions': len(action_questions),
            'evaluated_answers': len(evaluations),
            'individual_evaluations': evaluations,
            'aggregate_metrics': aggregate_metrics,
            'evaluation_summary': self._generate_action_evaluation_summary(aggregate_metrics)
        }
    
    def compare_action_specific_methods(self,
                                      vanilla_answers: List[Dict],
                                      graphrag_answers: List[Dict],
                                      questions: List[Dict]) -> Dict:
        """
        Compare vanilla LLM vs GraphRAG for action-specific questions with ground truth
        
        Args:
            vanilla_answers: Vanilla LLM answers
            graphrag_answers: GraphRAG answers
            questions: Questions with ground truth
        
        Returns:
            Comparison results for action-specific questions
        """
        print("Comparing methods on action-specific questions with ground truth...")
        
        # Evaluate both methods
        vanilla_results = self.evaluate_action_specific_answers(vanilla_answers, questions)
        graphrag_results = self.evaluate_action_specific_answers(graphrag_answers, questions)
        
        if vanilla_results.get('error') or graphrag_results.get('error'):
            return {"error": "Failed to evaluate one or both methods"}
        
        # Compare aggregate metrics
        comparison = self._compare_action_metrics(
            vanilla_results['aggregate_metrics'],
            graphrag_results['aggregate_metrics']
        )
        
        return {
            'vanilla_results': vanilla_results,
            'graphrag_results': graphrag_results,
            'comparison': comparison,
            'winner': self._determine_action_winner(comparison)
        }
    
    def _evaluate_single_action_answer(self, answer: Dict, question: Dict) -> GroundTruthEvaluation:
        """Evaluate a single action-specific answer against ground truth"""
        
        ground_truth = question.get('ground_truth_answer', '').strip()
        predicted = answer.get('answer', '').strip()
        question_id = question.get('id', '')
        
        # Calculate NLP metrics
        nlp_metrics = self._calculate_nlp_metrics(predicted, ground_truth)
        
        # LLM-based evaluation
        llm_evaluation = self._llm_evaluate_action_answer(
            question.get('question', ''), predicted, ground_truth
        )
        
        # Calculate overall score (weighted combination)
        overall_score = self._calculate_overall_action_score(nlp_metrics, llm_evaluation)
        
        return GroundTruthEvaluation(
            question_id=question_id,
            ground_truth=ground_truth,
            predicted_answer=predicted,
            nlp_metrics=nlp_metrics,
            llm_evaluation=llm_evaluation,
            overall_score=overall_score
        )
    
    def _calculate_nlp_metrics(self, predicted: str, ground_truth: str) -> ActionSpecificMetrics:
        """Calculate NLP metrics for action-specific answers"""
        
        if not NLP_METRICS_AVAILABLE:
            return ActionSpecificMetrics(
                bleu_score=0.0,
                meteor_score=0.0,
                rouge_l_f1=0.0,
                rouge_1_f1=0.0,
                rouge_2_f1=0.0,
                semantic_similarity=0.0,
                llm_evaluation_score=0.0,
                exact_match=False,
                explanation="NLP metrics libraries not available"
            )
        
        # Preprocess text
        predicted_clean = self._preprocess_text(predicted)
        ground_truth_clean = self._preprocess_text(ground_truth)
        
        # Exact match
        exact_match = predicted_clean.lower() == ground_truth_clean.lower()
        
        # Tokenize
        predicted_tokens = predicted_clean.split()
        ground_truth_tokens = ground_truth_clean.split()
        
        # BLEU score
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([ground_truth_tokens], predicted_tokens, smoothing_function=smoothing)
        
        # METEOR score
        try:
            meteor = meteor_score([ground_truth_clean], predicted_clean)
        except:
            meteor = 0.0
        
        # ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ground_truth_clean, predicted_clean)
            rouge_1_f1 = rouge_scores['rouge1'].fmeasure
            rouge_2_f1 = rouge_scores['rouge2'].fmeasure
            rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        except:
            rouge_1_f1 = rouge_2_f1 = rouge_l_f1 = 0.0
        
        # Semantic similarity (simple word overlap F1)
        semantic_sim = self._calculate_word_overlap_f1(predicted_tokens, ground_truth_tokens)
        
        return ActionSpecificMetrics(
            bleu_score=bleu,
            meteor_score=meteor,
            rouge_l_f1=rouge_l_f1,
            rouge_1_f1=rouge_1_f1,
            rouge_2_f1=rouge_2_f1,
            semantic_similarity=semantic_sim,
            llm_evaluation_score=0.0,  # Will be filled by LLM evaluation
            exact_match=exact_match,
            explanation=f"BLEU: {bleu:.3f}, METEOR: {meteor:.3f}, ROUGE-L: {rouge_l_f1:.3f}"
        )
    
    def _llm_evaluate_action_answer(self, question: str, predicted: str, ground_truth: str) -> Dict:
        """Use LLM to evaluate action-specific answer quality"""
        
        prompt = f"""
Evaluate this predicted answer for an action-specific aviation maintenance question against the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Rate the predicted answer on these criteria (1-5 scale):

1. CORRECTNESS: How factually correct is the predicted answer?
2. COMPLETENESS: Does it include all necessary action steps?
3. PRACTICALITY: How practical and actionable is the suggested action?
4. SAFETY: Does it maintain or improve safety standards?
5. CLARITY: How clear and understandable is the instruction?

For each criterion, provide:
- Score (1-5)
- Brief explanation

Then provide:
- Overall assessment of answer quality
- Whether the predicted answer would be safe to follow
- Key differences from ground truth
- Suggestions for improvement

Format as:
Correctness: [score] - [explanation]
Completeness: [score] - [explanation]
Practicality: [score] - [explanation]
Safety: [score] - [explanation]
Clarity: [score] - [explanation]

Overall Score: [average score] - [overall assessment]
Safety Assessment: [Safe/Unsafe/Partially Safe] - [explanation]
Key Differences: [main differences from ground truth]
Improvement Suggestions: [suggestions if any]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert aviation maintenance evaluator with deep knowledge of proper maintenance procedures and safety standards."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.2
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # Parse LLM evaluation scores
            llm_scores = self._parse_action_llm_scores(evaluation_text)
            
            return {
                'llm_scores': llm_scores,
                'evaluation_text': evaluation_text,
                'overall_llm_score': np.mean(list(llm_scores.values())) if llm_scores else 3.0
            }
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                'llm_scores': {},
                'evaluation_text': f"Error: {str(e)}",
                'overall_llm_score': 3.0
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP metrics"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove common maintenance text artifacts
        text = re.sub(r'[^\w\s&]', ' ', text)  # Keep & for maintenance actions
        text = ' '.join(text.split())
        return text
    
    def _calculate_word_overlap_f1(self, predicted_tokens: List[str], ground_truth_tokens: List[str]) -> float:
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
    
    def _parse_question_scores(self, evaluation_text: str) -> Dict:
        """Parse LLM evaluation scores for question quality"""
        scores = {}
        lines = evaluation_text.split('\n')
        
        criteria = ['Clarity', 'Relevance', 'Complexity', 'Actionability', 'Sensemaking Value', 'Overall Assessment']
        
        for line in lines:
            for criterion in criteria:
                if line.startswith(criterion):
                    try:
                        score_part = line.split(':')[1].strip()
                        # Extract the first number found
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', score_part)
                        if numbers:
                            score = float(numbers[0])
                            scores[criterion.lower().replace(' ', '_')] = score
                        else:
                            scores[criterion.lower().replace(' ', '_')] = 3.0
                    except:
                        scores[criterion.lower().replace(' ', '_')] = 3.0
        
        return scores

    def _parse_answer_scores(self, evaluation_text: str) -> EvaluationMetrics:
        """Parse GraphRAG-style evaluation scores from LLM response"""
        lines = evaluation_text.split('\n')
        scores = {}
        
        # Default scores
        default_scores = {
            'comprehensiveness': 3.0,
            'human_enfranchisement': 3.0,
            'diversity': 3.0,
            'faithfulness': 3.0
        }
        
        for line in lines:
            line = line.strip()
            for criterion in default_scores.keys():
                if line.lower().startswith(criterion.lower()):
                    try:
                        score_part = line.split(':')[1].strip()
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', score_part)
                        if numbers:
                            score = float(numbers[0])
                            scores[criterion] = score
                    except:
                        continue
        
        # Fill in missing scores with defaults
        for criterion, default_score in default_scores.items():
            if criterion not in scores:
                scores[criterion] = default_score
        
        overall_score = sum(scores.values()) / len(scores)
        
        return EvaluationMetrics(
            comprehensiveness=scores['comprehensiveness'],
            human_enfranchisement=scores['human_enfranchisement'],
            diversity=scores['diversity'],
            faithfulness=scores['faithfulness'],
            overall_score=overall_score,
            explanation=evaluation_text
        )

    def _parse_global_scores(self, evaluation_text: str) -> Dict:
        """Parse global sensemaking evaluation scores"""
        scores = {}
        lines = evaluation_text.split('\n')
        
        criteria = ['Global Perspective', 'Theme Identification', 'Synthesis Quality', 'Strategic Value', 'Pattern Recognition']
        
        for line in lines:
            for criterion in criteria:
                if line.startswith(criterion):
                    try:
                        score_part = line.split(':')[1].strip()
                        import re
                        numbers = re.findall(r'\d+(?:\.\d+)?', score_part)
                        if numbers:
                            score = float(numbers[0])
                            scores[criterion.lower().replace(' ', '_')] = score
                        else:
                            scores[criterion.lower().replace(' ', '_')] = 3.0
                    except:
                        scores[criterion.lower().replace(' ', '_')] = 3.0
        
        return scores

    def _perform_pairwise_comparisons(self, vanilla_answers: List[Dict], graphrag_answers: List[Dict], questions: List[Dict]) -> List[Dict]:
        """Perform pairwise comparisons between vanilla and GraphRAG answers"""
        comparisons = []
        
        # Create lookups
        vanilla_lookup = {a.get('question_id', ''): a for a in vanilla_answers}
        graphrag_lookup = {a.get('question_id', ''): a for a in graphrag_answers}
        question_lookup = {q.get('id', ''): q for q in questions}
        
        # Find matching questions
        common_question_ids = set(vanilla_lookup.keys()) & set(graphrag_lookup.keys())
        
        for question_id in list(common_question_ids)[:5]:  # Limit for demo
            try:
                vanilla_answer = vanilla_lookup[question_id]
                graphrag_answer = graphrag_lookup[question_id]
                question = question_lookup.get(question_id, {})
                
                comparison = self._compare_answer_pair(vanilla_answer, graphrag_answer, question)
                comparisons.append(comparison)
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in pairwise comparison for {question_id}: {e}")
        
        return comparisons

    def _compare_answer_pair(self, vanilla_answer: Dict, graphrag_answer: Dict, question: Dict) -> Dict:
        """Compare a pair of answers (vanilla vs GraphRAG)"""
        
        prompt = f"""
Compare these two answers to the same aviation maintenance question:

Question: {question.get('question', '')}

Answer A (Vanilla LLM): {vanilla_answer.get('answer', '')}
Answer B (GraphRAG): {graphrag_answer.get('answer', '')}

Compare the answers on these criteria:
1. Which is more comprehensive?
2. Which provides better supporting evidence?
3. Which offers more diverse perspectives?
4. Which is more factually accurate?

For each criterion, indicate: A, B, or Tie

Then provide an overall preference: A, B, or Tie

Format as:
Comprehensiveness: [A/B/Tie] - [explanation]
Supporting Evidence: [A/B/Tie] - [explanation]
Diverse Perspectives: [A/B/Tie] - [explanation]
Factual Accuracy: [A/B/Tie] - [explanation]

Overall Preference: [A/B/Tie] - [explanation]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator comparing analytical responses for aviation safety questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            comparison_text = response.choices[0].message.content.strip()
            
            # Parse comparison results
            comparison_scores = self._parse_comparison_scores(comparison_text)
            
            return {
                'question_id': question.get('id', ''),
                'vanilla_answer_id': vanilla_answer.get('id', ''),
                'graphrag_answer_id': graphrag_answer.get('id', ''),
                'comparison_scores': comparison_scores,
                'comparison_text': comparison_text
            }
            
        except Exception as e:
            return {
                'question_id': question.get('id', ''),
                'error': str(e)
            }

    def _parse_comparison_scores(self, comparison_text: str) -> Dict:
        """Parse pairwise comparison scores"""
        scores = {}
        lines = comparison_text.split('\n')
        
        criteria = ['Comprehensiveness', 'Supporting Evidence', 'Diverse Perspectives', 'Factual Accuracy', 'Overall Preference']
        
        for line in lines:
            for criterion in criteria:
                if line.startswith(criterion):
                    try:
                        result_part = line.split(':')[1].strip()
                        if result_part.startswith('A'):
                            scores[criterion.lower().replace(' ', '_')] = 'vanilla'
                        elif result_part.startswith('B'):
                            scores[criterion.lower().replace(' ', '_')] = 'graphrag'
                        else:
                            scores[criterion.lower().replace(' ', '_')] = 'tie'
                    except:
                        scores[criterion.lower().replace(' ', '_')] = 'tie'
        
        return scores

    def _calculate_comparison_metrics(self, vanilla_evaluations: List[Dict], graphrag_evaluations: List[Dict], pairwise_comparisons: List[Dict]) -> Dict:
        """Calculate comparison metrics between methods"""
        
        # Calculate average scores for each method
        vanilla_avg = self._calculate_average_scores(vanilla_evaluations)
        graphrag_avg = self._calculate_average_scores(graphrag_evaluations)
        
        # Analyze pairwise comparisons
        pairwise_results = self._analyze_pairwise_results(pairwise_comparisons)
        
        return {
            'vanilla_average_scores': vanilla_avg,
            'graphrag_average_scores': graphrag_avg,
            'pairwise_comparison_results': pairwise_results,
            'evaluation_summary': self._generate_comparison_summary(vanilla_avg, graphrag_avg, pairwise_results)
        }

    def _calculate_average_scores(self, evaluations: List[Dict]) -> Dict:
        """Calculate average scores from evaluations"""
        if not evaluations:
            return {}
        
        total_scores = {
            'comprehensiveness': 0,
            'human_enfranchisement': 0,
            'diversity': 0,
            'faithfulness': 0,
            'overall_score': 0
        }
        count = 0
        
        for evaluation in evaluations:
            if evaluation.get('error'):
                continue
            
            metrics = evaluation.get('evaluation_metrics')
            if metrics:
                for key in total_scores.keys():
                    if hasattr(metrics, key):
                        total_scores[key] += getattr(metrics, key)
                count += 1
        
        if count == 0:
            return total_scores
        
        # Calculate averages
        average_scores = {key: score / count for key, score in total_scores.items()}
        return average_scores

    def _analyze_pairwise_results(self, pairwise_comparisons: List[Dict]) -> Dict:
        """Analyze pairwise comparison results"""
        if not pairwise_comparisons:
            return {}
        
        criteria = ['comprehensiveness', 'supporting_evidence', 'diverse_perspectives', 'factual_accuracy', 'overall_preference']
        
        results = {}
        for criterion in criteria:
            vanilla_wins = 0
            graphrag_wins = 0
            ties = 0
            
            for comparison in pairwise_comparisons:
                if comparison.get('error'):
                    continue
                
                scores = comparison.get('comparison_scores', {})
                result = scores.get(criterion, 'tie')
                
                if result == 'vanilla':
                    vanilla_wins += 1
                elif result == 'graphrag':
                    graphrag_wins += 1
                else:
                    ties += 1
            
            total = vanilla_wins + graphrag_wins + ties
            if total > 0:
                results[criterion] = {
                    'vanilla_wins': vanilla_wins,
                    'graphrag_wins': graphrag_wins,
                    'ties': ties,
                    'vanilla_win_rate': vanilla_wins / total,
                    'graphrag_win_rate': graphrag_wins / total,
                    'tie_rate': ties / total
                }
        
        return results

    def _generate_comparison_summary(self, vanilla_avg: Dict, graphrag_avg: Dict, pairwise_results: Dict) -> str:
        """Generate a summary of comparison results"""
        
        summary_parts = []
        summary_parts.append("EVALUATION COMPARISON SUMMARY")
        summary_parts.append("=" * 40)
        
        # Average scores comparison
        if vanilla_avg and graphrag_avg:
            summary_parts.append("Average Scores:")
            for criterion in ['comprehensiveness', 'human_enfranchisement', 'diversity', 'faithfulness', 'overall_score']:
                vanilla_score = vanilla_avg.get(criterion, 0)
                graphrag_score = graphrag_avg.get(criterion, 0)
                summary_parts.append(f"  {criterion.title()}: Vanilla {vanilla_score:.2f} vs GraphRAG {graphrag_score:.2f}")
        
        # Pairwise results
        if pairwise_results:
            summary_parts.append("\nPairwise Comparison Results:")
            for criterion, results in pairwise_results.items():
                graphrag_rate = results.get('graphrag_win_rate', 0)
                summary_parts.append(f"  {criterion.replace('_', ' ').title()}: GraphRAG wins {graphrag_rate:.1%} of comparisons")
        
        return "\n".join(summary_parts)

    def _calculate_global_metrics(self, global_evaluations: List[Dict]) -> Dict:
        """Calculate metrics for global sensemaking evaluation"""
        if not global_evaluations:
            return {}
        
        criteria = ['global_perspective', 'theme_identification', 'synthesis_quality', 'strategic_value', 'pattern_recognition']
        
        total_scores = {criterion: 0 for criterion in criteria}
        count = 0
        
        for evaluation in global_evaluations:
            if evaluation.get('error'):
                continue
            
            global_metrics = evaluation.get('global_metrics', {})
            for criterion in criteria:
                if criterion in global_metrics:
                    total_scores[criterion] += global_metrics[criterion]
            count += 1
        
        if count == 0:
            return total_scores
        
        # Calculate averages
        average_scores = {key: score / count for key, score in total_scores.items()}
        average_scores['total_evaluated'] = count
        
        return average_scores
    
    def save_evaluation_results(self, results: Dict, output_path: str) -> None:
        """Save evaluation results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Evaluation results saved to {output_path}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")


if __name__ == "__main__":
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    evaluator = SensemakingEvaluator(openai_api_key)
    
    # Load questions and answers if available
    try:
        with open("aviation_sensemaking_questions.json", 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Evaluate questions
        question_evaluations = evaluator.evaluate_questions(questions[:5])  # Sample
        evaluator.save_evaluation_results(
            {'question_evaluations': question_evaluations}, 
            "question_evaluations.json"
        )
        
        # Separate action-specific questions for ground truth evaluation
        action_questions = [q for q in questions if q.get('category') == 'action_specific']
        if action_questions:
            print(f"Found {len(action_questions)} action-specific questions with ground truth")
        
    except FileNotFoundError:
        print("Questions file not found")
    
    # Load and evaluate answers if available
    try:
        with open("vanilla_answers.json", 'r') as f:
            vanilla_answers = json.load(f)
        
        with open("graphrag_answers.json", 'r') as f:
            graphrag_answers = json.load(f)
        
        # Standard comparison for non-action questions
        non_action_questions = [q for q in questions if q.get('category') != 'action_specific']
        if non_action_questions:
            comparison_results = evaluator.compare_answer_methods(
                vanilla_answers, graphrag_answers, non_action_questions[:5]
            )
            evaluator.save_evaluation_results(comparison_results, "method_comparison_results.json")
            print("Standard Evaluation Results:")
            print(comparison_results.get('evaluation_summary', 'No summary available'))
        
        # Action-specific evaluation with ground truth
        if action_questions:
            print("\n" + "="*50)
            print("ACTION-SPECIFIC EVALUATION WITH GROUND TRUTH")
            print("="*50)
            
            # Evaluate action-specific answers with ground truth metrics
            action_comparison = evaluator.compare_action_specific_methods(
                vanilla_answers, graphrag_answers, action_questions
            )
            
            if not action_comparison.get('error'):
                evaluator.save_evaluation_results(action_comparison, "action_specific_evaluation.json")
                
                print("\nAction-Specific Results Summary:")
                print(action_comparison['vanilla_results'].get('evaluation_summary', ''))
                print("\nGraphRAG vs Vanilla Comparison:")
                winner_info = action_comparison.get('winner', {})
                print(f"Overall Winner: {winner_info.get('overall_winner', 'Unknown')}")
                print(f"GraphRAG Win Rate: {winner_info.get('graphrag_win_rate', 0):.1%}")
                print(f"Vanilla Win Rate: {winner_info.get('vanilla_win_rate', 0):.1%}")
            else:
                print("Error in action-specific evaluation:", action_comparison.get('error'))
        
        # Global sensemaking evaluation
        global_questions = [q for q in questions if q.get('type') == 'global']
        if global_questions:
            print("\n" + "="*50)
            print("GLOBAL SENSEMAKING EVALUATION")
            print("="*50)
            
            global_results = evaluator.evaluate_global_sensemaking_capability(
                global_questions, graphrag_answers  # GraphRAG should be better for global questions
            )
            
            if not global_results.get('error'):
                evaluator.save_evaluation_results(global_results, "global_sensemaking_evaluation.json")
                print(f"Evaluated {global_results.get('global_questions_evaluated', 0)} global questions")
        
    except FileNotFoundError:
        print("Answer files not found. Generate answers first using answer_generator.py")
    
    print("\nEvaluation complete!")
    print("\nFiles generated:")
    print("- question_evaluations.json")
    print("- method_comparison_results.json")
    print("- action_specific_evaluation.json (if action questions found)")
    print("- global_sensemaking_evaluation.json (if global questions found)")
