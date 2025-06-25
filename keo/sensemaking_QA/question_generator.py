"""
Sensemaking Question Generator
Generates comprehensive sensemaking questions for aviation maintenance data using OpenAI API
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import random
import time
from data_analyzer import AviationDataAnalyzer


class SensemakingQuestionGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the question generator with OpenAI API
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.generated_questions = []
        
        # Question categories based on GraphRAG methodology
        self.question_categories = {
            'root_cause_analysis': {
                'description': 'Questions that identify underlying causes of maintenance failures',
                'template_starters': [
                    "What are the most frequent underlying causes of",
                    "Which maintenance oversights consistently lead to",
                    "What patterns emerge in incidents involving",
                    "How do multiple factors combine to create",
                    "What are the common denominators in"
                ]
            },
            'predictive_maintenance': {
                'description': 'Questions that help predict and prevent future failures',
                'template_starters': [
                    "Based on historical patterns, what early warning signs indicate",
                    "Which component combinations show the highest risk of",
                    "What maintenance intervals would prevent",
                    "How can we predict when",
                    "What indicators suggest imminent"
                ]
            },
            'safety_recommendations': {
                'description': 'Questions that generate actionable safety improvements',
                'template_starters': [
                    "What are the top 5 procedural changes that would prevent",
                    "Which training gaps contribute most to",
                    "What inspection protocols would catch",
                    "How can we improve communication to prevent",
                    "What safety measures would have the greatest impact on"
                ]
            },
            'system_level_understanding': {
                'description': 'Questions that reveal holistic patterns and relationships',
                'template_starters': [
                    "How do weather conditions interact with maintenance deficiencies to create",
                    "What are the relationships between aircraft age, maintenance type, and",
                    "Which operational factors correlate with",
                    "How do different maintenance philosophies impact",
                    "What are the systemic factors that contribute to"
                ]
            },
            'comparative_analysis': {
                'description': 'Questions that compare across different contexts',
                'template_starters': [
                    "How do maintenance-related incidents differ between",
                    "What are the key differences in failure patterns across",
                    "Which aircraft types show the most variation in",
                    "How do incident rates compare between",
                    "What factors distinguish high-risk from low-risk"
                ]
            },
            'trend_analysis': {
                'description': 'Questions about temporal and evolving patterns',
                'template_starters': [
                    "What trends are emerging in",
                    "How have maintenance failure patterns changed over",
                    "What seasonal variations exist in",
                    "How do failure rates correlate with",
                    "What long-term patterns suggest"
                ]
            }
        }
    
    def generate_comprehensive_questions(self, 
                                       analyzer: AviationDataAnalyzer,
                                       questions_per_category: int = 10,
                                       use_data_context: bool = True) -> List[Dict]:
        """
        Generate comprehensive sensemaking questions based on data analysis
        
        Args:
            analyzer: Analyzed aviation data
            questions_per_category: Number of questions per category
            use_data_context: Whether to use actual data patterns in generation
        
        Returns:
            List of generated questions with metadata
        """
        all_questions = []
        
        # Get analysis results for context
        if use_data_context and analyzer.analysis_results:
            data_context = self._prepare_data_context(analyzer.analysis_results)
        else:
            data_context = None
        
        print("Generating sensemaking questions...")
        
        for category, config in self.question_categories.items():
            print(f"\nGenerating {questions_per_category} questions for category: {category}")
            
            category_questions = self._generate_category_questions(
                category=category,
                config=config,
                data_context=data_context,
                num_questions=questions_per_category
            )
            
            all_questions.extend(category_questions)
            
            # Rate limiting
            time.sleep(1)
        
        self.generated_questions = all_questions
        return all_questions
    
    def generate_global_sensemaking_questions(self, 
                                            analyzer: AviationDataAnalyzer,
                                            num_questions: int = 20) -> List[Dict]:
        """
        Generate questions specifically for global sensemaking (GraphRAG style)
        
        Args:
            analyzer: Analyzed aviation data
            num_questions: Number of global questions to generate
        
        Returns:
            List of global sensemaking questions
        """
        print(f"Generating {num_questions} global sensemaking questions...")
        
        # Prepare comprehensive data summary
        data_summary = self._create_comprehensive_data_summary(analyzer)
        
        prompt = f"""
You are an expert in aviation safety and maintenance analysis. Based on the comprehensive aviation maintenance dataset analysis below, generate {num_questions} high-level, global sensemaking questions that require understanding patterns across the entire dataset.

Dataset Summary:
{data_summary}

These questions should be similar to "What are the main themes in the dataset?" or "What are the top 5 most critical safety patterns?" - questions that require synthesizing information from across the entire dataset to identify overarching themes, patterns, and insights.

Focus on questions that:
1. Require holistic understanding of the entire dataset
2. Identify overarching themes and patterns
3. Reveal systemic issues and relationships
4. Support strategic decision-making
5. Cannot be answered by looking at individual records

Generate questions in the following format:
- Each question should be comprehensive and require synthesis across multiple data points
- Questions should address different aspects: causes, patterns, trends, relationships, priorities
- Avoid questions that can be answered with simple counts or individual examples
- Focus on "What are the...", "How do...", "Which factors...", "What patterns..." type questions

Generate exactly {num_questions} questions, each on a new line without numbering:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst specializing in comprehensive data analysis and global pattern recognition."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.8
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Format as structured data
            structured_questions = []
            for i, question in enumerate(questions):
                structured_questions.append({
                    'id': f"global_{i+1:03d}",
                    'category': 'global_sensemaking',
                    'question': question,
                    'type': 'global',
                    'complexity': 'high',
                    'requires_synthesis': True,
                    'data_scope': 'entire_dataset'
                })
            
            return structured_questions
            
        except Exception as e:
            print(f"Error generating global questions: {e}")
            return []
    
    def generate_context_specific_questions(self,
                                          sample_data: pd.DataFrame,
                                          context_type: str,
                                          num_questions: int = 15) -> List[Dict]:
        """
        Generate questions based on specific data samples
        
        Args:
            sample_data: Specific data samples to base questions on
            context_type: Type of context (e.g., 'engine_failures', 'maintenance_text')
            num_questions: Number of questions to generate
        
        Returns:
            List of context-specific questions
        """
        print(f"Generating {num_questions} context-specific questions for {context_type}...")
        
        # Sample representative records
        sample_records = self._sample_representative_records(sample_data, 5)
        
        prompt = f"""
Based on these representative aviation maintenance records from the {context_type} context:

{sample_records}

Generate {num_questions} sensemaking questions that help understand patterns, causes, and relationships in this specific context. These questions should:

1. Focus on understanding WHY failures occur
2. Identify WHAT can be done to prevent similar issues
3. Explore HOW different factors interact
4. Reveal patterns across similar incidents
5. Support decision-making for maintenance and safety

Each question should require analysis across multiple records to answer comprehensively. Generate exactly {num_questions} questions, each on a new line:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an aviation maintenance expert focused on identifying actionable insights from incident data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Format as structured data
            structured_questions = []
            for i, question in enumerate(questions):
                structured_questions.append({
                    'id': f"{context_type}_{i+1:03d}",
                    'category': 'context_specific',
                    'context': context_type,
                    'question': question,
                    'type': 'contextual',
                    'complexity': 'medium',
                    'data_scope': context_type
                })
            
            return structured_questions
            
        except Exception as e:
            print(f"Error generating context-specific questions: {e}")
            return []
    
    def _generate_category_questions(self,
                                   category: str,
                                   config: Dict,
                                   data_context: Optional[str],
                                   num_questions: int) -> List[Dict]:
        """Generate questions for a specific category"""
        
        context_prompt = ""
        if data_context:
            context_prompt = f"\nBased on this aviation maintenance data analysis:\n{data_context}\n"
        
        prompt = f"""
You are an expert in aviation safety and maintenance analysis. Generate {num_questions} high-quality sensemaking questions for the category: {category}.

Category Description: {config['description']}

{context_prompt}

The questions should:
1. Require synthesis across multiple data points to answer
2. Focus on actionable insights for aviation safety
3. Be specific to aviation maintenance domain
4. Support strategic decision-making
5. Reveal patterns and relationships in the data

Use these starter patterns as inspiration but create varied, comprehensive questions:
{config['template_starters']}

Generate exactly {num_questions} questions, each on a new line without numbering:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert aviation safety analyst specializing in maintenance data analysis and pattern recognition."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.8
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Format as structured data
            structured_questions = []
            for i, question in enumerate(questions):
                structured_questions.append({
                    'id': f"{category}_{i+1:03d}",
                    'category': category,
                    'question': question,
                    'type': 'analytical',
                    'complexity': 'medium',
                    'requires_synthesis': True
                })
            
            return structured_questions
            
        except Exception as e:
            print(f"Error generating questions for category {category}: {e}")
            return []
    
    def _prepare_data_context(self, analysis_results: Dict) -> str:
        """Prepare data context summary for question generation"""
        context_parts = []
        
        if 'failure_patterns' in analysis_results:
            patterns = analysis_results['failure_patterns']
            context_parts.append("Key Failure Patterns:")
            
            if 'failure_types' in patterns:
                context_parts.append(f"- Failure types distribution: {patterns['failure_types']}")
            
            if 'problem_types' in patterns:
                context_parts.append(f"- Common problems: {patterns['problem_types']}")
            
            if 'action_types' in patterns:
                context_parts.append(f"- Maintenance actions: {patterns['action_types']}")
        
        if 'aircraft_patterns' in analysis_results:
            context_parts.append("\nAircraft Context:")
            aircraft_patterns = analysis_results['aircraft_patterns']
            for key, value in aircraft_patterns.items():
                if isinstance(value, dict) and value:
                    context_parts.append(f"- {key}: {list(value.keys())[:5]}")  # Top 5 items
        
        return "\n".join(context_parts)
    
    def _create_comprehensive_data_summary(self, analyzer: AviationDataAnalyzer) -> str:
        """Create comprehensive summary for global questions"""
        summary_parts = []
        
        # Dataset overview
        summary_parts.append("Dataset Overview:")
        summary_parts.append(f"- Total datasets: {len(analyzer.datasets)}")
        for name, df in analyzer.datasets.items():
            summary_parts.append(f"- {name}: {len(df)} records")
        
        # Analysis results summary
        if analyzer.analysis_results:
            summary_parts.append("\nKey Findings:")
            
            if 'failure_patterns' in analyzer.analysis_results:
                patterns = analyzer.analysis_results['failure_patterns']
                if 'failure_types' in patterns:
                    summary_parts.append(f"- Primary failure types: {patterns['failure_types']}")
                if 'problem_types' in patterns:
                    summary_parts.append(f"- Common maintenance problems: {patterns['problem_types']}")
            
            if 'sensemaking_themes' in analyzer.analysis_results:
                themes = analyzer.analysis_results['sensemaking_themes']
                summary_parts.append(f"- Identified {len(themes)} major analytical themes")
                for theme in themes:
                    summary_parts.append(f"  * {theme['category']}: {theme['description']}")
        
        return "\n".join(summary_parts)
    
    def _sample_representative_records(self, data: pd.DataFrame, num_samples: int) -> str:
        """Sample representative records from data"""
        if len(data) == 0:
            return "No data available"
        
        # Sample records
        sample_size = min(num_samples, len(data))
        sampled = data.sample(n=sample_size) if len(data) > sample_size else data
        
        # Format for prompt
        records = []
        for idx, row in sampled.iterrows():
            # Get text columns that contain meaningful content
            text_content = []
            for col in row.index:
                if pd.notna(row[col]) and isinstance(row[col], str) and len(str(row[col]).strip()) > 10:
                    text_content.append(f"{col}: {str(row[col]).strip()}")
            
            if text_content:
                records.append("Record: " + " | ".join(text_content[:3]))  # Limit to 3 fields
        
        return "\n".join(records[:num_samples])
    
    def save_questions(self, output_path: str, format: str = 'json') -> None:
        """Save generated questions to file"""
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.generated_questions, f, indent=2)
            elif format.lower() == 'csv':
                df = pd.DataFrame(self.generated_questions)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Format must be 'json' or 'csv'")
            
            print(f"Questions saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving questions: {e}")
    
    def get_questions_summary(self) -> Dict:
        """Get summary of generated questions"""
        if not self.generated_questions:
            return {"total_questions": 0}
        
        categories = {}
        for q in self.generated_questions:
            cat = q.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_questions": len(self.generated_questions),
            "categories": categories,
            "question_types": list(set(q.get('type', 'unknown') for q in self.generated_questions))
        }
    
    def generate_action_specific_questions(self, 
                                         analyzer: AviationDataAnalyzer,
                                         num_questions: int = 50) -> List[Dict]:
        """
        Generate action-specific questions from Aircraft_Annotation_DataFile.csv
        Creates practical "what to do when..." questions with ground truth answers
        
        Args:
            analyzer: Analyzed aviation data
            num_questions: Number of action-specific questions to generate
        
        Returns:
            List of action-specific questions with ground truth answers
        """
        print(f"Generating {num_questions} action-specific questions from aircraft annotation data...")
        
        action_questions = []
        
        if 'aircraft_annotation' not in analyzer.datasets:
            print("Warning: Aircraft annotation data not available for action-specific questions")
            return action_questions
        
        annotation_data = analyzer.datasets['aircraft_annotation']
        
        # Filter out rows with valid PROBLEM and ACTION data
        valid_data = annotation_data.dropna(subset=['PROBLEM', 'ACTION'])
        valid_data = valid_data[
            (valid_data['PROBLEM'].str.strip() != '') & 
            (valid_data['ACTION'].str.strip() != '')
        ]
        
        if len(valid_data) == 0:
            print("Warning: No valid problem-action pairs found")
            return action_questions
        
        # Sample diverse problem-action pairs
        sample_size = min(num_questions * 2, len(valid_data))  # Sample more to ensure diversity
        sampled_data = valid_data.sample(n=sample_size, random_state=42)
        
        # Group similar problems to avoid duplicates
        unique_problems = {}
        for _, row in sampled_data.iterrows():
            problem = str(row['PROBLEM']).strip().upper()
            action = str(row['ACTION']).strip().upper()
            
            # Create a simplified key for grouping similar problems
            problem_key = self._normalize_problem_text(problem)
            
            if problem_key not in unique_problems:
                unique_problems[problem_key] = {
                    'original_problem': problem,
                    'action': action,
                    'count': 1
                }
            else:
                # If we have a more complete action, use it
                if len(action) > len(unique_problems[problem_key]['action']):
                    unique_problems[problem_key]['action'] = action
                unique_problems[problem_key]['count'] += 1
        
        # Convert to action-specific questions
        question_id = 1
        for problem_key, data in unique_problems.items():
            if question_id > num_questions:
                break
                
            problem = data['original_problem']
            action = data['action']
            
            # Create "what to do when..." question
            question_text = self._create_action_question(problem)
            
            if question_text:  # Only add if question creation was successful
                action_questions.append({
                    'id': f"action_{question_id:03d}",
                    'category': 'action_specific',
                    'question': question_text,
                    'ground_truth_answer': action,
                    'original_problem': problem,
                    'type': 'actionable',
                    'complexity': 'low',
                    'requires_synthesis': False,
                    'data_scope': 'specific_incident',
                    'frequency': data['count']
                })
                question_id += 1
        
        print(f"Generated {len(action_questions)} action-specific questions from {len(unique_problems)} unique problems")
        return action_questions
    
    def _normalize_problem_text(self, problem: str) -> str:
        """Normalize problem text for grouping similar issues"""
        # Remove specific identifiers like numbers and standardize
        import re
        
        # Convert to lowercase
        normalized = problem.lower()
        
        # Remove specific numbers (like #1, #2, etc.) but keep general references
        normalized = re.sub(r'#\d+', '#X', normalized)
        
        # Remove specific measurements and replace with generic
        normalized = re.sub(r'\d+', 'X', normalized)
        
        # Standardize common terms
        normalized = re.sub(r'l/h|left', 'left', normalized)
        normalized = re.sub(r'r/h|right', 'right', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _create_action_question(self, problem: str) -> str:
        """Convert a problem statement into a 'what to do when...' question"""
        problem = problem.strip()
        
        if not problem:
            return None
        
        # Clean up the problem text
        problem_lower = problem.lower()
        
        # Remove trailing periods and clean up
        problem_clean = problem.rstrip('.').rstrip(',')
        
        # Create question based on problem structure
        if any(keyword in problem_lower for keyword in ['is leaking', 'are leaking', 'leak']):
            # For leaking issues
            component_part = problem_clean.lower().replace('is leaking', '').replace('are leaking', '').replace('leaking', '').strip()
            if component_part:
                return f"What should be done when {component_part} is leaking?"
            else:
                return f"What should be done when there is a leak?"
        
        elif any(keyword in problem_lower for keyword in ['loose', 'loosened']):
            # For loose components
            component_part = problem_clean.lower().replace('loose', '').replace('loosened', '').strip()
            if component_part:
                return f"What should be done when {component_part} are loose?"
            else:
                return f"What should be done when components are loose?"
        
        elif any(keyword in problem_lower for keyword in ['cracked', 'crack']):
            # For cracked components
            component_part = problem_clean.lower().replace('cracked', '').replace('crack', '').strip()
            if component_part:
                return f"What should be done when {component_part} is cracked?"
            else:
                return f"What should be done when there is a crack?"
        
        elif any(keyword in problem_lower for keyword in ['fouled', 'fouling']):
            # For fouled components
            component_part = problem_clean.lower().replace('fouled', '').replace('fouling', '').strip()
            if component_part:
                return f"What should be done when {component_part} is fouled?"
            else:
                return f"What should be done when components are fouled?"
        
        elif any(keyword in problem_lower for keyword in ['failed', 'failure', 'malfunction']):
            # For failures
            component_part = problem_clean.lower()
            for term in ['failed', 'failure', 'malfunction']:
                component_part = component_part.replace(term, '').strip()
            if component_part:
                return f"What should be done when {component_part} fails?"
            else:
                return f"What should be done when there is a component failure?"
        
        else:
            # Generic case - create question from the problem description
            return f"What should be done when: {problem_clean.lower()}?"
    

if __name__ == "__main__":
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize components (using data for question generation only)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    # Analyze data
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    analyzer.analyze_failure_patterns()
    analyzer.analyze_aircraft_types()
    analyzer.identify_sensemaking_themes()
    
    # Generate questions
    generator = SensemakingQuestionGenerator(openai_api_key)
    
    # Generate comprehensive questions
    comprehensive_questions = generator.generate_comprehensive_questions(
        analyzer, questions_per_category=8
    )
    
    # Generate global sensemaking questions
    global_questions = generator.generate_global_sensemaking_questions(
        analyzer, num_questions=15
    )
    
    # Generate action-specific questions (NEW: practical "what to do when..." questions)
    action_questions = generator.generate_action_specific_questions(
        analyzer, num_questions=50
    )
    
    # Generate context-specific questions
    if 'aircraft_annotation' in analyzer.datasets:
        context_questions = generator.generate_context_specific_questions(
            analyzer.datasets['aircraft_annotation'], 
            'maintenance_actions', 
            num_questions=10
        )
        comprehensive_questions.extend(context_questions)
    
    # Combine all questions
    all_questions = comprehensive_questions + global_questions + action_questions
    generator.generated_questions = all_questions
    
    # Save results
    generator.save_questions("aviation_sensemaking_questions.json")
    generator.save_questions("aviation_sensemaking_questions.csv", format='csv')
    
    # Print summary
    summary = generator.get_questions_summary()
    print(f"\nGenerated {summary['total_questions']} total questions")
    print("Questions by category:")
    for category, count in summary['categories'].items():
        print(f"  {category}: {count}")
