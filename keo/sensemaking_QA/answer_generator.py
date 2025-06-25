"""
GraphRAG-style Answer Generator for Aviation Maintenance Sensemaking Questions
Generates answers using both vanilla LLM and GraphRAG approaches
"""

import os
import json
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import time
import sys
sys.path.append('..')
from graph_rag.KEO_GraphRAG import GraphRetriever


class SensemakingAnswerGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the answer generator
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.knowledge_graph = None
        self.graph_retriever = None
        
    def load_knowledge_graph(self, graph_path: str) -> bool:
        """Load aviation maintenance knowledge graph"""
        try:
            self.knowledge_graph = nx.read_gml(graph_path)
            self.graph_retriever = GraphRetriever(self.knowledge_graph, os.getenv("OPENAI_API_KEY"))
            self.graph_retriever.generate_embeddings()
            print(f"Knowledge graph loaded: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            return False
    
    def generate_vanilla_answers(self, 
                               questions: List[Dict],
                               datasets: Dict[str, pd.DataFrame],
                               context_window_size: int = 4000) -> List[Dict]:
        """
        Generate answers using vanilla LLM approach with basic context
        
        Args:
            questions: List of question dictionaries
            datasets: Aviation maintenance datasets
            context_window_size: Max characters for context
        
        Returns:
            List of answers with metadata
        """
        print("Generating vanilla LLM answers...")
        answers = []
        
        # Prepare general context from datasets
        general_context = self._prepare_general_context(datasets, context_window_size)
        
        for question_data in tqdm(questions, desc="Generating vanilla answers"):
            try:
                question = question_data['question']
                
                # Basic context retrieval
                relevant_context = self._retrieve_basic_context(
                    question, datasets, context_window_size
                )
                
                prompt = f"""
You are an expert aviation safety analyst. Answer the following question based on the provided aviation maintenance data context.

Question: {question}

Context from Aviation Maintenance Data:
{relevant_context}

General Dataset Information:
{general_context}

Provide a comprehensive, analytical answer that:
1. Directly addresses the question
2. Uses specific examples from the data when relevant
3. Identifies patterns and relationships
4. Provides actionable insights
5. Maintains focus on aviation safety and maintenance

Answer:
"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert aviation safety analyst with deep knowledge of maintenance practices, failure analysis, and safety recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content.strip()
                
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question,
                    'category': question_data.get('category', ''),
                    'answer': answer,
                    'method': 'vanilla_llm',
                    'context_used': len(relevant_context),
                    'model': self.model
                })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating vanilla answer for question {question_data.get('id', '')}: {e}")
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question_data['question'],
                    'answer': f"Error generating answer: {e}",
                    'method': 'vanilla_llm',
                    'error': True
                })
        
        return answers
    
    def generate_graphrag_answers(self, 
                                questions: List[Dict],
                                datasets: Dict[str, pd.DataFrame],
                                max_path_length: int = 3,
                                top_k_nodes: int = 10) -> List[Dict]:
        """
        Generate answers using GraphRAG approach
        
        Args:
            questions: List of question dictionaries
            datasets: Aviation maintenance datasets
            max_path_length: Maximum path length for graph traversal
            top_k_nodes: Number of top relevant nodes to consider
        
        Returns:
            List of answers with graph-based context
        """
        if not self.knowledge_graph or not self.graph_retriever:
            print("Knowledge graph not loaded. Cannot generate GraphRAG answers.")
            return []
        
        print("Generating GraphRAG answers...")
        answers = []
        
        for question_data in tqdm(questions, desc="Generating GraphRAG answers"):
            try:
                question = question_data['question']
                
                # Graph-based context retrieval
                graph_context = self._retrieve_graph_context(
                    question, max_path_length, top_k_nodes
                )
                
                # Community-based context (GraphRAG style)
                community_context = self._get_community_summaries(question)
                
                # Combine with dataset context
                dataset_context = self._retrieve_basic_context(
                    question, datasets, 2000  # Smaller limit for GraphRAG
                )
                
                prompt = f"""
You are an expert aviation safety analyst using graph-based knowledge retrieval. Answer the following question using the provided graph-structured knowledge and data context.

Question: {question}

Graph-Based Knowledge Context:
{graph_context}

Community Insights:
{community_context}

Supporting Data Context:
{dataset_context}

Provide a comprehensive answer that:
1. Leverages the graph-structured relationships and patterns
2. Synthesizes information from multiple knowledge sources
3. Identifies complex interactions and dependencies
4. Provides evidence-based insights with graph provenance
5. Offers strategic recommendations based on systemic understanding

Ensure your answer demonstrates the value of graph-based reasoning for this complex question.

Answer:
"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert aviation safety analyst specializing in graph-based knowledge analysis and systemic pattern recognition in maintenance data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content.strip()
                
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question,
                    'category': question_data.get('category', ''),
                    'answer': answer,
                    'method': 'graphrag',
                    'graph_nodes_used': len(graph_context.split('\n')),
                    'model': self.model,
                    'graph_available': True
                })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating GraphRAG answer for question {question_data.get('id', '')}: {e}")
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question_data['question'],
                    'answer': f"Error generating answer: {e}",
                    'method': 'graphrag',
                    'error': True
                })
        
        return answers
    
    def generate_comparative_analysis(self,
                                    vanilla_answers: List[Dict],
                                    graphrag_answers: List[Dict]) -> List[Dict]:
        """
        Generate comparative analysis between vanilla and GraphRAG answers
        
        Args:
            vanilla_answers: Answers from vanilla LLM
            graphrag_answers: Answers from GraphRAG
        
        Returns:
            List of comparative analyses
        """
        print("Generating comparative analysis...")
        comparisons = []
        
        # Match answers by question_id
        vanilla_dict = {ans['question_id']: ans for ans in vanilla_answers if not ans.get('error')}
        graphrag_dict = {ans['question_id']: ans for ans in graphrag_answers if not ans.get('error')}
        
        common_ids = set(vanilla_dict.keys()) & set(graphrag_dict.keys())
        
        for question_id in tqdm(common_ids, desc="Comparing answers"):
            try:
                vanilla_ans = vanilla_dict[question_id]
                graphrag_ans = graphrag_dict[question_id]
                
                prompt = f"""
Compare and analyze these two answers to the same aviation maintenance question:

Question: {vanilla_ans['question']}

Vanilla LLM Answer:
{vanilla_ans['answer']}

GraphRAG Answer:
{graphrag_ans['answer']}

Provide a detailed comparison analysis covering:

1. Comprehensiveness: Which answer is more complete?
2. Specificity: Which provides more specific, actionable insights?
3. Evidence Use: How well does each use supporting evidence?
4. Systemic Understanding: Which better captures complex relationships?
5. Practical Value: Which is more useful for aviation safety professionals?

Rate each answer on these dimensions (1-5 scale) and provide an overall assessment.

Analysis:
"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of aviation safety analysis, skilled in assessing the quality and utility of analytical responses."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.2
                )
                
                analysis = response.choices[0].message.content.strip()
                
                comparisons.append({
                    'question_id': question_id,
                    'question': vanilla_ans['question'],
                    'category': vanilla_ans.get('category', ''),
                    'vanilla_answer': vanilla_ans['answer'],
                    'graphrag_answer': graphrag_ans['answer'],
                    'comparative_analysis': analysis,
                    'evaluation_model': self.model
                })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating comparison for question {question_id}: {e}")
        
        return comparisons
    
    def _prepare_general_context(self, datasets: Dict[str, pd.DataFrame], max_chars: int) -> str:
        """Prepare general context about the datasets"""
        context_parts = []
        
        context_parts.append("Aviation Maintenance Dataset Overview:")
        
        for name, df in datasets.items():
            context_parts.append(f"\n{name.upper()} ({len(df)} records):")
            
            # Sample some representative content
            if name == 'faa_sample' and 'c119' in df.columns:
                sample_incidents = df['c119'].dropna().sample(min(3, len(df))).tolist()
                context_parts.append("Sample incidents:")
                for incident in sample_incidents:
                    context_parts.append(f"- {str(incident)[:100]}...")
            
            elif name == 'aircraft_annotation' and 'PROBLEM' in df.columns:
                sample_problems = df['PROBLEM'].dropna().sample(min(3, len(df))).tolist()
                context_parts.append("Sample maintenance problems:")
                for problem in sample_problems:
                    context_parts.append(f"- {str(problem)[:80]}...")
        
        context = "\n".join(context_parts)
        return context[:max_chars] if len(context) > max_chars else context
    
    def _retrieve_basic_context(self, 
                              question: str, 
                              datasets: Dict[str, pd.DataFrame], 
                              max_chars: int) -> str:
        """Retrieve basic context relevant to the question"""
        question_lower = question.lower()
        context_parts = []
        
        # Simple keyword-based retrieval
        keywords = self._extract_question_keywords(question_lower)
        
        for dataset_name, df in datasets.items():
            relevant_records = []
            
            # Search in text columns
            for col in df.columns:
                if df[col].dtype == 'object':  # Text columns
                    mask = df[col].astype(str).str.lower().str.contains(
                        '|'.join(keywords), na=False, regex=True
                    )
                    if mask.any():
                        relevant_records.extend(
                            df[mask][col].dropna().head(3).tolist()
                        )
            
            if relevant_records:
                context_parts.append(f"\nRelevant {dataset_name} records:")
                for record in relevant_records[:5]:  # Limit records
                    context_parts.append(f"- {str(record)[:150]}...")
        
        context = "\n".join(context_parts)
        return context[:max_chars] if len(context) > max_chars else context
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from question"""
        # Common aviation/maintenance terms to prioritize
        aviation_terms = [
            'engine', 'fuel', 'brake', 'gear', 'hydraulic', 'electrical',
            'control', 'leak', 'crack', 'fail', 'malfunction', 'maintenance',
            'inspection', 'repair', 'replace', 'aircraft', 'flight', 'safety'
        ]
        
        words = question.lower().split()
        keywords = []
        
        # Add aviation-specific terms found in question
        for term in aviation_terms:
            if term in question:
                keywords.append(term)
        
        # Add other meaningful words (longer than 4 chars, not common words)
        common_words = {'what', 'how', 'why', 'when', 'where', 'which', 'that', 'this', 'these', 'those', 'with', 'from', 'they', 'them', 'have', 'been', 'will', 'would', 'could', 'should'}
        for word in words:
            if len(word) > 4 and word not in common_words and word not in keywords:
                keywords.append(word)
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _retrieve_graph_context(self, 
                              question: str, 
                              max_path_length: int, 
                              top_k_nodes: int) -> str:
        """Retrieve context using graph structure"""
        if not self.graph_retriever:
            return "Graph retriever not available"
        
        try:
            # Generate question embedding
            question_response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=question,
                encoding_format="float"
            )
            question_embedding = question_response.data[0].embedding
            
            # Find most relevant nodes
            node_similarities = []
            for node in self.knowledge_graph.nodes():
                if node in self.graph_retriever.embeddings:
                    node_embedding = self.graph_retriever.embeddings[node]
                    similarity = self.graph_retriever.calculate_semantic_similarity(
                        question_embedding, node_embedding
                    )
                    node_similarities.append((node, similarity))
            
            # Get top-k most relevant nodes
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in node_similarities[:top_k_nodes]]
            
            # Build context from graph structure
            context_parts = []
            context_parts.append("Relevant Knowledge Graph Information:")
            
            for node in top_nodes:
                # Node information
                node_attrs = self.knowledge_graph.nodes[node]
                context_parts.append(f"\nNode: {node}")
                if node_attrs:
                    context_parts.append(f"Attributes: {node_attrs}")
                
                # Connected nodes and relationships
                neighbors = list(self.knowledge_graph.neighbors(node))
                if neighbors:
                    context_parts.append("Connected to:")
                    for neighbor in neighbors[:3]:  # Limit connections
                        edge_data = self.knowledge_graph.get_edge_data(node, neighbor)
                        rel_type = edge_data.get('relationship', 'related') if edge_data else 'related'
                        context_parts.append(f"  - {neighbor} ({rel_type})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error retrieving graph context: {e}")
            return "Error retrieving graph context"
    
    def _get_community_summaries(self, question: str) -> str:
        """Get community-based summaries (simplified GraphRAG approach)"""
        if not self.knowledge_graph:
            return "No graph available for community analysis"
        
        try:
            # Convert directed graph to undirected for community detection
            if self.knowledge_graph.is_directed():
                # Convert to undirected graph for community detection
                undirected_graph = self.knowledge_graph.to_undirected()
                communities = list(nx.connected_components(undirected_graph))
            else:
                # Use original graph if already undirected
                communities = list(nx.connected_components(self.knowledge_graph))
            
            context_parts = []
            context_parts.append("Community-based Knowledge Summary:")
            
            # Analyze largest communities
            largest_communities = sorted(communities, key=len, reverse=True)[:3]
            
            for i, community in enumerate(largest_communities):
                context_parts.append(f"\nCommunity {i+1} ({len(community)} nodes):")
                
                # Sample nodes from community
                sample_nodes = list(community)[:5]
                context_parts.append(f"Key entities: {', '.join(sample_nodes)}")
                
                # Analyze relationships within community using original graph
                subgraph = self.knowledge_graph.subgraph(community)
                relationships = set()
                for u, v, data in subgraph.edges(data=True):
                    rel_type = data.get('relationship', 'related')
                    relationships.add(rel_type)
                
                if relationships:
                    context_parts.append(f"Relationship types: {', '.join(relationships)}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error getting community summaries: {e}")
            return "Error analyzing communities"
    
    def save_answers(self, answers: List[Dict], output_path: str, format: str = 'json') -> None:
        """Save answers to file"""
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(answers, f, indent=2)
            elif format.lower() == 'csv':
                df = pd.DataFrame(answers)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Format must be 'json' or 'csv'")
            
            print(f"Answers saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving answers: {e}")


if __name__ == "__main__":
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize answer generator
    generator = SensemakingAnswerGenerator(openai_api_key)
    
    # Load knowledge graph if available
    graph_path = "../kg/knowledge_graph.gml"
    if os.path.exists(graph_path):
        generator.load_knowledge_graph(graph_path)
    
    # Load questions
    try:
        with open("aviation_sensemaking_questions.json", 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions")
    except FileNotFoundError:
        print("Questions file not found. Please run question_generator.py first.")
        questions = []
    
    # Load datasets (excluding files used for KG construction)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    datasets = {}
    for name, path in data_paths.items():
        try:
            datasets[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(datasets[name])} records")
        except FileNotFoundError:
            print(f"Dataset {name} not found at {path}")
    
    if questions and datasets:
        # Sample questions for testing (use first 5)
        sample_questions = questions[:5]
        
        # Generate vanilla answers
        vanilla_answers = generator.generate_vanilla_answers(sample_questions, datasets)
        generator.save_answers(vanilla_answers, "vanilla_answers.json")
        
        # Generate GraphRAG answers if graph is available
        if generator.knowledge_graph:
            graphrag_answers = generator.generate_graphrag_answers(sample_questions, datasets)
            generator.save_answers(graphrag_answers, "graphrag_answers.json")
            
            # Generate comparative analysis
            comparisons = generator.generate_comparative_analysis(vanilla_answers, graphrag_answers)
            generator.save_answers(comparisons, "answer_comparisons.json")
        
        print("Answer generation complete!")
