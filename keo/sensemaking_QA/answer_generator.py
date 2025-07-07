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
import hashlib
import pickle
sys.path.append('..')
from graph_rag.KEO_GraphRAG import GraphRetriever


class SensemakingAnswerGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o", cache_dir: str = "./embedding_cache"):
        """
        Initialize the answer generator
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
            cache_dir: Directory to store embedding cache files
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.knowledge_graph = None
        self.graph_retriever = None
        self.cache_dir = cache_dir
        self.embedding_cache = {}
        self.chunk_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache if available
        self._load_cache()
        
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
                               questions: List[Dict]) -> List[Dict]:
        """
        Generate answers using vanilla LLM with simple prompts
        Args:
            questions: List of question dictionaries
        Returns:
            List of answers with metadata
        """
        print("Generating vanilla LLM answers...")
        answers = []
        
        for question_data in tqdm(questions, desc="Generating vanilla answers"):
            try:
                question = question_data['question']
                question_type = question_data.get('type', '')
                
                if question_type == 'actionable':
                    prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action.

Question: {question}

Provide a brief, specific answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Follows standard aviation maintenance procedures, and only gives the action to take, not the reasoning or background information

One example:
Q: what should be done when: engine oil leak detected?
A: Check the oil level, inspect for leaks, and replace any damaged seals or gaskets.

Answer:
"""
                else:
                    prompt = f"""
You are an expert aviation safety analyst. Answer the following question about aviation maintenance.

Question: {question}

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
                    max_tokens=300 if question_type == 'actionable' else 1000,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content.strip()
                
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question,
                    'category': question_data.get('category', ''),
                    'answer': answer,
                    'method': 'vanilla_llm',
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


    def generate_textchunkrag_answers(self, 
                               questions: List[Dict],
                               datasets: Dict[str, pd.DataFrame],
                               context_window_size: int = 4000) -> List[Dict]:
        """
        Generate answers using text chunking based RAG approach
        
        Args:
            questions: List of question dictionaries
            datasets: Aviation maintenance datasets
            context_window_size: Max characters for context
        
        Returns:
            List of answers with metadata
        """
        print("Generating text-chunk RAG answers...")
        answers = []
        
        for question_data in tqdm(questions, desc="Generating text-chunk RAG answers"):
            try:
                question = question_data['question']
                question_type = question_data.get('type', '')
                
                # Basic context retrieval
                relevant_context = self._retrieve_basic_context(
                    question, datasets, context_window_size, top_k=5
                )
                
                if question_type == 'actionable':
                    prompt = f"""
You are an expert aviation maintenance technician. Answer the following question with a direct, concise action based on the provided aviation maintenance data context.

Question: {question}

Context from Aviation Maintenance Data:
{relevant_context}

Provide a brief, specific answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Follows standard aviation maintenance procedures, and only gives the action to take, not the reasoning or background information
4. Uses information from the context when relevant

One example:
Q: what should be done when: engine oil leak detected?
A: Check the oil level, inspect for leaks, and replace any damaged seals or gaskets.

Answer:
"""
                else:
                    prompt = f"""
You are an expert aviation safety analyst. Answer the following question based on the provided aviation maintenance data context.

Question: {question}

Context from Aviation Maintenance Data:
{relevant_context}

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
                    max_tokens=300 if question_type == 'actionable' else 1000,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content.strip()
                
                answers.append({
                    'question_id': question_data.get('id', ''),
                    'question': question,
                    'category': question_data.get('category', ''),
                    'answer': answer,
                    'method': 'textchunkrag',
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
                    'method': 'textchunkrag',
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
                question_type = question_data.get('type', '')
                
                # Graph-based context retrieval
                graph_context = self._retrieve_graph_context(
                    question, max_path_length, top_k_nodes
                )
                
                # Community-based context (GraphRAG style)
                community_context = self._get_community_summaries(question)
                
                # Combine with dataset context
                dataset_context = self._retrieve_basic_context(
                    question, datasets, 2000, top_k=3  # Smaller limit for GraphRAG
                )
                
                if question_type == 'actionable':
                    prompt = f"""
You are an expert aviation maintenance technician using graph-based knowledge retrieval. Answer the following question with a direct, concise action based on the provided graph-structured knowledge and data context.

Question: {question}

Graph-Based Knowledge Context:
{graph_context}

Community Insights:
{community_context}

Supporting Data Context:
{dataset_context}

Provide a brief, specific answer that:
1. States the exact maintenance action to take
2. Is clear and actionable
3. Follows standard aviation maintenance procedures, and only gives the action to take, not the reasoning or background information
4. Uses relevant graph-based insights when applicable

One example:
Q: what should be done when: engine oil leak detected?
A: Check the oil level, inspect for leaks, and replace any damaged seals or gaskets.

Answer:
"""
                else:
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
                    max_tokens=300 if question_type == 'actionable' else 1200,
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
    
    def _retrieve_basic_context(self, 
                              question: str, 
                              datasets: Dict[str, pd.DataFrame], 
                              max_chars: int,
                              top_k: int = 3,
                              chunk_size: int = 500) -> str:
        """Retrieve basic context relevant to the question using text chunk-based RAG
        
        Args:
            question: The question to find relevant context for
            datasets: Dictionary of datasets to search through
            max_chars: Maximum characters to return
            top_k: Number of most similar chunks to retrieve per dataset
            chunk_size: Size of text chunks for embedding
        """
        context_parts = []
        
        try:
            # Generate question embedding (with caching)
            question_embedding = self._get_embedding_cached(question)
            if question_embedding is None:
                raise Exception("Could not generate question embedding")
            
            # Check if we have cached chunks for this dataset combination
            dataset_cache_key = self._get_dataset_cache_key(datasets, chunk_size)
            
            if dataset_cache_key in self.chunk_cache:
                cached_data = self.chunk_cache[dataset_cache_key]
            else:
                print("→ Generating new text chunks and embeddings...")
                cached_data = {}
                
                # Text chunking and embedding for each dataset
                for dataset_name, df in datasets.items():
                    chunks = self._create_text_chunks(df, chunk_size=chunk_size)
                    
                    if not chunks:
                        continue
                    
                    # Generate embeddings for chunks (with caching)
                    chunk_embeddings = []
                    for chunk in tqdm(chunks, desc=f"Processing {dataset_name} chunks"):
                        embedding = self._get_embedding_cached(chunk)
                        chunk_embeddings.append(embedding)
                    
                    # Store in cache
                    cached_data[dataset_name] = {
                        'chunks': chunks,
                        'embeddings': chunk_embeddings
                    }
                
                # Save the dataset cache
                self.chunk_cache[dataset_cache_key] = cached_data
                self._save_cache()
            
            # Text chunking and retrieval for each dataset
            for dataset_name, dataset_data in cached_data.items():
                chunks = dataset_data['chunks']
                chunk_embeddings = dataset_data['embeddings']
                
                # Calculate similarities and find most relevant chunks
                chunk_similarities = []
                for i, chunk_embedding in enumerate(chunk_embeddings):
                    if chunk_embedding is not None:
                        similarity = self._calculate_cosine_similarity(
                            question_embedding, chunk_embedding
                        )
                        chunk_similarities.append((i, similarity, chunks[i]))
                
                # Sort by similarity and select top chunks
                chunk_similarities.sort(key=lambda x: x[1], reverse=True)
                top_chunks = chunk_similarities[:top_k]  # Top k most relevant chunks
                
                if top_chunks:
                    context_parts.append(f"\nRelevant {dataset_name} context:")
                    for _, similarity, chunk in top_chunks:
                        context_parts.append(f"- {chunk[:200]}...")
                        
        except Exception as e:
            print(f"Error in text chunk retrieval: {e}")
            # Fallback to simple text extraction
            for dataset_name, df in datasets.items():
                text_data = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_data.extend(df[col].dropna().astype(str).head(3).tolist())
                
                if text_data:
                    context_parts.append(f"\nRelevant {dataset_name} records:")
                    for record in text_data[:3]:
                        context_parts.append(f"- {record[:150]}...")
        
        context = "\n".join(context_parts)
        return context[:max_chars] if len(context) > max_chars else context
        
        context = "\n".join(context_parts)
        return context[:max_chars] if len(context) > max_chars else context
    

    
    def _retrieve_graph_context(self, 
                              question: str, 
                              max_path_length: int, 
                              top_k_nodes: int) -> str:
        """
        Retrieve context using weighted graph structure with maximum spanning tree approach
        
        Args:
            question: The question to find relevant context for
            max_path_length: Maximum hops for neighbor expansion (m-hop)
            top_k_nodes: Number of top relevant nodes to consider
            
        Returns:
            Textual representation of the most relevant graph knowledge
        """
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
            
            # Step 1: Create subtrees containing these top-k nodes
            subtree_nodes = set(top_nodes)
            
            # Step 2: Expand with m-hop neighbors
            for hop in range(max_path_length):
                new_nodes = set()
                for node in subtree_nodes:
                    if node in self.knowledge_graph:
                        neighbors = list(self.knowledge_graph.neighbors(node))
                        new_nodes.update(neighbors)
                subtree_nodes.update(new_nodes)
            
            # Step 3: Extract subgraph with all relevant nodes
            relevant_subgraph = self.knowledge_graph.subgraph(subtree_nodes)
            
            # Step 4: Find connected components in the subgraph
            if relevant_subgraph.is_directed():
                # Convert to undirected for component analysis
                undirected_subgraph = relevant_subgraph.to_undirected()
                connected_components = list(nx.connected_components(undirected_subgraph))
            else:
                connected_components = list(nx.connected_components(relevant_subgraph))
            
            # Step 5: Process each connected component
            context_narratives = []
            
            for i, component in enumerate(connected_components):
                if len(component) < 2:  # Skip isolated nodes
                    continue
                    
                # Extract component subgraph
                component_subgraph = relevant_subgraph.subgraph(component)
                
                # Convert to undirected for MST if needed
                if component_subgraph.is_directed():
                    component_undirected = component_subgraph.to_undirected()
                else:
                    component_undirected = component_subgraph
                
                # Step 6: Find Maximum Spanning Tree for this component
                mst = self._find_maximum_spanning_tree(component_undirected)
                
                if mst and len(mst.edges()) > 0:
                    # Step 7: Generate narrative from MST using DFS
                    narrative = self._generate_narrative_from_mst(mst, component_subgraph)
                    if narrative:
                        context_narratives.append(f"Knowledge Path {i+1}:\n{narrative}")
            
            # Combine all narratives
            if context_narratives:
                return "Graph-based Knowledge Context:\n\n" + "\n\n".join(context_narratives)
            else:
                return "No significant graph patterns found for this question."
                
        except Exception as e:
            print(f"Error retrieving graph context: {e}")
            return "Error retrieving graph context"
    
    def _find_maximum_spanning_tree(self, graph: nx.Graph) -> nx.Graph:
        """
        Find Maximum Spanning Tree using edge weights
        
        Args:
            graph: Undirected graph with edge weights
            
        Returns:
            Maximum spanning tree as a graph
        """
        try:
            # Ensure we have a connected graph
            if not nx.is_connected(graph):
                # Get the largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_cc)
            
            # Check if edges have weights
            has_weights = any('weight' in data for _, _, data in graph.edges(data=True))
            
            if has_weights:
                # For maximum spanning tree, we need to negate weights for minimum spanning tree algorithm
                # or use maximum_spanning_tree if available
                try:
                    mst = nx.maximum_spanning_tree(graph, weight='weight')
                except:
                    # Fallback: negate weights and use minimum spanning tree
                    temp_graph = graph.copy()
                    for u, v, data in temp_graph.edges(data=True):
                        if 'weight' in data:
                            data['weight'] = -data['weight']
                    mst = nx.minimum_spanning_tree(temp_graph, weight='weight')
                    # Restore original weights
                    for u, v, data in mst.edges(data=True):
                        if 'weight' in data:
                            data['weight'] = -data['weight']
            else:
                # If no weights, treat all edges as weight 1
                for u, v in graph.edges():
                    graph[u][v]['weight'] = 1.0
                mst = nx.maximum_spanning_tree(graph, weight='weight')
            
            return mst
            
        except Exception as e:
            print(f"Error finding MST: {e}")
            return nx.Graph()
    
    def _generate_narrative_from_mst(self, mst: nx.Graph, original_graph: nx.Graph) -> str:
        """
        Generate textual narrative from MST using DFS traversal
        
        Args:
            mst: Maximum spanning tree
            original_graph: Original graph with all edge attributes
            
        Returns:
            Textual narrative describing the knowledge path
        """
        try:
            if len(mst.edges()) == 0:
                return ""
            
            # Find the edge with maximum weight to start DFS
            max_weight = -float('inf')
            start_edge = None
            
            for u, v, data in mst.edges(data=True):
                weight = data.get('weight', 0)
                if weight > max_weight:
                    max_weight = weight
                    start_edge = (u, v)
            
            if not start_edge:
                return ""
            
            # Start DFS from one node of the maximum weight edge
            start_node = start_edge[0]
            visited = set()
            narrative_parts = []
            
            def dfs_narrative(node, parent=None):
                if node in visited:
                    return
                
                visited.add(node)
                
                # Add node information
                node_attrs = original_graph.nodes.get(node, {})
                node_desc = self._format_node_description(node, node_attrs)
                
                if parent is not None:
                    # Add edge information
                    edge_data = original_graph.get_edge_data(parent, node) or original_graph.get_edge_data(node, parent)
                    relationship = self._format_relationship_description(parent, node, edge_data)
                    narrative_parts.append(f"{relationship} {node_desc}")
                else:
                    narrative_parts.append(f"{node_desc}")
                
                # Visit neighbors in order of edge weight (descending)
                neighbors = []
                for neighbor in mst.neighbors(node):
                    if neighbor not in visited:
                        edge_data = mst.get_edge_data(node, neighbor)
                        weight = edge_data.get('weight', 0) if edge_data else 0
                        neighbors.append((neighbor, weight))
                
                # Sort by weight (descending) for priority traversal
                neighbors.sort(key=lambda x: x[1], reverse=True)
                
                for neighbor, _ in neighbors:
                    dfs_narrative(neighbor, node)
            
            # Start DFS traversal
            dfs_narrative(start_node)
            
            # Join narrative parts into coherent text with natural flow
            if narrative_parts:
                # Create a more natural narrative flow
                narrative_text = narrative_parts[0]  # Start with first element
                for i in range(1, len(narrative_parts)):
                    narrative_text += f", which {narrative_parts[i]}"
                return narrative_text
            else:
                return ""
                
        except Exception as e:
            print(f"Error generating narrative: {e}")
            return ""
    
    def _format_node_description(self, node: str, attributes: Dict) -> str:
        """Format node information for narrative as simple text"""
        # Filter out technical IDs and unwanted attributes including all incident-related fields
        excluded_keys = {'incident_id', 'qid', 'id', 'node_id', 'type', 'incident_ids', 'incident ids'}
        
        if attributes:
            # Start with the node name (cleaned)
            clean_node = node.replace('_', ' ').replace('-', ' ').title()
            
            # Extract meaningful attributes, excluding technical IDs and incident references
            meaningful_attrs = []
            for key, value in attributes.items():
                key_lower = str(key).lower()
                if (key_lower not in excluded_keys and 
                    value and str(value).strip() and 
                    not key_lower.endswith('_id') and
                    not key_lower.endswith('_ids') and
                    not key_lower.startswith('id_') and
                    not key_lower.startswith('incident') and
                    'incident' not in key_lower):
                    
                    # Clean up the attribute value
                    clean_value = str(value).strip()
                    if len(clean_value) > 50:
                        clean_value = clean_value[:50] + "..."
                    
                    # Format as natural text
                    clean_key = key.replace('_', ' ').replace('-', ' ').title()
                    meaningful_attrs.append(f"{clean_key}: {clean_value}")
            
            # Return only the cleaned node name without any attributes to avoid incident IDs
            return clean_node
        else:
            # Just return cleaned node name
            return node.replace('_', ' ').replace('-', ' ').title()
    
    def _format_relationship_description(self, source: str, target: str, edge_data: Dict) -> str:
        """Format relationship information for narrative as simple text"""
        if edge_data:
            relationship = edge_data.get('relationship', 'relates to')
            
            # Clean up the relationship text
            clean_relationship = relationship.replace('_', ' ').replace('-', ' ').lower()
            
            # # Make it more natural language
            # if clean_relationship in ['relates to', 'connected to', 'links to']:
            #     return "connects to"
            # elif clean_relationship in ['causes', 'leads to', 'results in']:
            #     return "causes"
            # elif clean_relationship in ['involves', 'includes', 'contains']:
            #     return "involves"
            # elif clean_relationship in ['affects', 'impacts', 'influences']:
            #     return "affects"
            # else:
            #     return clean_relationship
            return clean_relationship
        else:
            return "connects to"
    
    def _get_community_summaries(self, question: str) -> str:
        """Get community-based summaries using DFS traversal from highest weight edges"""
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
                
                # Get subgraph for this community
                community_subgraph = self.knowledge_graph.subgraph(community)
                
                # Generate DFS-based summary for this community
                community_summary = self._generate_community_dfs_summary(community_subgraph)
                if community_summary:
                    context_parts.append(community_summary)
                else:
                    # Fallback to simple node listing
                    sample_nodes = list(community)[:5]
                    context_parts.append(f"Key entities: {', '.join(sample_nodes)}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error getting community summaries: {e}")
            return "Error analyzing communities"
    
    def _generate_community_dfs_summary(self, subgraph: nx.Graph) -> str:
        """Generate a DFS-based summary of a community, finding the best path with highest-weight edges"""
        try:
            if not subgraph.nodes() or not subgraph.edges():
                # No edges, just list nodes
                if subgraph.nodes():
                    nodes = list(subgraph.nodes())[:5]
                    clean_nodes = [node.replace('_', ' ').replace('-', ' ').title() for node in nodes]
                    return f"Isolated entities: {', '.join(clean_nodes)}"
                return ""
            
            # Strategy: Try multiple starting points and find the path that includes the most high-weight entities
            best_path = []
            best_score = -1
            
            # Get all edges sorted by weight (descending)
            all_edges = [(u, v, data.get('weight', 0)) for u, v, data in subgraph.edges(data=True)]
            all_edges.sort(key=lambda x: x[2], reverse=True)
            
            # Try starting from nodes of the top 3 highest-weight edges
            start_candidates = set()
            for u, v, weight in all_edges[:min(6, len(all_edges))]:  # Top 3 edges = up to 6 nodes
                start_candidates.add(u)
                start_candidates.add(v)
            
            for start_node in list(start_candidates)[:3]:  # Limit to 3 attempts for performance
                path = self._find_best_path_from_node(subgraph, start_node, max_nodes=5)
                if path:
                    # Score the path based on edge weights and diversity
                    score = self._score_path(subgraph, path)
                    if score > best_score:
                        best_score = score
                        best_path = path
            
            # If no good path found, fall back to simple highest-weight edge approach
            if not best_path:
                if all_edges:
                    start_edge = (all_edges[0][0], all_edges[0][1])
                    best_path = self._find_best_path_from_node(subgraph, start_edge[0], max_nodes=5)
                else:
                    # No edges with weights, just pick any path
                    start_node = list(subgraph.nodes())[0]
                    best_path = self._find_best_path_from_node(subgraph, start_node, max_nodes=5)
            
            # Create narrative from the best path
            return self._create_narrative_from_path(subgraph, best_path)
                
        except Exception as e:
            print(f"Error generating community DFS summary: {e}")
            return ""
    
    def _find_best_path_from_node(self, subgraph: nx.Graph, start_node: str, max_nodes: int = 5) -> List[str]:
        """Find the best path starting from a given node using greedy selection of highest-weight edges"""
        if start_node not in subgraph.nodes():
            return []
        
        path = [start_node]
        current_node = start_node
        visited = {start_node}
        
        while len(path) < max_nodes:
            # Get all unvisited neighbors with their edge weights
            neighbors = []
            for neighbor in subgraph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = subgraph.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 0) if edge_data else 0
                    neighbors.append((neighbor, weight))
            
            if not neighbors:
                break
            
            # Choose the neighbor with the highest edge weight
            neighbors.sort(key=lambda x: x[1], reverse=True)
            next_node = neighbors[0][0]
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _score_path(self, subgraph: nx.Graph, path: List[str]) -> float:
        """Score a path based on edge weights and entity diversity"""
        if len(path) < 2:
            return 0
        
        total_weight = 0
        for i in range(len(path) - 1):
            edge_data = subgraph.get_edge_data(path[i], path[i+1])
            weight = edge_data.get('weight', 0) if edge_data else 0
            total_weight += weight
        
        # Bonus for longer paths (more entities)
        length_bonus = len(path) * 0.1
        
        return total_weight + length_bonus
    
    def _create_narrative_from_path(self, subgraph: nx.Graph, path: List[str]) -> str:
        """Create a natural language narrative from a path of nodes"""
        if not path:
            return ""
        
        narrative_parts = []
        
        # Start with the first node
        first_node_attrs = subgraph.nodes.get(path[0], {})
        first_desc = self._format_node_description(path[0], first_node_attrs)
        narrative_parts.append(first_desc)
        
        # Add subsequent nodes with their relationships
        for i in range(1, len(path)):
            current_node = path[i]
            previous_node = path[i-1]
            
            # Get edge information
            edge_data = subgraph.get_edge_data(previous_node, current_node) or subgraph.get_edge_data(current_node, previous_node)
            relationship = self._format_relationship_description(previous_node, current_node, edge_data)
            
            # Format current node
            node_attrs = subgraph.nodes.get(current_node, {})
            node_desc = self._format_node_description(current_node, node_attrs)
            
            narrative_parts.append(f"{relationship} {node_desc}")
        
        # Create natural language narrative
        if len(narrative_parts) == 1:
            return f"Key path: {narrative_parts[0]}"
        else:
            narrative_text = narrative_parts[0]  # Start with first element
            for j in range(1, len(narrative_parts)):
                narrative_text += f", which {narrative_parts[j]}"
            return f"Key path: {narrative_text}"

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
    
    def _create_text_chunks(self, df: pd.DataFrame, chunk_size: int = 500) -> List[str]:
        """Create text chunks from dataframe using only the 'c119' column"""
        chunks = []
        
        # Only use the 'c119' column if it exists
        if 'c119' not in df.columns:
            print(f"Warning: Column 'c119' not found in dataframe. Available columns: {list(df.columns)}")
            return chunks
        
        for _, row in df.iterrows():
            # Only use the 'c119' column value
            if pd.notna(row['c119']) and str(row['c119']).strip():
                text_content = str(row['c119']).strip()
                
                # Split into chunks if text is too long
                if len(text_content) <= chunk_size:
                    chunks.append(text_content)
                else:
                    # Split into smaller chunks
                    words = text_content.split()
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= chunk_size:
                            current_chunk.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_length = len(word)
                    
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        chunk_cache_file = os.path.join(self.cache_dir, "chunk_cache.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.embedding_cache)} cached embeddings")
            
            if os.path.exists(chunk_cache_file):
                with open(chunk_cache_file, 'rb') as f:
                    self.chunk_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.chunk_cache)} cached chunk sets")
                
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            self.embedding_cache = {}
            self.chunk_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
            chunk_cache_file = os.path.join(self.cache_dir, "chunk_cache.pkl")
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            with open(chunk_cache_file, 'wb') as f:
                pickle.dump(self.chunk_cache, f)
                
            print(f"✓ Saved cache with {len(self.embedding_cache)} embeddings and {len(self.chunk_cache)} chunk sets")
            
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_dataset_cache_key(self, datasets: Dict[str, pd.DataFrame], chunk_size: int) -> str:
        """Generate cache key for dataset chunks"""
        # Create a hash based on dataset content and chunk size
        dataset_info = []
        for name, df in datasets.items():
            # Use shape and a few sample values to create a signature
            dataset_signature = f"{name}_{df.shape}_{chunk_size}"
            if len(df) > 0:
                # Add sample of first few rows for content signature
                sample_text = str(df.head(3).to_dict())[:200]
                dataset_signature += f"_{hashlib.md5(sample_text.encode()).hexdigest()[:8]}"
            dataset_info.append(dataset_signature)
        
        combined_signature = "_".join(sorted(dataset_info))
        return hashlib.md5(combined_signature.encode('utf-8')).hexdigest()
    
    def _get_embedding_cached(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache or generate and cache it"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def cleanup(self):
        """Clean up and save cache before shutting down"""
        self._save_cache()
        print("✓ Cache saved and cleanup completed")


if __name__ == "__main__":
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize answer generator
    generator = SensemakingAnswerGenerator(openai_api_key)
    
    # Load knowledge graph if available
    graph_path = "../kg/output/knowledge_graph.gml"
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
