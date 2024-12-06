# Standard libraries
import os
import re
import json
import typing
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLP and Embedding Libraries
import spacy
from spacy.tokens import Doc
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Graph Libraries
import networkx as nx
from networkx.algorithms import shortest_paths
import matplotlib.pyplot as plt

import pandas as pd
import re
import openai
from openai import OpenAI
openai_api_key = 'Your_OpenAI_API_Key'

class DataPreparer:
    def __init__(self, file_path):
        """
        Initialize the DataPreparer with the CSV file path.
        """
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None

    def load_data(self):
        """
        Load the data from the CSV file.
        """
        try:
            self.raw_data = pd.read_csv(self.file_path)
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")

    def clean_data(self):
        """
        Perform cleaning operations on the dataset:
        - Drop unnecessary columns.
        - Rename columns for readability.
        - Handle missing values.
        """
        if self.raw_data is None:
            print("No data to clean. Please load data first.")
            return

        self.cleaned_data = self.raw_data.drop(columns=['Unnamed: 0'], errors='ignore')
        self.cleaned_data.rename(
            columns={
                'c119': 'Incident_Description',
                'c77': 'Contributing_Factor',
                'c79': 'Event_Context',
                'c81': 'Role',
                'c146': 'Weight_Category',
                'c148': 'Aircraft_Type',
                'c150': 'Power_Characteristics',
                'c161': 'Outcome',
            },
            inplace=True,
        )
        self.cleaned_data.fillna('Unknown', inplace=True)
        print("Data cleaned successfully!")

    def normalize_text(self):
        """
        Normalize text fields:
        - Convert text to lowercase.
        - Remove special characters.
        """
        if self.cleaned_data is None:
            print("No data to normalize. Please clean data first.")
            return

        def normalize(text):
            text = str(text).lower()  # Ensure text is a string before normalization
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return text.strip()

        for column in self.cleaned_data.columns:
            if self.cleaned_data[column].dtype == "object":
                self.cleaned_data[column] = self.cleaned_data[column].apply(normalize)

        print("Text fields normalized successfully!")

    def get_prepared_data(self):
        """
        Return the cleaned and prepared data as a Pandas DataFrame.
        """
        if self.cleaned_data is None:
            print("Data is not prepared yet. Please clean and normalize the data.")
            return None
        return self.cleaned_data

class DynamicGraphProcessor:
    def __init__(self, data):
        """
        Initialize the graph processor with prepared data.
        """
        self.data = data
        self.graph = nx.DiGraph()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities_and_relationships(self, text):
        """
        Enhanced entity and relationship extraction.
        """
        doc = self.nlp(text)
        relationships = []

        # Improved entity extraction
        entities = []
        for ent in doc.ents:
            if not ent.text.lower() in ['on', 'in', 'of', 'the', 'a', 'an']:  # Filter common words
                entities.append((ent.text, ent.label_))

        # Enhanced relationship extraction
        for token in doc:
            # Expanded dependency patterns
            if token.dep_ in ("nsubj", "dobj", "pobj", "amod", "compound"):
                if not token.is_stop and token.head.pos_ in ['VERB', 'NOUN']:  # More specific conditions
                    subject = token.head.text
                    obj = token.text
                    rel = token.dep_

                    # Get fuller context
                    context = " ".join([t.text for t in token.head.subtree
                                     if not t.is_stop])  # Remove stopwords from context

                    # Add more semantic information
                    relationships.append({
                        "subject": subject,
                        "relation": rel,
                        "object": obj,
                        "context": context,
                        "verb": token.head.lemma_ if token.head.pos_ == 'VERB' else None,
                        "confidence": 1.0 if token.dep_ in ("nsubj", "dobj") else 0.8
                    })

        return relationships, entities

    def build_graph(self):
        """
        Enhanced graph building with better entity handling.
        """
        for _, row in self.data.iterrows():
            incident_description = row.get("Incident_Description", "")
            relationships, entities = self.extract_entities_and_relationships(incident_description)

            # Add entity nodes with more context
            for entity, entity_type in entities:
                if len(entity.split()) <= 3:  # Filter out overly long phrases
                    self.graph.add_node(
                        entity,
                        type=entity_type,
                        category=row.get("Contributing_Factor", "Unknown"),
                        context=row.get("Event_Context", "")
                    )

            # Add relationship edges with enhanced information
            for rel in relationships:
                subject = rel["subject"]
                obj = rel["object"]

                # Skip very short or common words
                if (len(subject) <= 2 or len(obj) <= 2 or
                    subject.lower() in ['on', 'in', 'of'] or
                    obj.lower() in ['on', 'in', 'of']):
                    continue

                # Add nodes if they don't exist
                for node in [subject, obj]:
                    if node not in self.graph:
                        self.graph.add_node(
                            node,
                            type="Entity",
                            category=row.get("Contributing_Factor", "Unknown"),
                            context=row.get("Event_Context", "")
                        )

                # Add edge with rich metadata
                self.graph.add_edge(
                    subject,
                    obj,
                    relationship=rel["relation"],
                    context=rel["context"],
                    confidence=rel.get("confidence", 1.0),
                    incident_type=row.get("Contributing_Factor", "Unknown")
                )

        print(f"Graph built successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges!")

    def detect_communities(self):
        """
        Detect communities in the graph using Label Propagation.
        """
        from networkx.algorithms.community import asyn_lpa_communities

        # Convert to undirected graph for community detection
        undirected_graph = self.graph.to_undirected()
        self.communities = list(asyn_lpa_communities(undirected_graph))

        # Assign community labels to nodes
        community_mapping = {}
        for i, community in enumerate(self.communities):
            for node in community:
                community_mapping[node] = i
        nx.set_node_attributes(self.graph, community_mapping, 'community')

        print(f"Detected {len(self.communities)} communities!")

    def summarize_communities(self, openai_api_key):
        """
        Summarize each community using OpenAI's ChatCompletion API.
        """
        def summarize_community(community_nodes):
            """Summarize a single community."""
            context = "This is a community of entities and their relationships:\n"

            # Add nodes and their types
            for node in community_nodes:
                node_type = self.graph.nodes[node].get('type', 'Unknown')
                context += f"- {node} (Type: {node_type})\n"

                # Add relationships for this node
                for neighbor in self.graph.neighbors(node):
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data:
                        relationship = edge_data.get('relationship', 'related_to')
                        context += f"  → {relationship} → {neighbor}\n"

            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant specializing in analyzing entity relationships in aviation safety incidents."},
                    {"role": "user", "content": f"Analyze and summarize the following entity relationships, focusing on key patterns and insights:\n{context}"}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        self.community_summaries = {}
        for i, community in enumerate(self.communities):
            try:
                summary = summarize_community(community)
                self.community_summaries[i] = summary
                print(f"Community {i} Summary:\n{summary}\n")
            except Exception as e:
                print(f"Error summarizing community {i}: {e}")

    def get_graph_summary(self):
        """
        Provide a detailed summary of the graph structure.
        """
        summary = {
            "Total Nodes": self.graph.number_of_nodes(),
            "Total Edges": self.graph.number_of_edges(),
            "Communities Detected": len(self.communities) if hasattr(self, "communities") else 0,
            "Node Types": self._get_node_type_distribution(),
            "Relationship Types": self._get_relationship_distribution(),
            "Average Degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
        return summary

    def _get_node_type_distribution(self):
        """Helper method to get distribution of node types."""
        type_count = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            type_count[node_type] = type_count.get(node_type, 0) + 1
        return type_count

    def _get_relationship_distribution(self):
        """Helper method to get distribution of relationship types."""
        rel_count = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'Unknown')
            rel_count[rel_type] = rel_count.get(rel_type, 0) + 1
        return rel_count


def visualize_graph_with_communities(graph):
    """
    Visualize the graph with nodes colored by their community.
    """
    # Get community labels
    communities = nx.get_node_attributes(graph, 'community')
    if not communities:
        print("No communities detected. Please run community detection first.")
        return

    # Assign colors to communities
    community_colors = {node: communities[node] for node in graph.nodes()}

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)  # Generate layout for visualization
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=[community_colors.get(node, 0) for node in graph.nodes()],
        cmap=plt.cm.rainbow,
        node_size=500,
        font_size=8,
    )
    plt.title("Graph Visualization with Communities")
    plt.show()


class GraphRetriever:
    def __init__(self, graph, embedding_model="text-embedding-3-small"):
        """
        Initialize the graph retriever.

        Args:
            graph: The graph object containing nodes and edges.
            embedding_model: Model to use for generating embeddings.
        """
        self.graph = graph
        self.embedding_model = embedding_model
        self.embeddings = {}
        self.openai_client = None

    def set_openai_client(self, api_key):
        """
        Set up the OpenAI client.

        Args:
            api_key: OpenAI API key.
        """
        self.openai_client = OpenAI(api_key=api_key)

    def generate_embeddings(self):
        """Generate embeddings with caching."""
        cache_file = "embeddings_cache.json"

        # Load cache if exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.embeddings = json.load(f)

        # Generate missing embeddings
        new_nodes = set(self.graph.nodes()) - set(self.embeddings.keys())
        if new_nodes:
            for node in new_nodes:
                # Generate embedding as before
                node_text = f"{node} ({self.graph.nodes[node].get('type', 'Unknown')})"
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=node_text,
                    encoding_format="float"
                )
                self.embeddings[node] = response.data[0].embedding

            # Save updated cache
            with open(cache_file, 'w') as f:
                json.dump(self.embeddings, f)

    def calculate_similarity(self, query_embedding, node_embedding):
        """
        Calculate cosine similarity between query and node embeddings.
        """
        dot_product = np.dot(query_embedding, node_embedding)
        query_norm = np.linalg.norm(query_embedding)
        node_norm = np.linalg.norm(node_embedding)
        return dot_product / (query_norm * node_norm)

    def retrieve(self, query, k=5):
        """
        Retrieve k most relevant nodes based on the query.

        Args:
            query: The search query.
            k: Number of nodes to retrieve.

        Returns:
            List of (node, similarity_score) tuples.
        """
        if not self.embeddings:
            raise ValueError("No embeddings generated. Call generate_embeddings first.")

        # Generate embedding for the query
        query_response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query,
            encoding_format="float"
        )
        query_embedding = query_response.data[0].embedding

        # Calculate similarities
        similarities = []
        for node, node_embedding in self.embeddings.items():
            similarity = self.calculate_similarity(query_embedding, node_embedding)
            similarities.append((node, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def retrieve_with_context(self, query, k=5, include_neighbors=True):
        """
        Retrieve relevant nodes with their neighborhood context.

        Args:
            query: The search query.
            k: Number of nodes to retrieve.
            include_neighbors: Whether to include neighboring nodes.

        Returns:
            Dictionary with relevant nodes and their context.
        """
        relevant_nodes = self.retrieve(query, k)

        results = {}
        for node, score in relevant_nodes:
            context = {
                'similarity_score': score,
                'node_type': self.graph.nodes[node].get('type', 'Unknown'),
                'neighbors': [],
                'edges': []
            }

            if include_neighbors:
                # Get neighboring nodes
                neighbors = list(self.graph.neighbors(node))
                neighbor_data = []
                for neighbor in neighbors:
                    neighbor_data.append({
                        'node': neighbor,
                        'type': self.graph.nodes[neighbor].get('type', 'Unknown')
                    })
                context['neighbors'] = neighbor_data

                # Get edges with these neighbors
                edges = []
                for neighbor in neighbors:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if edge_data:
                        edges.append({
                            'source': node,
                            'target': neighbor,
                            'attributes': edge_data
                        })
                context['edges'] = edges

            results[node] = context

        return results

    def search(self, query, k=5, threshold=0.5):
        """
        Search the graph with a semantic query.

        Args:
            query: Search query.
            k: Maximum number of results to return.
            threshold: Minimum similarity score threshold.

        Returns:
            List of relevant results with context.
        """
        results = self.retrieve_with_context(query, k)

        # Filter by threshold
        filtered_results = {
            node: data
            for node, data in results.items()
            if data['similarity_score'] >= threshold
        }

        return filtered_results
    def hybrid_search(self, query, k=5, alpha=0.5):
        """
        Combine semantic and structural search.

        Args:
            query: Search query
            k: Number of results
            alpha: Weight between semantic (0) and structural (1) similarity
        """
        semantic_results = self.search(query, k=k)

        # Add PageRank scores for structural importance
        pagerank_scores = nx.pagerank(self.graph)

        # Combine scores
        combined_results = {}
        for node, data in semantic_results.items():
            combined_score = (
                alpha * pagerank_scores[node] +
                (1 - alpha) * data['similarity_score']
            )
            data['combined_score'] = combined_score
            combined_results[node] = data

        return combined_results


# Pipeline integration code
def run_analysis_pipeline(csv_path, openai_api_key, cache_dir="cache"):
    """
    Run the complete analysis pipeline with enhanced features.

    Args:
        csv_path: Path to the FAA data CSV
        openai_api_key: OpenAI API key
        cache_dir: Directory for caching (reserved for future use)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Data Preparation
    print("Step 1: Preparing data...")
    data_preparer = DataPreparer(csv_path)
    data_preparer.load_data()
    data_preparer.clean_data()
    data_preparer.normalize_text()
    prepared_data = data_preparer.get_prepared_data()

    # Step 2: Graph Processing
    print("\nStep 2: Building and processing graph...")
    graph_processor = DynamicGraphProcessor(prepared_data)
    graph_processor.build_graph()

    # Step 3: Community Detection
    print("\nStep 3: Detecting communities...")
    graph_processor.detect_communities()

    # Step 4: Set up retriever
    print("\nStep 4: Setting up retriever and generating embeddings...")
    retriever = GraphRetriever(
        graph_processor.graph,
        embedding_model="text-embedding-3-small"
    )
    retriever.set_openai_client(openai_api_key)
    retriever.generate_embeddings()

    return {
        'data_preparer': data_preparer,
        'graph_processor': graph_processor,
        'retriever': retriever
    }

def query_graph(retriever, query, k=10, threshold=0.3):
    """
    Query the graph and generate a coherent summary answer.
    """
    try:
        print("Executing search...")
        results = retriever.search(query, k=k, threshold=threshold)

        if not results:
            return "No relevant information found in the incident data."

        # Collect and analyze the findings
        findings = {
            'incidents': [],
            'categories': set(),
            'contexts': set(),
            'relationships': []
        }

        for node, data in results.items():
            node_attrs = retriever.graph.nodes[node]

            # Collect incident categories
            if 'category' in node_attrs:
                findings['categories'].add(node_attrs['category'])

            # Collect contexts
            if 'context' in node_attrs:
                findings['contexts'].add(node_attrs['context'])

            # Collect relationships and their contexts
            if data.get('edges'):
                for edge in data['edges']:
                    attrs = edge.get('attributes', {})
                    if 'context' in attrs:
                        findings['relationships'].append({
                            'source': edge['source'],
                            'target': edge['target'],
                            'context': attrs['context']
                        })

        # Generate a coherent summary
        client = OpenAI(api_key=retriever.openai_client.api_key)

        # Prepare the context for GPT
        context = f"""
            Based on the analysis of aviation incident data:

            Categories of incidents: {', '.join(findings['categories'])}
            Incident contexts: {', '.join(findings['contexts'])}

            Key relationships found:
            {chr(10).join([f"- {r['source']} related to {r['target']}: {r['context']}" for r in findings['relationships']])}

            Original query: {query}
            """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an aviation safety analyst. Provide a clear, concise summary of incident data findings."},
                {"role": "user", "content": context}
            ],
            max_tokens=150,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()

        # Print detailed findings for reference
        print("\nDetailed Findings:")
        print("="*50)
        for node, data in results.items():
            print(f"\nNode: {node}")
            print(f"Similarity Score: {data['similarity_score']:.3f}")
            print(f"Node Type: {data['node_type']}")
            if data.get('neighbors'):
                print("\nRelated factors:")
                for neighbor in data['neighbors']:
                    print(f"- {neighbor['node']}")

        print("\nSummary Answer:")
        print("="*50)
        print(summary)

        return summary

    except Exception as e:
        print(f"Error during query: {e}")
        return f"Error processing query: {str(e)}"


# Update the main execution part:
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline_results = run_analysis_pipeline(
        csv_path="../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv",
        openai_api_key=openai_api_key,
        cache_dir="graph_cache"
    )

    print("\nPipeline Statistics:")
    print("-"*30)
    retriever = pipeline_results['retriever']
    graph_processor = pipeline_results['graph_processor']
    print(f"Total nodes in graph: {graph_processor.graph.number_of_nodes()}")
    print(f"Total edges in graph: {graph_processor.graph.number_of_edges()}")
    print(f"Number of embeddings generated: {len(retriever.embeddings)}")

    # Add graph analysis
    print("\nGraph Analysis:")
    print("-"*30)
    print("Most connected nodes:")
    degrees = sorted([(n, d) for n, d in graph_processor.graph.degree()],
                    key=lambda x: x[1], reverse=True)[:5]
    for node, degree in degrees:
        print(f"- {node}: {degree} connections")

    # Run example queries with more variation
    queries = [
        "What is the most common cause of engine failure?",
        "Describe incidents involving engine problems",
        "How do weather conditions affect incidents?",
        "What are the main factors in landing incidents?",
        "What types of pilot errors are reported?",
        "What safety issues are most common?"
    ]

    for query in queries:
        print(f"\nExecuting Query: {query}")
        print("-"*50)
        results = query_graph(
            pipeline_results['retriever'],
            query,
            k=8,  # Increased number of results
            threshold=0.3  # Lowered threshold
        )

        if not results:
            print("No results found for this query.")
        print("\n")

    # # Example usage
    # query = "What is the most common cause of engine failure?"
    # summary = query_graph(retriever, query)