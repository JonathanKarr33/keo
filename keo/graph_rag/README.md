# KEO Graph-RAG

## Overview
This module implements Knowledge Extraction for Operations (KEO) using Graph-based Retrieval Augmented Generation (Graph-RAG). It combines aviation maintenance knowledge graphs with OpenAI's language models to provide intelligent querying and analysis capabilities.

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Environment Setup

1. **Install Dependencies**
   ```bash
   pip install networkx matplotlib openai numpy
   ```

2. **Set OpenAI API Key**
   
   Set your OpenAI API key as an environment variable:
   
   **Linux/macOS:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```
   
   **Windows:**
   ```bash
   set OPENAI_API_KEY=your-openai-api-key-here
   ```
   
   **Or add to your shell profile (.bashrc, .zshrc, etc.):**
   ```bash
   echo 'export OPENAI_API_KEY="your-openai-api-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Usage

1. **Basic Graph Loading and Visualization:**
   ```python
   from KEO_GraphRAG import load_aviation_graph, visualize_graph
   
   # Load the graph
   graph = load_aviation_graph('knowledge_graph.gml')
   
   # Visualize it
   if graph:
       visualize_graph(graph)
   ```

2. **Graph-RAG Querying:**
   ```python
   from KEO_GraphRAG import GraphRetriever
   
   # Initialize retriever
   retriever = GraphRetriever(graph, os.getenv("OPENAI_API_KEY"))
   
   # Generate embeddings (run once)
   retriever.generate_embeddings()
   
   # Query the system
   results = retriever.query("What are common issues with aircraft brakes?", k=3)
   
   # Get structured answer
   answer = retriever.generate_structured_answer("What are common issues with aircraft brakes?", results)
   print(answer)
   ```

### Files

- `KEO_GraphRAG.py` - Main Python implementation
- `KEO_GraphRAG.ipynb` - Jupyter notebook version
- `KEO_GraphRAG_spacy.py` - SpaCy-enhanced version with additional NLP capabilities
- `knowledge_graph.gml` - Aviation maintenance knowledge graph data

### Configuration

The system uses the following default configurations:
- **Model**: GPT-4o for answer generation
- **Embedding Model**: text-embedding-3-small
- **Hybrid Search Weights**: 
  - Semantic similarity: 0.4
  - Path-based relevance: 0.4
  - Edge type importance: 0.2

### Troubleshooting

**Error: "OPENAI_API_KEY environment variable not set"**
- Make sure you've set the environment variable correctly
- Restart your terminal/IDE after setting the variable
- Verify the variable is set: `echo $OPENAI_API_KEY` (Linux/macOS) or `echo %OPENAI_API_KEY%` (Windows)

**Graph loading errors:**
- Ensure the `knowledge_graph.gml` file exists in the correct directory
- Check file permissions and path

**Import errors:**
- Install missing dependencies: `pip install networkx matplotlib openai numpy`
- For spaCy version: `pip install spacy sentence-transformers transformers`