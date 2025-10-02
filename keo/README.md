# KEO - Knowledge Extraction for Operations

**KEO** (Knowledge Extraction for Operations) is a comprehensive framework for extracting, processing, and analyzing knowledge from operations and maintenance data, particularly focused on aviation maintenance incidents. This project provides tools for Named Entity Recognition (NER), Coreference Resolution (CR), Named Entity Linking (NEL), Relation Extraction (RE), and advanced knowledge graph construction using Large Language Models (LLMs).

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key (for LLM-based components)
- CUDA-compatible GPU (recommended for local LLM inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nd-crane/trusted_ke.git
   cd trusted_ke
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Set up OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **For knowledge graph generation (optional):**
   ```bash
   cd keo/kg
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
keo/
├── graph_rag/           # Graph-based Retrieval Augmented Generation
├── kg/                  # Knowledge Graph generation and analysis
├── QA_benchmark/       # Question-Answering evaluation framework
├── sensemaking_QA/     # Advanced QA with sensemaking capabilities
└── vanilla_LLM/        # Baseline LLM implementations
```

## 🔧 Main Components

### 1. Knowledge Graph Generation (`kg/`)
Advanced knowledge graph construction using multiple LLMs and entity normalization:

- **Multi-LLM Support**: Gemma, Phi, GPT-4o, and other models
- **Entity Normalization**: SBERT-based semantic similarity for entity deduplication
- **Batch Processing**: Parallel processing across multiple GPUs
- **Visualization**: Interactive graph visualization and analysis

**Key Scripts:**
- `generate_kg_triplets_with_nodes.py` - Extract triplets using LLMs
- `fix_entity_mentions.py` - Normalize entity mentions
- `generate_fixed_kg.py` - Convert to graph formats (GML, PNG)

**Example Usage:**
```bash
cd keo/kg
# Generate knowledge graph triplets
python scripts/generate_kg_triplets_with_nodes.py --size small --model-shortname gemma3_4b_it --gpu 1

# Normalize entities
python scripts/fix_entity_mentions.py --input output/kg_llm/

# Generate final graph
python scripts/generate_fixed_kg.py --input output/kg_llm/model_fixed.csv
```

### 2. Graph-RAG System (`graph_rag/`)
Retrieval Augmented Generation using knowledge graphs:

- **Hybrid Search**: Combines semantic similarity, path-based relevance, and edge importance
- **Multiple Models**: GPT-4o, GPT-3.5-turbo support
- **SpaCy Integration**: Enhanced NLP capabilities
- **Interactive Querying**: Natural language queries over knowledge graphs

**Example Usage:**
```python
from KEO_GraphRAG import GraphRetriever, load_aviation_graph

# Load knowledge graph
graph = load_aviation_graph('knowledge_graph.gml')

# Initialize retriever
retriever = GraphRetriever(graph, os.getenv("OPENAI_API_KEY"))
retriever.generate_embeddings()

# Query the system
results = retriever.query("What are common aircraft brake issues?", k=3)
answer = retriever.generate_structured_answer("What are common aircraft brake issues?", results)
```

### 3. QA Benchmark Framework (`QA_benchmark/`)
Comprehensive evaluation of question-answering systems:

- **Multiple Approaches**: Graph-RAG, Vanilla LLM, SpaCy-enhanced
- **Automated Evaluation**: Using GPT-4o and Llama models
- **Comparative Analysis**: Performance metrics across different approaches

**Key Features:**
- Question generation from maintenance data
- Automated answer evaluation
- Performance comparison across models

### 4. Sensemaking QA (`sensemaking_QA/`)
Advanced question-answering with sensemaking capabilities:

- **Context-Aware Processing**: Multi-turn conversations
- **Knowledge Integration**: Combines multiple data sources
- **Quantitative Evaluation**: Automated scoring and analysis

### 5. Vanilla LLM (`vanilla_LLM/`)
Baseline implementations for comparison:

- **Multiple Models**: GPT-4, Llama 3.1, Llama 3.2 3B
- **Ollama Integration**: Local model inference
- **Performance Benchmarking**: Comparative analysis tools

## 📊 Dataset Integration

KEO works with the **Operations and Maintenance Intelligence (OMIn) Dataset**, which includes:

- **Aviation Maintenance Data**: FAA accident/incident reports
- **Gold Standards**: Annotated data for NER, CR, and NEL tasks
- **Structured Metadata**: Aircraft details, failure codes, dates
- **Domain-Specific**: Aviation terminology and maintenance procedures

## 🛠️ Advanced Features

### Multi-GPU Processing
```bash
# Run parallel processing on multiple GPUs
nohup python generate_kg_triplets_with_nodes.py --size small --model-shortname gemma3_4b_it --gpu 0 > gpu0.out 2>&1 &
nohup python generate_kg_triplets_with_nodes.py --size small --model-shortname gemma3_4b_it --gpu 1 > gpu1.out 2>&1 &
```

### Entity Normalization
```bash
# Normalize entities with semantic similarity
python fix_entity_mentions.py --input output/kg_llm/ --force
```

### Graph Visualization
```bash
# Generate interactive visualizations
python generate_fixed_kg.py --input path/to/fixed_triplets.csv
```

