# Aviation Maintenance Sensemaking QA System

This directory contains a comprehensive question-answering system for aviation maintenance sensemaking using Knowledge Graph-enhanced Retrieval-Augmented Generation (GraphRAG). The system generates and evaluates questions and answers for aviation maintenance scenarios using multiple methodologies.

## Overview

The system implements a complete pipeline for:
1. **Data Sampling**: Randomly sampling aviation maintenance data
2. **Question Generation**: Creating sensemaking and actionable questions
3. **Answer Generation**: Using vanilla LLM, text-chunk RAG, and GraphRAG methods
4. **Evaluation**: Multi-faceted evaluation including direct assessment, pairwise comparison, and NLP metrics

## System Architecture

```
Data Sampling → Question Generation → Answer Generation → Evaluation
     ↓                ↓                     ↓               ↓
Aviation Data → Sensemaking QA → Multi-method Answers → Comprehensive Metrics
```

## Quick Start

Run the complete pipeline using the provided shell script:

```bash
chmod +x main.sh
./main.sh
```

## Pipeline Steps

### 1. Data Sampling (Optional)
```bash
python sample_aviation_data.py
```
- Randomly samples 5×100 datapoints from the aviation dataset
- Used for knowledge graph construction
- **Note**: Currently commented out in main.sh

### 2. Question Generation (Optional)
```bash
python generate_questions.py \
    --output-file ./output/aviation_sensemaking_questions.json \
    --question-model gpt-4o
```
- Generates two types of questions:
  - **Sensemaking questions**: From OMIn dataset for broad understanding
  - **Actionable questions**: From MaintNet dataset for specific maintenance actions
- Uses all MaintNet and OMIn datasets plus randomly sampled datapoints
- **Note**: Currently commented out in main.sh

### 3. Answer Generation
```bash
python generate_answers.py \
    --question-files ./output/aviation_sensemaking_questions.json \
    --output-file ./output/answers_gpt-4o-mini_sample20.json \
    --kg-path ../kg/output/knowledge_graph.gml \
    --answer-model gpt-4o-mini \
    --sample-size 20
```

**Parameters:**
- `--question-files`: Input questions JSON file
- `--output-file`: Output answers JSON file
- `--kg-path`: Path to knowledge graph file (.gml format)
- `--answer-model`: LLM model for answer generation
- `--sample-size`: Number of questions to process (20 for demo)

**Answer Generation Methods:**
1. **Vanilla LLM**: Direct LLM response without external context
2. **Text-chunk RAG**: Traditional RAG with text chunking
3. **GraphRAG**: Knowledge graph-enhanced RAG with:
   - Graph context retrieval using weighted maximum spanning trees
   - Community-based summaries
   - Multi-hop entity expansion
   - Real relationship names (cleaned of technical IDs)

### 4. Evaluation
```bash
python run_evaluation.py \
    --questions-file ./output/aviation_sensemaking_questions.json \
    --answers-file ./output/answers_gpt-4o-mini_sample20.json \
    --output-dir ./evaluation_results/answer_gpt-4o-mini_evaluator_gpt-4o \
    --evaluation-model gpt-4o \
    --sample-size 20
```

**Parameters:**
- `--questions-file`: Input questions JSON file
- `--answers-file`: Generated answers JSON file
- `--output-dir`: Directory for evaluation results
- `--evaluation-model`: LLM model for evaluation
- `--sample-size`: Number of answers to evaluate (20 for demo)

**Evaluation Methods:**
1. **Direct Evaluation**: Individual answer quality assessment using GraphRAG-style metrics:
   - Comprehensiveness
   - Human Enfranchisement
   - Diversity
   - Faithfulness

2. **Pairwise Comparison**: Head-to-head comparison between different answer generation methods

3. **NLP Metrics for Actionable Questions**: Ground truth-based evaluation using:
   - BLEU Score
   - METEOR Score
   - ROUGE-L F1
   - Semantic Similarity
   - Exact Match Rate
   - LLM-based evaluation

## Key Features

### GraphRAG Enhancements
- **Weighted Graph Traversal**: Uses maximum spanning trees for context selection
- **Community Summaries**: DFS-based narrative generation from graph communities
- **Clean Output**: Removes technical/incident IDs, uses human-readable relationship names
- **Targeted Text Chunking**: Focuses on specific columns (e.g., 'c119') for relevant content

### Evaluation Capabilities
- **Multi-dimensional Assessment**: Covers quality, accuracy, and utility
- **Method Comparison**: Direct comparison between vanilla, text-chunk, and graph RAG
- **Ground Truth Validation**: For actionable questions with known correct answers
- **Comprehensive Metrics**: Both qualitative (LLM-based) and quantitative (NLP-based) evaluation

## File Structure

```
sensemaking_QA/
├── main.sh                          # Main pipeline script
├── README.md                        # This file
├── sample_aviation_data.py          # Data sampling script
├── generate_questions.py            # Question generation
├── question_generator.py            # Question generation logic
├── generate_answers.py              # Answer generation
├── answer_generator.py              # Answer generation logic
├── run_evaluation.py                # Evaluation runner
├── evaluator.py                     # Evaluation logic
├── data_analyzer.py                 # Data analysis utilities
├── output/                          # Generated files
│   ├── aviation_sensemaking_questions.json
│   └── answers_gpt-4o-mini_sample20.json
└── evaluation_results/              # Evaluation outputs
    └── answer_gpt-4o-mini_evaluator_gpt-4o/
```

## Configuration

### Demo vs Full Experiment
The current configuration uses small sample sizes for demonstration:
- **Question Generation**: All available data
- **Answer Generation**: 20 questions (`--sample-size 20`)
- **Evaluation**: 20 answers (`--sample-size 20`)

For full experiments, remove the `--sample-size` parameters or increase the values.

### Model Selection
- **Question Generation**: GPT-4o (higher quality for question formulation)
- **Answer Generation**: GPT-4o-mini (cost-effective for bulk generation)
- **Evaluation**: GPT-4o (higher quality for assessment)

## Dependencies

Ensure you have the required dependencies:
```bash
pip install openai pandas numpy networkx tqdm
pip install nltk rouge-score # For NLP metrics
```

## Knowledge Graph Requirements

The system requires a knowledge graph file in GML format at:
```
../kg/output/knowledge_graph.gml
```

This graph should contain aviation maintenance entities and relationships.

## Output Files

### Questions File Format
```json
{
  "questions": [
    {
      "id": "question_id",
      "question": "What are the main...",
      "category": "sensemaking|action_specific",
      "type": "global|local",
      "ground_truth_answer": "..." // For action_specific only
    }
  ]
}
```

### Answers File Format
```json
{
  "answers": [
    {
      "question_id": "question_id",
      "method": "vanilla|text_chunk|graph_rag",
      "answer": "The main factors are...",
      "generation_time": 1.23
    }
  ]
}
```

## Evaluation Results

Evaluation generates comprehensive reports including:
- Individual answer assessments
- Method comparison matrices
- Aggregate performance metrics
- Statistical significance tests
- Detailed scoring breakdowns

## Usage Examples

### Custom Question Generation
```bash
python generate_questions.py \
    --output-file ./custom_questions.json \
    --question-model gpt-4o \
    --num-questions 50
```

### Single Method Answer Generation
```bash
python generate_answers.py \
    --question-files ./questions.json \
    --output-file ./vanilla_answers.json \
    --answer-model gpt-4o \
    --methods vanilla
```

### Targeted Evaluation
```bash
python run_evaluation.py \
    --questions-file ./questions.json \
    --answers-file ./answers.json \
    --output-dir ./eval_results \
    --evaluation-model gpt-4o \
    --eval-types direct_evaluation
```

## Performance Notes

- **Runtime**: Full pipeline can take several hours depending on sample size
- **API Costs**: Monitor OpenAI API usage, especially with GPT-4o models
- **Memory**: Knowledge graph processing may require significant RAM for large graphs
- **Rate Limiting**: Built-in delays prevent API rate limit issues

## Troubleshooting

### Common Issues
1. **Missing Knowledge Graph**: Ensure the KG file exists at the specified path
2. **API Key**: Set `OPENAI_API_KEY` environment variable
3. **Memory Issues**: Reduce sample size or use smaller knowledge graphs
4. **Rate Limits**: Increase sleep intervals in the code if needed

### Debug Mode
For troubleshooting, debug prints have been removed from the evaluation code for clean output. To add debugging, modify the evaluator.py file as needed.

## Research Context

This system is designed for aviation maintenance sensemaking research, comparing different RAG approaches for:
- Safety incident analysis
- Maintenance procedure understanding
- Cross-domain knowledge synthesis
- Actionable insight generation

The GraphRAG approach specifically addresses limitations in traditional RAG systems by leveraging structured knowledge representations and community-based context selection.
