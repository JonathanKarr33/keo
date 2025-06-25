# Aviation Maintenance Sensemaking QA System

This directory contains a comprehensive system for generating and evaluating sensemaking questions and answers for aviation maintenance data, following the GraphRAG methodology.

## Overview

The system implements a GraphRAG-inspired approach to create and evaluate sensemaking questions about aviation maintenance failures, following Microsoft Research's methodology for global sensemaking benchmarks.

## Files Description

### Core Components

1. **`data_analyzer.py`** - Analyzes OMIn aviation maintenance datasets to extract patterns and themes
2. **`question_generator.py`** - Generates sensemaking questions using OpenAI API
3. **`answer_generator.py`** - Generates answers using both vanilla LLM and GraphRAG approaches
4. **`evaluator.py`** - Evaluates questions and answers using GraphRAG-style qualitative metrics
5. **`main_pipeline.py`** - Orchestrates the complete workflow

### Question Categories

The system generates questions in these categories based on GraphRAG methodology:

1. **Root Cause Analysis** - Understanding underlying causes of maintenance failures
2. **Predictive Maintenance** - Identifying early warning signs and prevention strategies
3. **Safety Recommendations** - Developing actionable safety improvements
4. **System-Level Understanding** - Holistic view of aviation maintenance ecosystem
5. **Comparative Analysis** - Comparing across different contexts
6. **Trend Analysis** - Temporal and evolving patterns
7. **Global Sensemaking** - Dataset-wide themes and patterns

### Evaluation Metrics

Following GraphRAG's evaluation framework:

- **Comprehensiveness** - Completeness within question context
- **Human Enfranchisement** - Provision of supporting source material
- **Diversity** - Multiple viewpoints and angles
- **Faithfulness** - Factual accuracy and grounding

## Quick Start

### Prerequisites

```bash
pip install pandas numpy networkx openai tqdm matplotlib
export OPENAI_API_KEY="your-openai-api-key"
```

### Basic Usage

1. **Run the complete pipeline:**
```bash
python main_pipeline.py
```

2. **Run with custom settings:**
```bash
python main_pipeline.py --output-dir ./results --sample 20
```

3. **Skip certain steps:**
```bash
python main_pipeline.py --skip-analysis --skip-evaluation
```

### Individual Components

1. **Data Analysis:**
```python
from data_analyzer import AviationDataAnalyzer

data_paths = {
    'faa_sample': "../../OMIn_dataset/data/FAA_data/FAA_sample_100.csv",
    'maintenance_text': "../../OMIn_dataset/data/FAA_data/Maintenance_text_data.csv",
    'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
}

analyzer = AviationDataAnalyzer(data_paths)
analyzer.load_datasets()
analyzer.analyze_failure_patterns()
```

2. **Question Generation:**
```python
from question_generator import SensemakingQuestionGenerator

generator = SensemakingQuestionGenerator(api_key="your-key")
questions = generator.generate_comprehensive_questions(analyzer)
```

3. **Answer Generation:**
```python
from answer_generator import SensemakingAnswerGenerator

answer_gen = SensemakingAnswerGenerator(api_key="your-key")
vanilla_answers = answer_gen.generate_vanilla_answers(questions, datasets)
```

4. **Evaluation:**
```python
from evaluator import SensemakingEvaluator

evaluator = SensemakingEvaluator(api_key="your-key")
comparison_results = evaluator.compare_answer_methods(vanilla_answers, graphrag_answers, questions)
```

## Configuration

Create a custom configuration file:

```json
{
    "data_paths": {
        "faa_sample": "path/to/FAA_sample_100.csv",
        "maintenance_text": "path/to/Maintenance_text_data.csv", 
        "aircraft_annotation": "path/to/Aircraft_Annotation_DataFile.csv"
    },
    "knowledge_graph_path": "path/to/knowledge_graph.gml",
    "output_dir": "./output",
    "model": "gpt-4o",
    "questions_per_category": 8,
    "global_questions": 15,
    "context_questions": 10
}
```

Then run: `python main_pipeline.py --config config.json`

## Output Files

The system generates:

- **Analysis Results:** `analysis_results.json`
- **Questions:** `aviation_sensemaking_questions.json/csv`
- **Vanilla Answers:** `vanilla_answers.json`
- **GraphRAG Answers:** `graphrag_answers.json` (if knowledge graph available)
- **Evaluations:** `evaluation_results.json`
- **Complete Results:** `complete_pipeline_results.json`
- **Summary Report:** `pipeline_report.txt`

## GraphRAG Integration

To use GraphRAG capabilities:

1. Ensure you have a knowledge graph file (`.gml` format)
2. Place it in the specified path in config
3. The system will automatically use it for GraphRAG answer generation

Without a knowledge graph, the system will still work but only generate vanilla LLM answers.

## Example Questions Generated

### Root Cause Analysis
- "What are the most frequent underlying causes of engine failures across different aircraft types?"
- "Which maintenance oversights consistently lead to emergency landings?"

### Global Sensemaking
- "What are the top 5 most critical safety patterns across the entire maintenance dataset?"
- "How do different maintenance philosophies impact overall aviation safety?"

### Predictive Maintenance
- "Based on historical patterns, what early warning signs indicate potential hydraulic system failures?"
- "Which component combinations show the highest risk of cascade failures?"

## Evaluation Results

The system provides comprehensive evaluation including:

- Individual question quality scores
- Answer quality comparisons between vanilla LLM and GraphRAG
- Pairwise comparisons with win rates
- Global sensemaking capability assessment
- Detailed explanations for all evaluations

## Notes

- The system uses OpenAI's API and requires proper API key setup
- Processing time depends on the number of questions generated and API rate limits
- The system implements rate limiting to avoid API quota issues
- All intermediate results are saved for inspection and debugging

## Extending the System

The modular design allows easy extension:

- Add new question categories in `question_generator.py`
- Implement new evaluation metrics in `evaluator.py`
- Add new data analysis methods in `data_analyzer.py`
- Modify the pipeline flow in `main_pipeline.py`
