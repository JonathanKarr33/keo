# Data Separation for KG Construction vs Sensemaking Questions

## Overview
The aviation maintenance data has been carefully separated to avoid data leakage between knowledge graph construction and sensemaking question generation.

## Data Sources and Usage

### 📊 **Original Datasets**
- `FAA_sample_100.csv` - 100 FAA incident records
- `Maintenance_Text_data.csv` - 2,763 maintenance text records  
- `Aircraft_Annotation_DataFile.csv` - Aircraft problem-action annotations

### 🔧 **For Knowledge Graph Construction**
**Used by GraphRAG for creating aviation maintenance knowledge graph:**

1. **FAA_sample_100.csv** (100 records)
   - Complete FAA incident data
   - Used for entity extraction and relationship mapping

2. **Maintenance Sample Files** (5 × 100 = 500 records)
   - `Maintenance_sample_100_01.csv`
   - `Maintenance_sample_100_02.csv` 
   - `Maintenance_sample_100_03.csv`
   - `Maintenance_sample_100_04.csv`
   - `Maintenance_sample_100_05.csv`
   - Randomly sampled from original maintenance data (seed=42)

**Total for KG: 600 records**

### 💭 **For Sensemaking Question Generation**
**Used by question generator, answer generator, and evaluator:**

1. **Maintenance_remaining_for_questions.csv** (2,263 records)
   - Remaining maintenance data after sampling
   - Ensures no overlap with KG construction data

2. **Aircraft_Annotation_DataFile.csv** (full dataset)
   - Problem-action pair annotations
   - Not used for KG construction

**Total for Questions: 2,263+ records**

## Implementation Details

### Sampling Process
```python
# Random sampling with seed=42 for reproducibility
np.random.seed(42)
df_shuffled = df.sample(frac=1, random_state=42)

# Extract 5×100 samples for KG
samples = df_shuffled.iloc[0:500]  # For KG construction

# Remaining data for questions  
remaining = df_shuffled.iloc[500:]  # For question generation
```

### Data Analyzer Updates
The `AviationDataAnalyzer` class has been updated to:
- Load only `maintenance_remaining` and `aircraft_annotation` datasets
- Exclude all KG construction files from analysis
- Document excluded files in class metadata

### File Structure
```
OMIn_dataset/data/FAA_data/
├── FAA_sample_100.csv                    # → KG Construction
├── Maintenance_Text_data.csv             # → Original (don't use directly)
└── sampled_for_kg/
    ├── Maintenance_sample_100_01.csv     # → KG Construction
    ├── Maintenance_sample_100_02.csv     # → KG Construction  
    ├── Maintenance_sample_100_03.csv     # → KG Construction
    ├── Maintenance_sample_100_04.csv     # → KG Construction
    ├── Maintenance_sample_100_05.csv     # → KG Construction
    ├── Maintenance_remaining_for_questions.csv  # → Question Generation
    └── sampling_metadata.json            # → Metadata
```

## Benefits of This Approach

1. **No Data Leakage**: Strict separation between KG and question data
2. **Reproducibility**: Fixed random seed ensures consistent sampling
3. **Balanced Coverage**: 600 records for KG, 2,263+ for questions
4. **Quality Evaluation**: Can properly assess GraphRAG vs vanilla LLM
5. **Domain Coverage**: Both incident data (FAA) and maintenance data represented

## Usage in Code

### Knowledge Graph Construction
```python
# Use these files for GraphRAG knowledge graph creation:
kg_files = [
    "FAA_sample_100.csv",
    "Maintenance_sample_100_01.csv",
    "Maintenance_sample_100_02.csv", 
    "Maintenance_sample_100_03.csv",
    "Maintenance_sample_100_04.csv",
    "Maintenance_sample_100_05.csv"
]
```

### Sensemaking Question Generation
```python
# Use these files for question/answer generation:
data_paths = {
    'maintenance_remaining': "sampled_for_kg/Maintenance_remaining_for_questions.csv",
    'aircraft_annotation': "Aircraft_Annotation_DataFile.csv"
}
```

## Verification

The sampling process has been verified:
- ✅ Total records: 2,763 (original)
- ✅ Sampled for KG: 500 records  
- ✅ Remaining for questions: 2,263 records
- ✅ Total accounted: 2,763 ✓ Match!

This ensures complete data separation while maintaining comprehensive coverage for both knowledge graph construction and sensemaking question generation.
