#!/usr/bin/env python3
"""
Generate Sensemaking Questions for Aviation Maintenance Data
Uses only the data designated for question generation (excludes KG construction files)
"""

import os
import json
import argparse
from datetime import datetime
from data_analyzer import AviationDataAnalyzer
from question_generator import SensemakingQuestionGenerator

def generate_aviation_sensemaking_questions(output_file, question_model="gpt-4o"):
    """Generate comprehensive sensemaking questions for aviation maintenance"""
    
    print("=" * 70)
    print("AVIATION MAINTENANCE SENSEMAKING QUESTION GENERATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verify OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    print("✓ OpenAI API key verified")
    
    # Data paths - using only question generation datasets (excludes KG files)
    data_paths = {
        'maintenance_remaining': "../../OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
        'aircraft_annotation': "../../OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv"
    }
    
    print("\n" + "=" * 70)
    print("STEP 1: DATA ANALYSIS")
    print("=" * 70)
    print("Loading and analyzing data for question generation...")
    print("EXCLUDED: FAA_sample_100.csv and 5 maintenance sample files (used for KG)")
    print("INCLUDED: Maintenance_remaining_for_questions.csv and Aircraft_Annotation_DataFile.csv")
    
    # Initialize and run data analysis
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    
    print("\nRunning comprehensive data analysis...")
    
    # Run all analysis methods
    failure_patterns = analyzer.analyze_failure_patterns()
    component_patterns = analyzer.analyze_components()
    text_patterns = analyzer.analyze_text_patterns()
    temporal_patterns = analyzer.analyze_temporal_patterns()
    aircraft_patterns = analyzer.analyze_aircraft_types()
    themes = analyzer.identify_sensemaking_themes()
    
    print("✓ Data analysis completed")
    print(f"  - Failure patterns: {len(failure_patterns)} categories")
    print(f"  - Component patterns: {len(component_patterns)} categories")
    print(f"  - Text patterns: {len(text_patterns)} categories")
    print(f"  - Sensemaking themes: {len(themes)} identified")
    
    print("\n" + "=" * 70)
    print("STEP 2: QUESTION GENERATION")
    print("=" * 70)
    
    # Initialize question generator
    generator = SensemakingQuestionGenerator(openai_api_key, model=question_model)
    
    all_questions = []
    
    # 1. Generate comprehensive category-based questions
    print("Generating category-based sensemaking questions...")
    comprehensive_questions = generator.generate_comprehensive_questions(
        analyzer, 
        questions_per_category=8,
        use_data_context=True
    )
    all_questions.extend(comprehensive_questions)
    print(f"✓ Generated {len(comprehensive_questions)} category-based questions")
    
    # 2. Generate global sensemaking questions
    print("\nGenerating global sensemaking questions...")
    global_questions = generator.generate_global_sensemaking_questions(
        analyzer, 
        num_questions=15
    )
    all_questions.extend(global_questions)
    print(f"✓ Generated {len(global_questions)} global questions")
    
    # 3. Generate context-specific questions for maintenance data
    print("\nGenerating context-specific questions for maintenance narratives...")
    if 'maintenance_remaining' in analyzer.datasets:
        maint_sample = analyzer.datasets['maintenance_remaining'].sample(n=min(100, len(analyzer.datasets['maintenance_remaining'])))
        context_questions_maint = generator.generate_context_specific_questions(
            maint_sample,
            'maintenance_narratives',
            num_questions=10
        )
        all_questions.extend(context_questions_maint)
        print(f"✓ Generated {len(context_questions_maint)} maintenance context questions")
    
    # 4. Generate action-specific questions (NEW: practical "what to do when..." questions)
    print("\nGenerating action-specific questions with ground truth answers...")
    action_questions = generator.generate_action_specific_questions(
        analyzer,
        num_questions=50
    )
    all_questions.extend(action_questions)
    print(f"✓ Generated {len(action_questions)} action-specific questions with ground truth answers")
    
    # 5. Generate context-specific questions for aircraft annotation data
    print("\nGenerating context-specific questions for aircraft annotations...")
    if 'aircraft_annotation' in analyzer.datasets:
        annotation_sample = analyzer.datasets['aircraft_annotation'].sample(n=min(100, len(analyzer.datasets['aircraft_annotation'])))
        context_questions_aircraft = generator.generate_context_specific_questions(
            annotation_sample,
            'aircraft_problems_actions',
            num_questions=10
        )
        all_questions.extend(context_questions_aircraft)
        print(f"✓ Generated {len(context_questions_aircraft)} aircraft annotation context questions")
    
    # Store all questions in generator
    generator.generated_questions = all_questions
    
    print("\n" + "=" * 70)
    print("STEP 3: RESULTS SUMMARY")
    print("=" * 70)
    
    # Generate summary
    summary = generator.get_questions_summary()
    print(f"Total questions generated: {summary['total_questions']}")
    print("\nQuestions by category:")
    for category, count in summary['categories'].items():
        print(f"  - {category}: {count} questions")
    
    print(f"\nQuestion types: {', '.join(summary['question_types'])}")
    
    print("\n" + "=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save questions in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON (using provided output file path)
    generator.save_questions(output_file, format='json')
    
    # Save as CSV (same path and name but with .csv extension)
    base_name = os.path.splitext(output_file)[0]
    csv_file = f"{base_name}.csv"
    generator.save_questions(csv_file, format='csv')
    
    # Save analysis results (with timestamp)
    analysis_file = f"{base_name}_analysis_{timestamp}.json"
    analyzer.save_analysis_results(analysis_file)
    
    # Save generation metadata (with timestamp)
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'data_sources': {
            'maintenance_remaining': len(analyzer.datasets.get('maintenance_remaining', [])),
            'aircraft_annotation': len(analyzer.datasets.get('aircraft_annotation', []))
        },
        'excluded_files': [
            'FAA_sample_100.csv',
            'Maintenance_sample_100_01.csv',
            'Maintenance_sample_100_02.csv', 
            'Maintenance_sample_100_03.csv',
            'Maintenance_sample_100_04.csv',
            'Maintenance_sample_100_05.csv'
        ],
        'question_summary': summary,
        'openai_model': question_model,
        'data_separation_enforced': True
    }
    
    metadata_file = f"{base_name}_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Questions saved to: {output_file}")
    print(f"✓ Questions saved to: {csv_file}")
    print(f"✓ Analysis results saved to: {analysis_file}")
    print(f"✓ Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("QUESTION GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Generated {summary['total_questions']} sensemaking questions")
    print("Data separation enforced - KG construction files excluded")
    print("Ready for answer generation and evaluation!")
    print("=" * 70)
    
    return all_questions, summary

def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate sensemaking questions for aviation maintenance data"
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output JSON file for generated questions. Required.'
    )
    parser.add_argument(
        '--question-model',
        default="gpt-4o",
        help='OpenAI model to use for question generation (default: gpt-4o)'
    )
    
    args = parser.parse_args()
    
    try:
        questions, summary = generate_aviation_sensemaking_questions(
            output_file=args.output_file,
            question_model=args.question_model
        )
        print(f"\n🎉 SUCCESS: Generated {summary['total_questions']} sensemaking questions!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
