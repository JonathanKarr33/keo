"""
Sample Maintenance Data for Knowledge Graph Construction
Creates 5 separate CSV files with 100 datapoints each from Maintenance_Text_data.csv
These samples will be used for KG construction and excluded from sensemaking question generation
"""

import pandas as pd
import numpy as np
import os
import json
from typing import List


def sample_maintenance_data(
    input_file: str,
    output_dir: str,
    num_files: int = 5,
    samples_per_file: int = 100,
    random_seed: int = 42
) -> List[str]:
    """
    Sample maintenance data into multiple files for KG construction
    
    Args:
        input_file: Path to Maintenance_Text_data.csv
        output_dir: Directory to save sampled files
        num_files: Number of files to create (default: 5)
        samples_per_file: Number of samples per file (default: 100)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of created file paths
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load the maintenance data
    print(f"Loading maintenance data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from maintenance data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    
    # Check if we have enough data
    total_samples_needed = num_files * samples_per_file
    if len(df) < total_samples_needed:
        print(f"Warning: Only {len(df)} records available, but {total_samples_needed} needed")
        print(f"Adjusting samples per file to {len(df) // num_files}")
        samples_per_file = len(df) // num_files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    created_files = []
    sampled_indices = []
    
    for i in range(num_files):
        # Calculate start and end indices for this sample
        start_idx = i * samples_per_file
        end_idx = start_idx + samples_per_file
        
        # Extract the sample
        sample_df = df_shuffled.iloc[start_idx:end_idx].copy()
        
        # Track sampled indices
        sampled_indices.extend(range(start_idx, end_idx))
        
        # Create filename similar to FAA_sample_100.csv format
        output_file = os.path.join(output_dir, f"Maintenance_sample_{samples_per_file}_{i+1:02d}.csv")
        
        # Save the sample
        try:
            sample_df.to_csv(output_file, index=False)
            created_files.append(output_file)
            print(f"Created {output_file} with {len(sample_df)} records")
            
            # Print some basic stats about this sample
            if 'c8' in sample_df.columns:  # Date column
                date_range = f"{sample_df['c8'].min()} to {sample_df['c8'].max()}"
                print(f"  Sample {i+1} date range: {date_range}")
            
        except Exception as e:
            print(f"Error saving {output_file}: {e}")
    
    # Save the remaining data (not sampled) for question generation
    remaining_df = df_shuffled.iloc[total_samples_needed:].copy()
    remaining_file = os.path.join(output_dir, "Maintenance_remaining_for_questions.csv")
    
    try:
        remaining_df.to_csv(remaining_file, index=False)
        print(f"\nSaved remaining {len(remaining_df)} records to {remaining_file}")
        print("This file should be used for sensemaking question generation")
        created_files.append(remaining_file)
        
    except Exception as e:
        print(f"Error saving remaining data: {e}")
    
    # Create a metadata file with sampling information
    metadata = {
        'total_original_records': len(df),
        'num_sample_files': num_files,
        'samples_per_file': samples_per_file,
        'total_sampled_records': len(sampled_indices),
        'remaining_records_for_questions': len(remaining_df),
        'random_seed': random_seed,
        'sample_files': [os.path.basename(f) for f in created_files[:-1]],  # Exclude remaining file
        'remaining_file': os.path.basename(remaining_file),
        'excluded_from_questions': [
            "FAA_sample_100.csv",  # Used for KG construction
            *[os.path.basename(f) for f in created_files[:-1]]  # Sampled files for KG
        ]
    }
    
    metadata_file = os.path.join(output_dir, "sampling_metadata.json")
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved sampling metadata to {metadata_file}")
        
    except Exception as e:
        print(f"Error saving metadata: {e}")
    
    return created_files


def verify_sampling(sample_files: List[str], original_file: str) -> None:
    """Verify that sampling was done correctly"""
    
    print("\n" + "="*50)
    print("SAMPLING VERIFICATION")
    print("="*50)
    
    # Load original data
    original_df = pd.read_csv(original_file)
    print(f"Original data: {len(original_df)} records")
    
    total_sampled = 0
    for sample_file in sample_files:
        if 'remaining' not in sample_file:  # Skip the remaining file
            sample_df = pd.read_csv(sample_file)
            total_sampled += len(sample_df)
            print(f"{os.path.basename(sample_file)}: {len(sample_df)} records")
    
    # Check remaining file
    remaining_file = next((f for f in sample_files if 'remaining' in f), None)
    if remaining_file:
        remaining_df = pd.read_csv(remaining_file)
        print(f"Remaining for questions: {len(remaining_df)} records")
        
        # Verify totals
        total_accounted = total_sampled + len(remaining_df)
        print(f"\nTotal accounted: {total_accounted}")
        print(f"Original total: {len(original_df)}")
        print(f"Match: {total_accounted == len(original_df)}")


if __name__ == "__main__":
    # Configuration
    input_file = "../../OMIn_dataset/data/FAA_data/Maintenance_Text_data.csv"
    output_dir = "../../OMIn_dataset/data/FAA_data/sampled_for_kg"
    
    # Create samples
    created_files = sample_maintenance_data(
        input_file=input_file,
        output_dir=output_dir,
        num_files=5,
        samples_per_file=100,
        random_seed=42
    )
    
    if created_files:
        # Verify sampling
        verify_sampling(created_files, input_file)
        
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print("1. Use the 5 sample files + FAA_sample_100.csv for KG construction")
        print("2. Use Maintenance_remaining_for_questions.csv + Aircraft_Annotation_DataFile.csv for question generation")
        print("3. Update data_analyzer.py to use the correct files")
        
    else:
        print("Failed to create sample files")
