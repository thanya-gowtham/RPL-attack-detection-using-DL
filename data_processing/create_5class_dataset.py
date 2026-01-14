#!/usr/bin/env python3
import pandas as pd
import numpy as np

def create_5class_dataset():
    """Create a perfectly balanced 5-class dataset with all attack types."""
    
    print("Creating 5-class balanced dataset...")
    
    # Load all datasets
    blackhole_df = pd.read_csv('data/raw/blackhole.csv')
    dodag_df = pd.read_csv('data/raw/dodag.csv')
    flooding_df = pd.read_csv('data/raw/flooding.csv')
    rank_df = pd.read_csv('data/raw/rank.csv')
    
    print(f"Loaded datasets:")
    print(f"  Blackhole: {blackhole_df.shape}")
    print(f"  Dodag: {dodag_df.shape}")
    print(f"  Flooding: {flooding_df.shape}")
    print(f"  Rank: {rank_df.shape}")
    
    # Extract samples from each category
    blackhole_attacks = blackhole_df[blackhole_df['category'] == 'Blackhole'].head(1000)
    dodag_attacks = dodag_df[dodag_df['category'] == 'Dodag'].head(1000)
    flooding_attacks = flooding_df[flooding_df['category'] == 'Flooding'].head(1000)
    rank_attacks = rank_df[rank_df['category'] == 'Rank'].head(1000)
    
    # Get Normal samples from all datasets (mix them for variety)
    normal_blackhole = blackhole_df[blackhole_df['category'] == 'Normal'].head(250)
    normal_dodag = dodag_df[dodag_df['category'] == 'Normal'].head(250)
    normal_flooding = flooding_df[flooding_df['category'] == 'Normal'].head(250)
    normal_rank = rank_df[rank_df['category'] == 'Normal'].head(250)
    
    # Combine Normal samples
    normal_samples = pd.concat([normal_blackhole, normal_dodag, normal_flooding, normal_rank], ignore_index=True)
    
    print(f"Extracted samples:")
    print(f"  Blackhole attacks: {len(blackhole_attacks)}")
    print(f"  Dodag attacks: {len(dodag_attacks)}")
    print(f"  Flooding attacks: {len(flooding_attacks)}")
    print(f"  Rank attacks: {len(rank_attacks)}")
    print(f"  Normal samples: {len(normal_samples)}")
    
    # Combine all samples
    all_samples = pd.concat([
        blackhole_attacks, dodag_attacks, flooding_attacks,
        rank_attacks, normal_samples
    ], ignore_index=True)
    
    # Shuffle the data
    all_samples = all_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save balanced dataset
    all_samples.to_csv('data/5class_dataset.csv', index=False)
    
    # Create input version (without category and label columns)
    exclude_cols = ['category', 'label', 'info', 'destination']
    test_input = all_samples.drop(columns=exclude_cols)
    test_input.to_csv('data/5class_test_input.csv', index=False)
    
    print(f"Created 5-class dataset with {len(all_samples)} samples")
    print("Final class distribution:")
    print(all_samples['category'].value_counts())
    print(f"Saved test input to: data/5class_test_input.csv")

if __name__ == "__main__":
    create_5class_dataset()
