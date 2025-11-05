#!/usr/bin/env python3
"""
Script to split dataset based on specified column values.
Updates the 'split' column in place in the CSV file.

Usage:
    python split_dataset.py --csv-file seed1-30.csv --split-column seed --train-values 1-18 --val-values 19-24 --test-values 25-30
"""

import argparse
import pandas as pd
import sys
from typing import List, Union


def parse_range(range_str: str) -> List[int]:
    """Parse range string like '1-18' or '1,2,3' into list of integers."""
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in range_str.split(',')]


def update_split_column(csv_file: str, split_column: str, 
                       train_values: List[int], val_values: List[int], 
                       test_values: List[int]) -> None:
    """Update the split column in the CSV file based on specified values."""
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if split column exists, create if not
    if 'split' not in df.columns:
        df['split'] = ''
    
    # Check if the split_column exists
    if split_column not in df.columns:
        print(f"Error: Column '{split_column}' not found in CSV file.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Create mapping for split values
    split_mapping = {}
    
    # Add train values
    for val in train_values:
        split_mapping[val] = 'train'
    
    # Add validation values
    for val in val_values:
        split_mapping[val] = 'val'
    
    # Add test values
    for val in test_values:
        split_mapping[val] = 'test'
    
    # Update the split column based on the split_column values
    df['split'] = df[split_column].map(split_mapping).fillna('')
    
    # Count splits for verification
    split_counts = df['split'].value_counts()
    print(f"Split distribution:")
    for split_type in ['train', 'val', 'test']:
        count = split_counts.get(split_type, 0)
        print(f"  {split_type}: {count} rows")
    
    # Save the updated CSV
    try:
        df.to_csv(csv_file, index=False)
        print(f"Successfully updated '{csv_file}' with split assignments.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset based on specified column values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_dataset.py --csv-file seed1-30.csv --split-column seed --train-values 1-18 --val-values 19-24 --test-values 25-30
  python split_dataset.py --csv-file data.csv --split-column N --train-values 15 --val-values 16 --test-values 15,16
        """
    )
    
    parser.add_argument('--csv-file', required=True, help='Path to the CSV file to update')
    parser.add_argument('--split-column', required=True, help='Column name to use for splitting')
    parser.add_argument('--train-values', required=True, help='Values for train set (e.g., "1-18" or "1,2,3")')
    parser.add_argument('--val-values', required=True, help='Values for validation set (e.g., "19-24" or "4,5,6")')
    parser.add_argument('--test-values', required=True, help='Values for test set (e.g., "25-30" or "7,8,9")')
    
    args = parser.parse_args()
    
    # Parse the value ranges
    try:
        train_values = parse_range(args.train_values)
        val_values = parse_range(args.val_values)
        test_values = parse_range(args.test_values)
    except ValueError as e:
        print(f"Error parsing value ranges: {e}")
        sys.exit(1)
    
    # Check for overlapping values
    all_train = set(train_values)
    all_val = set(val_values)
    all_test = set(test_values)
    
    if all_train & all_val:
        print("Warning: Overlapping values between train and validation sets")
    if all_train & all_test:
        print("Warning: Overlapping values between train and test sets")
    if all_val & all_test:
        print("Warning: Overlapping values between validation and test sets")
    
    # Update the split column
    update_split_column(args.csv_file, args.split_column, train_values, val_values, test_values)


if __name__ == "__main__":
    main()
