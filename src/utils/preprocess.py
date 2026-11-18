#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Data Preprocessing Script for CICIoT2023 Dataset
=============================================================================
Project: ResFed-IDS - Development of an Efficient and Robust Intrusion
         Detection System in Heterogeneous IoT Networks with a
         Resource-Aware Federated Learning Approach

Author: Seidy KANTE
Date: January 2026

Description:
    This script performs comprehensive data preprocessing including:
    - Label taxonomy mapping (34 classes -> 8 categories)
    - Class balancing via undersampling
    - Feature normalization (StandardScaler)
    - Label encoding (LabelEncoder)
    - Train/Test split with stratification
    - Saving processed data and transformers for future use
=============================================================================
"""

import sys
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ciciot2023" / "part-00000-ciciot2023.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Balancing parameters
MAX_SAMPLES_PER_CLASS = 10000
RANDOM_STATE = 42

# Train/Test split ratio
TEST_SIZE = 0.2


# =============================================================================
# LABEL TAXONOMY MAPPING
# =============================================================================

# Mapping from 34 original classes to 8 main categories
LABEL_MAPPING: Dict[str, str] = {
    # DDoS Attacks (Distributed Denial of Service)
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    
    # DoS Attacks (Denial of Service - single source)
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    
    # Mirai Botnet Attacks
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    
    # Reconnaissance Attacks
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',
    
    # Web-based Attacks
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'CommandInjection': 'Web',
    'BrowserHijacking': 'Web',
    'Uploading_Attack': 'Web',
    'Backdoor_Malware': 'Web',
    
    # Brute Force Attacks
    'DictionaryBruteForce': 'BruteForce',
    
    # Spoofing Attacks
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',
    
    # Benign Traffic
    'BenignTraffic': 'Benign',
}

# Category order for consistent display (only keeping main 6 classes)
CATEGORY_ORDER = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing']

# Classes to exclude (too few samples)
EXCLUDED_CLASSES = ['Web', 'BruteForce']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section title."""
    line = char * 70
    print(f"\n{line}")
    print(f" {title}")
    print(f"{line}\n")


def print_distribution(df: pd.DataFrame, label_col: str, title: str) -> None:
    """Print class distribution in a formatted table."""
    print(f"📊 {title}\n")
    
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100
    
    print(f"{'Category':<20} {'Count':>12} {'Percentage':>12}")
    print("-" * 46)
    
    for label in CATEGORY_ORDER:
        if label in counts.index:
            count = counts[label]
            pct = percentages[label]
            print(f"{label:<20} {count:>12,} {pct:>11.2f}%")
    
    print("-" * 46)
    print(f"{'TOTAL':<20} {len(df):>12,} {'100.00':>11}%")
    print(f"\n   • Number of classes: {df[label_col].nunique()}")


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the CICIoT2023 dataset from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
    return df


def map_labels(df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
    """
    Map original 34 classes to 8 main categories.
    
    Args:
        df: Input DataFrame
        label_col: Name of the label column
        
    Returns:
        DataFrame with mapped labels
    """
    print_section("1. LABEL TAXONOMY MAPPING")
    
    # Check for unmapped labels
    original_labels = set(df[label_col].unique())
    mapped_labels = set(LABEL_MAPPING.keys())
    unmapped = original_labels - mapped_labels
    
    if unmapped:
        print(f"⚠️  Warning: {len(unmapped)} unmapped labels found: {unmapped}")
        print("   These will be mapped to 'Unknown'")
        for label in unmapped:
            LABEL_MAPPING[label] = 'Unknown'
    
    # Apply mapping
    df = df.copy()
    df['category'] = df[label_col].map(LABEL_MAPPING)
    
    print(f"✅ Mapped {len(original_labels)} original classes to {df['category'].nunique()} categories")
    print_distribution(df, 'category', "Distribution BEFORE balancing:")
    
    return df


def filter_classes(df: pd.DataFrame,
                   label_col: str = 'category',
                   excluded_classes: list = EXCLUDED_CLASSES) -> pd.DataFrame:
    """
    Filter out minority classes with too few samples.
    
    Args:
        df: Input DataFrame
        label_col: Name of the category column
        excluded_classes: List of classes to exclude
        
    Returns:
        Filtered DataFrame
    """
    print_section("2. CLASS FILTERING")
    
    print(f"📋 Excluding minority classes: {excluded_classes}")
    print(f"   Reason: Insufficient samples for reliable learning\n")
    
    original_count = len(df)
    
    # Filter out excluded classes
    df_filtered = df[~df[label_col].isin(excluded_classes)].copy()
    
    removed_count = original_count - len(df_filtered)
    
    for excluded_class in excluded_classes:
        count = (df[label_col] == excluded_class).sum()
        print(f"   ❌ Removed {excluded_class}: {count:,} samples")
    
    print(f"\n✅ Filtering complete: {original_count:,} → {len(df_filtered):,} samples")
    print(f"   ({removed_count:,} samples removed)")
    
    print_distribution(df_filtered, label_col, "Distribution AFTER filtering:")
    
    return df_filtered


def balance_classes(df: pd.DataFrame, 
                    label_col: str = 'category',
                    max_samples: int = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Balance classes using undersampling for majority classes.
    
    Args:
        df: Input DataFrame
        label_col: Name of the category column
        max_samples: Maximum samples per class
        
    Returns:
        Balanced DataFrame
    """
    print_section("3. CLASS BALANCING (Undersampling)")
    
    print(f"📋 Strategy: Cap each class to max {max_samples:,} samples")
    print(f"   Classes with fewer samples will be kept as-is\n")
    
    balanced_dfs = []
    
    for category in df[label_col].unique():
        category_df = df[df[label_col] == category]
        original_count = len(category_df)
        
        if original_count > max_samples:
            # Undersample majority class
            sampled_df = category_df.sample(n=max_samples, random_state=RANDOM_STATE)
            print(f"   ⬇️  {category}: {original_count:,} → {max_samples:,} (undersampled)")
        else:
            # Keep all samples for minority class
            sampled_df = category_df
            print(f"   ✓  {category}: {original_count:,} (kept all)")
        
        balanced_dfs.append(sampled_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\n✅ Balancing complete: {len(df):,} → {len(balanced_df):,} samples")
    print_distribution(balanced_df, label_col, "Distribution AFTER balancing:")
    
    return balanced_df


def encode_labels(df: pd.DataFrame, label_col: str = 'category') -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode categorical labels to integers.
    
    Args:
        df: Input DataFrame
        label_col: Name of the category column
        
    Returns:
        Tuple of (encoded labels array, fitted LabelEncoder)
    """
    print_section("4. LABEL ENCODING")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col])
    
    print("📋 Label encoding mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        count = (y == idx).sum()
        print(f"   {idx} → {label} ({count:,} samples)")
    
    print(f"\n✅ Encoded {len(label_encoder.classes_)} classes")
    
    return y, label_encoder


def normalize_features(df: pd.DataFrame, 
                       exclude_cols: list = None) -> Tuple[np.ndarray, StandardScaler, list]:
    """
    Normalize numerical features using StandardScaler.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (normalized features array, fitted StandardScaler, feature names)
    """
    print_section("5. FEATURE NORMALIZATION")
    
    if exclude_cols is None:
        exclude_cols = ['label', 'category']
    
    # Select feature columns (all except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"📋 Normalizing {len(feature_cols)} features using StandardScaler")
    print(f"   Excluded columns: {exclude_cols}")
    
    # Extract features
    X = df[feature_cols].values
    
    # Fit and transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Statistics
    print(f"\n📊 Feature statistics after normalization:")
    print(f"   • Mean (should be ~0): {X_scaled.mean():.6f}")
    print(f"   • Std (should be ~1):  {X_scaled.std():.6f}")
    print(f"   • Shape: {X_scaled.shape}")
    
    print(f"\n✅ Normalized {len(feature_cols)} features")
    
    return X_scaled, scaler, feature_cols


def split_data(X: np.ndarray, 
               y: np.ndarray, 
               test_size: float = TEST_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets with stratification.
    
    Args:
        X: Feature array
        y: Label array
        test_size: Proportion of test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print_section("6. TRAIN/TEST SPLIT")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"📋 Split configuration:")
    print(f"   • Test size: {test_size * 100:.0f}%")
    print(f"   • Stratified: Yes (class distribution preserved)")
    print(f"   • Random state: {RANDOM_STATE}")
    
    print(f"\n📊 Split results:")
    print(f"   • Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"   • Test set:     {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    print(f"\n✅ Data split complete")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        scaler: StandardScaler,
                        label_encoder: LabelEncoder,
                        feature_names: list,
                        output_dir: Path) -> None:
    """
    Save all processed data and transformers.
    
    Args:
        X_train, X_test, y_train, y_test: Split data arrays
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature column names
        output_dir: Directory to save files
    """
    print_section("7. SAVING PROCESSED DATA")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    
    print(f"📁 Saved numpy arrays:")
    print(f"   • X_train.npy: {X_train.shape}")
    print(f"   • X_test.npy:  {X_test.shape}")
    print(f"   • y_train.npy: {y_train.shape}")
    print(f"   • y_test.npy:  {y_test.shape}")
    
    # Save transformers and metadata
    metadata = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'label_mapping': LABEL_MAPPING,
        'category_order': CATEGORY_ORDER,
        'n_features': len(feature_names),
        'n_classes': len(label_encoder.classes_),
        'class_names': list(label_encoder.classes_),
        'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
    }
    
    with open(output_dir / "preprocessor.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n📁 Saved preprocessing metadata:")
    print(f"   • preprocessor.pkl (scaler, label_encoder, feature_names, etc.)")
    
    # Save a summary text file
    summary_path = output_dir / "preprocessing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESSING SUMMARY - ResFed-IDS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: CICIoT2023\n")
        f.write(f"Date: {pd.Timestamp.now()}\n\n")
        f.write(f"Training samples: {X_train.shape[0]:,}\n")
        f.write(f"Test samples: {X_test.shape[0]:,}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Classes: {len(label_encoder.classes_)}\n\n")
        f.write("Class mapping:\n")
        for idx, name in enumerate(label_encoder.classes_):
            f.write(f"  {idx}: {name}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"   • preprocessing_summary.txt")
    
    print(f"\n✅ All files saved to: {output_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def preprocess_ciciot2023(data_path: Path = DATA_PATH,
                          output_dir: Path = OUTPUT_DIR,
                          max_samples: int = MAX_SAMPLES_PER_CLASS) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for CICIoT2023 dataset.
    
    Args:
        data_path: Path to input CSV
        output_dir: Directory for output files
        max_samples: Maximum samples per class for balancing
        
    Returns:
        Dictionary containing all processed data and metadata
    """
    print_section("PREPROCESSING PIPELINE - CICIoT2023", "█")
    print("Project: ResFed-IDS")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Input: {data_path}")
    print(f"Output: {output_dir}")
    
    try:
        # Step 1: Load data
        df = load_data(data_path)
        
        # Step 2: Map labels to categories
        df = map_labels(df)
        
        # Step 3: Filter out minority classes
        df_filtered = filter_classes(df)
        
        # Step 4: Balance classes
        df_balanced = balance_classes(df_filtered, max_samples=max_samples)
        
        # Step 5: Encode labels
        y, label_encoder = encode_labels(df_balanced)
        
        # Step 6: Normalize features
        X, scaler, feature_names = normalize_features(df_balanced)
        
        # Step 7: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 8: Save everything
        save_processed_data(
            X_train, X_test, y_train, y_test,
            scaler, label_encoder, feature_names,
            output_dir
        )
        
        # Final summary
        print_section("PREPROCESSING COMPLETE", "█")
        print("📋 Final summary:")
        print(f"   • Original samples:  238,687")
        print(f"   • After balancing:   {len(df_balanced):,}")
        print(f"   • Training samples:  {X_train.shape[0]:,}")
        print(f"   • Test samples:      {X_test.shape[0]:,}")
        print(f"   • Features:          {X_train.shape[1]}")
        print(f"   • Classes:           {len(label_encoder.classes_)}")
        
        print("\n🚀 Ready for model training!")
        print("   Next step: Implement centralized baseline model")
        
        print("\n" + "=" * 70)
        print(" PREPROCESSING COMPLETED SUCCESSFULLY ✅")
        print("=" * 70 + "\n")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_processed_data(processed_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """
    Load previously processed data and transformers.
    
    Args:
        processed_dir: Directory containing processed files
        
    Returns:
        Dictionary with all loaded data and metadata
    """
    print(f"📂 Loading processed data from: {processed_dir}")
    
    # Load numpy arrays
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    # Load metadata
    with open(processed_dir / "preprocessor.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✅ Loaded: X_train{X_train.shape}, X_test{X_test.shape}")
    print(f"   Classes: {metadata['class_names']}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        **metadata
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    preprocess_ciciot2023()
