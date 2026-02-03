"""
Run the entire traffic signal optimization pipeline from data preprocessing to model evaluation
"""

import sys
import argparse
import subprocess
from pathlib import Path


# Utility helpers
def print_header(text):
    print("\n")
    print(text)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# Pipeline runner
def run_complete_pipeline(data_path, sample_recordings=None):
    """
    Run the complete pipeline 

    Args:
        data_path: Path containing inD CSV files
        sample_recordings: Number of recordings to process
    """

    print_header("TRAFFIC SIGNAL OPTIMIZATION PIPELINE")

    print("Pipeline steps:")
    print("1. Data Preprocessing")
    print("2. Feature Extraction")
    print("3. Label Generation")
    print("4. Model Training")
    print("5. Model Evaluation")

    data_path = Path(data_path)

    # Output directories
    processed_path = Path("data/processed")
    features_path = Path("data/features")
    labels_path = Path("data/labels")
    models_path = Path("models/saved_models")
    results_path = Path("results")

    for p in [
        processed_path,
        features_path,
        labels_path,
        models_path,
        results_path,
    ]:
        ensure_dir(p)
        
    # STEP 1: DATA PREPROCESSING
    print_header("DATA PREPROCESSING")

    preprocess_cmd = [
        sys.executable,
        "src/data_preprocessing.py",
        "--data_path",
        str(data_path),
        "--output_path",
        str(processed_path),
    ]

    if sample_recordings is not None:
        preprocess_cmd.extend(["--sample", str(sample_recordings)])

    result = subprocess.run(preprocess_cmd)
    if result.returncode != 0:
        print("Preprocessing failed. Stopping pipeline.")
        return False

    # STEP 2: FEATURE EXTRACTION
    print_header("FEATURE EXTRACTION")

    feature_cmd = [
        sys.executable,
        "src/feature_extraction.py",
        "--processed_path",
        str(processed_path),
        "--output_path",
        str(features_path),
        "--time_window",
        "30",
    ]

    result = subprocess.run(feature_cmd)
    if result.returncode != 0:
        print("Feature extraction failed. Stopping pipeline.")
        return False

    combined_features = features_path / "all_features_combined.csv"
    if not combined_features.exists():
        print("Combined features file not found.")
        return False

    # STEP 3: LABEL GENERATION
    print_header("LABEL GENERATION")

    label_cmd = [
        sys.executable,
        "src/label_generation.py",
        "--features_path",
        str(combined_features),
        "--output_path",
        str(labels_path),
        "--method",
        "adaptive",
    ]

    result = subprocess.run(label_cmd)
    if result.returncode != 0:
        print("Label generation failed. Stopping pipeline.")
        return False

    labeled_file = labels_path / "labeled_data_adaptive.csv"
    if not labeled_file.exists():
        print("Labeled data file not found.")
        return False

    # STEP 4: MODEL TRAINING
    print_header("MODEL TRAINING")

    train_cmd = [
        sys.executable,
        "src/model_training.py",
        "--data_path",
        str(labeled_file),
        "--output_path",
        str(models_path),
        "--models",
        "all",
    ]

    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print("Model training failed. Stopping pipeline.")
        return False

    # STEP 5: MODEL EVALUATION
    print_header("MODEL EVALUATION")

    eval_cmd = [
        sys.executable,
        "src/evaluation.py",
        "--models_path",
        str(models_path),
        "--results_path",
        str(results_path),
        "--visualize",
    ]

    result = subprocess.run(eval_cmd)
    if result.returncode != 0:
        print("Evaluation failed.")
        return False

    # DONE
    print_header("PIPELINE COMPLETE")

    print("Outputs:")
    print(f"Processed data: {processed_path}")
    print(f"Features: {features_path}")
    print(f"Labels: {labels_path}")
    print(f"Models: {models_path}")
    print(f"Results: {results_path}")

    return True

# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Run traffic signal optimization pipeline with inD dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="data/raw",
        help="Path containing inD CSV files",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of recordings to process",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"Data path not found: {data_path}")
        return

    run_complete_pipeline(data_path, args.sample)


if __name__ == "__main__":
    main()
