#!/usr/bin/env python3
"""
Entry point for running the training pipeline.
Used by GitHub Actions workflow.
"""

import os
import sys
from datetime import datetime, timedelta

# Ensure pipelines directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from training_pipeline import run_training_pipeline


def main():
    """Run the training pipeline with sensible defaults."""
    print(f"Training Pipeline Runner - {datetime.now().isoformat()}")
    print("-" * 50)

    # By default, use last 20% of data for testing
    # In production, you might want to set a specific date
    test_start_date = os.environ.get("TEST_START_DATE", None)
    upload_model = os.environ.get("UPLOAD_MODEL", "true").lower() == "true"

    if test_start_date:
        print(f"Using test_start_date: {test_start_date}")
    else:
        print("Using 80/20 time-based split")

    print(f"Upload model: {upload_model}")
    print("-" * 50)

    try:
        model, metrics = run_training_pipeline(
            test_start_date=test_start_date,
            upload_model=upload_model
        )
        print("\nPipeline completed successfully!")
        print(f"Final metrics: {metrics}")
        return 0
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
