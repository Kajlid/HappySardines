#!/usr/bin/env python3
"""
Entry point for running the inference pipeline.
Used by GitHub Actions workflow.
"""

import os
import sys
from datetime import datetime

# Ensure pipelines directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from inference_pipeline import run_inference_pipeline


def main():
    """Run the inference pipeline with sensible defaults."""
    print(f"Inference Pipeline Runner - {datetime.now().isoformat()}")
    print("-" * 50)

    # Configuration from environment
    hours_ahead = int(os.environ.get("HOURS_AHEAD", "48"))
    upload_predictions = os.environ.get("UPLOAD_PREDICTIONS", "true").lower() == "true"

    print(f"Hours ahead: {hours_ahead}")
    print(f"Upload predictions: {upload_predictions}")
    print("-" * 50)

    try:
        predictions_df = run_inference_pipeline(
            hours_ahead=hours_ahead,
            upload_predictions=upload_predictions,
        )
        print("\nPipeline completed successfully!")
        print(f"Total predictions generated: {len(predictions_df)}")
        return 0
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
