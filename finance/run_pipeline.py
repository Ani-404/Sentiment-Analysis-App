"""
End-to-end finance pipeline.

Transcripts -> sentence sentiment -> aggregated features -> post-earnings
returns -> XGBoost return prediction.

Run from the project root:
    python -m finance.run_pipeline
"""
import os

import pandas as pd

from finance.financial_analysis import run_prediction_model
from finance.processor import (
    aggregate_sentiment_features,
    get_classifier,
    get_stock_returns,
    ingest_transcripts,
    preprocess_and_split,
    score_sentences,
)


def build_features(project_root: str) -> pd.DataFrame:
    """Turn raw transcripts into a feature+target DataFrame."""
    transcripts = ingest_transcripts(project_root)
    classifier = get_classifier()

    rows = []
    for _, row in transcripts.iterrows():
        sentences = preprocess_and_split(row["transcript"])
        scores = score_sentences(sentences, classifier)
        features = aggregate_sentiment_features(scores)

        returns = get_stock_returns(row["ticker"], str(row["earnings_date"]))
        if not returns:
            print(f"  ! Skipping {row['ticker']}: no return data available.")
            continue

        rows.append(
            {
                "company_name": row["company_name"],
                "ticker": row["ticker"],
                "earnings_date": row["earnings_date"],
                **features,
                **returns,
            }
        )

    return pd.DataFrame(rows)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Building features from transcripts...")
    features_df = build_features(project_root)

    if features_df.empty:
        print("No usable rows produced (likely no stock data). Exiting.")
        return

    print("\nEngineered features:")
    print(features_df.to_string(index=False))

    print("\nTraining prediction model...")
    results, message = run_prediction_model(features_df)
    print(message)

    if results:
        print(f"\nTest MSE (1-day return): {results['mse']:.6f}")
        print("\nFeature importance:")
        print(results["feature_importance"].to_string())
        print("\nPredictions vs actual:")
        print(results["test_predictions"].to_string(index=False))


if __name__ == "__main__":
    main()
