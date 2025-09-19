# modeling_analysis.py

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_prediction_model(features_df):
    """
    Trains an XGBoost model on the engineered features to predict returns.
    """
    if features_df.shape[0] < 2:
        return None, "Not enough data to train a model."

    # Defining features (X) - all columns except the target and identifiers
    features = [col for col in features_df.columns if '_mean' in col]
    target_1d = 'return_1d'
    
    X = features_df[features]
    y = features_df[target_1d]

    if X.empty or y.empty:
        return None, "Feature or target data is empty."
        
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training a model for 1-day returns
    model_1d = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model_1d.fit(X_train, y_train)
    preds_1d = model_1d.predict(X_test)
    mse_1d = mean_squared_error(y_test, preds_1d)

    results = {
        "model": model_1d,
        "mse": mse_1d,
        "feature_importance": pd.Series(model_1d.feature_importances_, index=features).sort_values(ascending=False),
        "test_predictions": pd.DataFrame({'actual': y_test, 'predicted': preds_1d})
    }

    return results, "XGBoost model trained successfully on 1-Day returns."