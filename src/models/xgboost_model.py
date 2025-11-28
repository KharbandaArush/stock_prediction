import xgboost as xgb
import numpy as np
import joblib
import os

class XGBoostPredictor:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        }
        self.model_high = None
        self.model_low = None
        
    def _flatten_input(self, X):
        # X shape: (samples, seq_len, features)
        # Flatten to (samples, seq_len * features)
        nsamples, nx, ny = X.shape
        return X.reshape((nsamples, nx*ny))
        
    def fit(self, X, y_high, y_low):
        """
        Fit separate models for high and low prediction.
        """
        X_flat = self._flatten_input(X)
        
        print("Training XGBoost High Model...")
        self.model_high = xgb.XGBRegressor(**self.params)
        self.model_high.fit(X_flat, y_high)
        
        print("Training XGBoost Low Model...")
        self.model_low = xgb.XGBRegressor(**self.params)
        self.model_low.fit(X_flat, y_low)
        
    def predict(self, X):
        """
        Predict high and low prices.
        Returns: (high_preds, low_preds)
        """
        if self.model_high is None or self.model_low is None:
            raise ValueError("Models not trained yet.")
            
        X_flat = self._flatten_input(X)
        
        high_pred = self.model_high.predict(X_flat)
        low_pred = self.model_low.predict(X_flat)
        
        # XGBoost doesn't natively support multi-output quantile regression in one go easily
        # For simplicity in this wrapper, we return point estimates.
        # To get intervals, we would need quantile loss objective or multiple models.
        # For now, returning point estimates.
        
        return high_pred, low_pred
        
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.model_high, os.path.join(path, 'xgb_high.pkl'))
        joblib.dump(self.model_low, os.path.join(path, 'xgb_low.pkl'))
        
    def load(self, path):
        self.model_high = joblib.load(os.path.join(path, 'xgb_high.pkl'))
        self.model_low = joblib.load(os.path.join(path, 'xgb_low.pkl'))
