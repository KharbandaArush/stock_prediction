import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.target_columns = None
        
    def fit(self, df, feature_cols, target_cols=None):
        """
        Fit the scaler on the training data.
        """
        self.feature_columns = feature_cols
        self.scaler.fit(df[feature_cols])
        
        if target_cols:
            self.target_columns = target_cols
            self.target_scaler.fit(df[target_cols])
        
    def transform(self, df):
        """
        Transform the data using the fitted scaler.
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        if self.target_columns:
            # Ensure target columns exist
            valid_targets = [c for c in self.target_columns if c in df.columns]
            if valid_targets:
                df_scaled[valid_targets] = self.target_scaler.transform(df[valid_targets])
                
        return df_scaled

    def inverse_transform_target(self, data):
        """
        Inverse transform target data.
        Args:
            data (np.array): Scaled target data (n_samples, n_targets).
        Returns:
            np.array: Inverse transformed data.
        """
        if self.target_columns is None:
            return data
        return self.target_scaler.inverse_transform(data)

    def inverse_transform(self, data, columns):
        """
        Inverse transform specific columns.
        
        Args:
            data (np.array): Scaled data array (n_samples, n_features).
            columns (list): List of column names corresponding to the data.
            
        Returns:
            np.array: Inverse transformed data.
        """
        # We need to construct a dummy array with the same shape as the scaler expects
        # This is a limitation of sklearn's MinMaxScaler which expects the same number of features
        # A better approach for targets is to have a separate scaler
        pass
        
    def create_sequences(self, df, target_col_high, target_col_low):
        """
        Create sequences for time series models.
        
        Args:
            df (pd.DataFrame): Scaled DataFrame.
            target_col_high (str): Name of the target column for High price.
            target_col_low (str): Name of the target column for Low price.
            
        Returns:
            X (np.array): Input sequences (samples, seq_len, features).
            y_high (np.array): Target high values.
            y_low (np.array): Target low values.
        """
        data = df[self.feature_columns].values
        # For targets, we might want unscaled or scaled. 
        # Usually better to predict scaled and inverse transform.
        # But here we need next day's high/low.
        # Since we are using minute data, "next day" is tricky.
        # We need to aggregate minute data to find the next day's high/low.
        
        # NOTE: This function assumes the df passed is already prepared with 
        # the target values aligned (e.g. 'NextDayHigh', 'NextDayLow' columns added).
        
        if target_col_high not in df.columns or target_col_low not in df.columns:
             raise ValueError(f"Target columns {target_col_high}, {target_col_low} not found.")
             
        targets_high = df[target_col_high].values
        targets_low = df[target_col_low].values
        
        X, y_h, y_l = [], [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y_h.append(targets_high[i + self.sequence_length])
            y_l.append(targets_low[i + self.sequence_length])
            
        return np.array(X), np.array(y_h), np.array(y_l)

def prepare_targets(df, date_col='Datetime'):
    """
    Prepare target variables: Next Day High and Next Day Low.
    This is complex with minute data.
    
    Strategy:
    1. Group by Date.
    2. Find High/Low for each Date.
    3. Shift to get Next Day's High/Low.
    4. Map back to minute data.
    """
    df = df.copy()
    
    # Ensure date column exists
    if date_col not in df.columns:
        # Try index
        if isinstance(df.index, pd.DatetimeIndex):
            df[date_col] = df.index
        else:
             raise ValueError(f"Date column {date_col} not found.")
             
    # Extract Date part
    df['DateOnly'] = df[date_col].dt.date
    
    # Calculate Daily High/Low
    daily_stats = df.groupby('DateOnly').agg({
        'High': 'max',
        'Low': 'min'
    }).rename(columns={'High': 'DailyHigh', 'Low': 'DailyLow'})
    
    # Shift to get Next Day targets
    daily_stats['NextDayHigh'] = daily_stats['DailyHigh'].shift(-1)
    daily_stats['NextDayLow'] = daily_stats['DailyLow'].shift(-1)
    
    # Merge back to minute data
    df = df.merge(daily_stats[['NextDayHigh', 'NextDayLow']], on='DateOnly', how='left')
    
    # Drop rows where NextDay targets are NaN (last day)
    df.dropna(subset=['NextDayHigh', 'NextDayLow'], inplace=True)
    
    return df
