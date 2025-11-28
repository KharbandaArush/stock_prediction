import pandas as pd
import numpy as np

def calculate_execution_rate(y_true_high, y_true_low, y_pred_high, y_pred_low):
    """
    Calculate execution rate for High and Low predictions.
    
    Args:
        y_true_high: Actual High prices.
        y_true_low: Actual Low prices.
        y_pred_high: Predicted High prices (Sell Limit).
        y_pred_low: Predicted Low prices (Buy Limit).
        
    Returns:
        dict: Execution rates.
    """
    # High Prediction (Sell): Executed if Actual High >= Predicted High
    executed_high = y_true_high >= y_pred_high
    rate_high = np.mean(executed_high)
    
    # Low Prediction (Buy): Executed if Actual Low <= Predicted Low
    executed_low = y_true_low <= y_pred_low
    rate_low = np.mean(executed_low)
    
    return {
        'execution_rate_high': rate_high,
        'execution_rate_low': rate_low,
        'executed_mask_high': executed_high,
        'executed_mask_low': executed_low
    }

def calculate_alpha(y_true_930, y_pred_high, y_pred_low, executed_mask_high, executed_mask_low):
    """
    Calculate Alpha vs 9:30 AM baseline.
    
    Args:
        y_true_930: Price at 9:30 AM.
        y_pred_high: Predicted High (Sell).
        y_pred_low: Predicted Low (Buy).
        executed_mask_high: Boolean mask of executed sell orders.
        executed_mask_low: Boolean mask of executed buy orders.
        
    Returns:
        dict: Alpha metrics.
    """
    # Alpha for Sell (Exit):
    # Baseline: Sell at 9:30 AM.
    # Strategy: Sell at Pred High (if executed).
    # If executed: Gain = Pred High - 9:30 Price.
    # If not executed: Gain = 0 (or assume sold at Close/9:30? User implies comparison on executed).
    # Let's calculate Average Alpha per Attempt.
    
    # For executed trades:
    alpha_high_executed = (y_pred_high[executed_mask_high] - y_true_930[executed_mask_high]) / y_true_930[executed_mask_high]
    avg_alpha_high = np.mean(alpha_high_executed) if len(alpha_high_executed) > 0 else 0
    
    # Alpha for Buy (Entry):
    # Baseline: Buy at 9:30 AM.
    # Strategy: Buy at Pred Low (if executed).
    # If executed: Gain = 9:30 Price - Pred Low (Buying cheaper is better).
    
    alpha_low_executed = (y_true_930[executed_mask_low] - y_pred_low[executed_mask_low]) / y_true_930[executed_mask_low]
    avg_alpha_low = np.mean(alpha_low_executed) if len(alpha_low_executed) > 0 else 0
    
    return {
        'avg_alpha_high': avg_alpha_high,
        'avg_alpha_low': avg_alpha_low
    }

def evaluate_model(df_results):
    """
    Comprehensive evaluation wrapper.
    Expects df_results to have columns: 
    ['ActualHigh', 'ActualLow', 'PredHigh', 'PredLow', 'Price930']
    """
    exec_metrics = calculate_execution_rate(
        df_results['ActualHigh'], 
        df_results['ActualLow'], 
        df_results['PredHigh'], 
        df_results['PredLow']
    )
    
    alpha_metrics = calculate_alpha(
        df_results['Price930'],
        df_results['PredHigh'],
        df_results['PredLow'],
        exec_metrics['executed_mask_high'],
        exec_metrics['executed_mask_low']
    )
    
    return {**exec_metrics, **alpha_metrics}
