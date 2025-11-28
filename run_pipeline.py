import pandas as pd
import os
import logging
import numpy as np
import torch
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.cohort_analysis import assign_cohorts, analyze_cohorts
from src.evaluation.visualizations import generate_visualizations
from src.data.loader import get_market_cap
from src.models.ensemble_model import EnsemblePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    # 1. Configuration
    # 1. Configuration
    # Load tickers from CSV
    tickers_file = os.path.join('data', 'filtered_tickers.csv')
    if os.path.exists(tickers_file):
        logger.info(f"Loading tickers from {tickers_file}...")
        df_tickers = pd.read_csv(tickers_file)
        tickers = df_tickers['Ticker'].tolist()
    else:
        logger.warning(f"Ticker file {tickers_file} not found. Using default list.")
        tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'] 
    
    config = {
        'period': '7d', 
        'seq_len': 30,  
        'epochs': 5,    
        'batch_size': 32,
        'hidden_dim': 64,
        'd_model': 64,
        'xgb_params': {'n_estimators': 50, 'max_depth': 5}
    }
    
    results_dir = 'stock_prediction/results'
    os.makedirs(results_dir, exist_ok=True)
    
    market_cap_map = {}
    
    trainer = Trainer(config)
    
    model_types = ['lstm', 'transformer', 'xgboost']
    
    # Store results for all models
    all_model_results = []
    
    # 2. Loop over tickers
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}...")
        
        # Get Market Cap
        mcap = get_market_cap(ticker)
        if mcap:
            market_cap_map[ticker] = mcap
        
        trained_models = []
        test_data_cache = None
        
            # Train individual models
            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} for {ticker}...")
                    model, test_data, preprocessor = trainer.run(ticker, model_type)
                    
                    if model is None:
                        continue
                    
                    if test_data_cache is None:
                        test_data_cache = test_data
                    
                    trained_models.append(model)
                    
                    # Predict and Evaluate individual model
                    X_test, y_h_test, y_l_test = test_data
                    
                    pred_high, pred_low = predict_with_model(model, X_test, model_type, trainer.device)
                    
                    # Create result dataframe
                    df_res = create_result_df(ticker, y_h_test, y_l_test, pred_high, pred_low, model_type)
                    all_model_results.append(df_res)
                    
                except Exception as e:
                    logger.error(f"Failed {model_type} for {ticker}: {e}")
            
            # Ensemble
            if len(trained_models) > 1:
                try:
                    logger.info(f"Running Ensemble for {ticker}...")
                    ensemble = EnsemblePredictor(trained_models)
                    X_test, y_h_test, y_l_test = test_data_cache
                    
                    # Ensemble prediction logic needs to handle different input types if needed
                    # Our EnsemblePredictor handles it, but we need to pass X_test correctly
                    # XGBoost expects numpy, PyTorch expects tensor (handled inside ensemble or here)
                    # Let's pass numpy X_test, EnsemblePredictor handles conversion
                    
                    pred_high, pred_low = ensemble.predict(X_test)
                    
                    df_res = create_result_df(ticker, y_h_test, y_l_test, pred_high, pred_low, 'ensemble')
                    all_model_results.append(df_res)
                    
                except Exception as e:
                    logger.error(f"Failed Ensemble for {ticker}: {e}")

        except Exception as e:
            logger.error(f"Critical error processing {ticker}: {e}")
            continue

    if not all_model_results:
        logger.error("No results generated.")
        return
        
    # 3. Aggregate and Analyze
    full_results = pd.concat(all_model_results)
    full_results = assign_cohorts(full_results, market_cap_map)
    
    # Analyze per model type
    for m_type in full_results['Model'].unique():
        logger.info(f"\nAnalysis for {m_type}:")
        model_results = full_results[full_results['Model'] == m_type]
        
        mcap_perf, vol_perf = analyze_cohorts(model_results)
        print(f"\n{m_type} - Market Cap Cohort Performance:")
        print(mcap_perf)
        
        # Visualize specific model results
        model_results_dir = os.path.join(results_dir, m_type)
        generate_visualizations(mcap_perf, vol_perf, model_results_dir)

    # Compare Models Overall
    compare_models(full_results, results_dir)
    
    logger.info(f"All results saved to {results_dir}")

def predict_with_model(model, X, model_type, device):
    if model_type in ['lstm', 'transformer']:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            if model_type == 'lstm':
                h, l, _ = model(X_tensor)
            else:
                h, l = model(X_tensor)
            
            # Take median quantile (index 1) if 3 outputs
            if h.shape[1] == 3:
                pred_high = h[:, 1].cpu().numpy()
                pred_low = l[:, 1].cpu().numpy()
            else:
                pred_high = h.cpu().numpy().flatten()
                pred_low = l.cpu().numpy().flatten()
    else:
        # XGBoost
        pred_high, pred_low = model.predict(X)
        
    return pred_high.flatten(), pred_low.flatten()

def create_result_df(ticker, y_h, y_l, pred_h, pred_l, model_name):
    df_res = pd.DataFrame({
        'Ticker': ticker,
        'Model': model_name,
        'ActualHigh': y_h.flatten(),
        'ActualLow': y_l.flatten(),
        'PredHigh': pred_h.flatten(),
        'PredLow': pred_l.flatten(),
        'Price930': y_l.flatten() * 1.01 # Mock
    })
    # Mock Volume with variation
    df_res['Volume_INR'] = np.random.uniform(50000000, 500000000, size=len(df_res))
    return df_res

def compare_models(full_results, save_dir):
    # Aggregate metrics by Model
    metrics = full_results.groupby('Model').apply(lambda x: pd.Series(evaluate_model(x)))
    print("\nOverall Model Comparison:")
    print(metrics)
    metrics.to_csv(os.path.join(save_dir, 'model_comparison.csv'))

if __name__ == "__main__":
    run_pipeline()
