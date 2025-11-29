import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
from src.data.loader import fetch_stock_data, get_earnings_dates
from src.data.features import generate_features
from src.data.preprocessor import DataPreprocessor, prepare_targets
from src.models.lstm_model import LSTMPredictor, QuantileLoss
from src.models.transformer_model import TransformerPredictor
from src.models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_data(self, ticker):
        # 1. Fetch Data
        df = fetch_stock_data(ticker, period=self.config.get('period', '7d'), refresh_data=self.config.get('refresh_data', False))
        if df is None:
            return None, None, None
            
        earnings = get_earnings_dates(ticker)
        
        # 2. Generate Features
        df = generate_features(df, earnings)
        
        # 3. Prepare Targets
        df = prepare_targets(df)
        
        # 4. Split Train/Test (Time-based)
        train_size = int(len(df) * self.config.get('train_split', 0.8))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # 5. Preprocess
        preprocessor = DataPreprocessor(sequence_length=self.config.get('seq_len', 60))
        feature_cols = [c for c in df.columns if c not in ['Datetime', 'Date', 'DateOnly', 'NextDayHigh', 'NextDayLow']]
        target_cols = ['NextDayHigh', 'NextDayLow']
        
        preprocessor.fit(train_df, feature_cols, target_cols)
        train_scaled = preprocessor.transform(train_df)
        test_scaled = preprocessor.transform(test_df)
        
        # 6. Create Sequences
        X_train, y_h_train, y_l_train = preprocessor.create_sequences(train_scaled, 'NextDayHigh', 'NextDayLow')
        X_test, y_h_test, y_l_test = preprocessor.create_sequences(test_scaled, 'NextDayHigh', 'NextDayLow')
        
        return (X_train, y_h_train, y_l_train), (X_test, y_h_test, y_l_test), preprocessor

    def train_lstm(self, train_data, test_data, input_dim):
        X_train, y_h_train, y_l_train = train_data
        X_test, y_h_test, y_l_test = test_data
        
        # Convert to Tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_h_train = torch.FloatTensor(y_h_train).to(self.device)
        y_l_train = torch.FloatTensor(y_l_train).to(self.device)
        
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_h_test = torch.FloatTensor(y_h_test).to(self.device)
        y_l_test = torch.FloatTensor(y_l_test).to(self.device)
        
        model = LSTMPredictor(input_dim, hidden_dim=self.config.get('hidden_dim', 64)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001))
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 32)
        patience = self.config.get('patience', 3)
        
        dataset = TensorDataset(X_train, y_h_train, y_l_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, yh_batch, yl_batch in loader:
                optimizer.zero_grad()
                h_preds, l_preds, _ = model(X_batch)
                
                loss_h = criterion(h_preds, yh_batch)
                loss_l = criterion(l_preds, yl_batch)
                loss = loss_h + loss_l
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss/len(loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                h_preds_val, l_preds_val, _ = model(X_test)
                val_loss_h = criterion(h_preds_val, y_h_test)
                val_loss_l = criterion(l_preds_val, y_l_test)
                val_loss = (val_loss_h + val_loss_l).item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            model.train()
            
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
            
        return model

    def train_transformer(self, train_data, test_data, input_dim):
        X_train, y_h_train, y_l_train = train_data
        X_test, y_h_test, y_l_test = test_data
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_h_train = torch.FloatTensor(y_h_train).to(self.device)
        y_l_train = torch.FloatTensor(y_l_train).to(self.device)
        
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_h_test = torch.FloatTensor(y_h_test).to(self.device)
        y_l_test = torch.FloatTensor(y_l_test).to(self.device)
        
        model = TransformerPredictor(input_dim, d_model=self.config.get('d_model', 64)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001))
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 32)
        patience = self.config.get('patience', 3)
        
        dataset = TensorDataset(X_train, y_h_train, y_l_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, yh_batch, yl_batch in loader:
                optimizer.zero_grad()
                h_preds, l_preds = model(X_batch)
                
                loss_h = criterion(h_preds, yh_batch)
                loss_l = criterion(l_preds, yl_batch)
                loss = loss_h + loss_l
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss/len(loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                h_preds_val, l_preds_val = model(X_test)
                val_loss_h = criterion(h_preds_val, y_h_test)
                val_loss_l = criterion(l_preds_val, y_l_test)
                val_loss = (val_loss_h + val_loss_l).item()
                
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            model.train()
            
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
            
        return model

    def train_xgboost(self, train_data, test_data):
        X_train, y_h_train, y_l_train = train_data
        
        model = XGBoostPredictor(params=self.config.get('xgb_params'))
        model.fit(X_train, y_h_train, y_l_train)
        
        return model

    def run(self, ticker, model_type='lstm'):
        logger.info(f"Starting training for {ticker} with {model_type}")
        
        train_data, test_data, preprocessor = self.prepare_data(ticker)
        if train_data is None:
            logger.error("Failed to prepare data")
            return None
            
        input_dim = train_data[0].shape[2]
        
        if model_type == 'lstm':
            model = self.train_lstm(train_data, test_data, input_dim)
            torch.save(model.state_dict(), os.path.join(self.models_dir, f'{ticker}_lstm.pth'))
        elif model_type == 'transformer':
            model = self.train_transformer(train_data, test_data, input_dim)
            torch.save(model.state_dict(), os.path.join(self.models_dir, f'{ticker}_transformer.pth'))
        elif model_type == 'xgboost':
            model = self.train_xgboost(train_data, test_data)
            model.save(os.path.join(self.models_dir, f'{ticker}_xgb'))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        logger.info("Training completed")
        return model, test_data, preprocessor

if __name__ == "__main__":
    config = {
        'period': '7d',
        'seq_len': 60,
        'epochs': 5,
        'batch_size': 32,
        'hidden_dim': 64
    }
    trainer = Trainer(config)
    trainer.run('RELIANCE.NS', 'lstm')
