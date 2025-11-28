import optuna
import logging
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

class Tuner:
    def __init__(self, ticker, n_trials=10):
        self.ticker = ticker
        self.n_trials = n_trials
        
    def objective_lstm(self, trial):
        config = {
            'period': '7d',
            'seq_len': trial.suggest_int('seq_len', 30, 120),
            'epochs': 5, # Keep low for tuning speed
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 128),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5)
        }
        
        trainer = Trainer(config)
        # We need a way to get validation loss from trainer
        # For now, we'll just run it and catch exceptions
        try:
            model, test_data, _ = trainer.run(self.ticker, 'lstm')
            # Calculate validation loss here or return a metric
            # Placeholder: return 0.0
            return 0.0 
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')

    def tune(self, model_type='lstm'):
        study = optuna.create_study(direction='minimize')
        
        if model_type == 'lstm':
            study.optimize(self.objective_lstm, n_trials=self.n_trials)
        
        logger.info(f"Best params: {study.best_params}")
        return study.best_params

if __name__ == "__main__":
    tuner = Tuner('RELIANCE.NS', n_trials=2)
    tuner.tune('lstm')
