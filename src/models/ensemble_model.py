import numpy as np

class EnsemblePredictor:
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of trained model instances (or wrappers).
            weights: List of weights for each model. If None, equal weights.
        """
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
        
    def predict(self, X):
        """
        Aggregate predictions from all models.
        """
        high_preds_list = []
        low_preds_list = []
        
        for model in self.models:
            # Handle different model output formats
            # Deep learning models return tensors and potentially quantiles
            # XGBoost returns numpy array point estimates
            
            try:
                # Check if it's a PyTorch model
                if hasattr(model, 'eval'):
                    import torch
                    model.eval()
                    with torch.no_grad():
                        # Assume X is already tensor if needed, or convert
                        if not isinstance(X, torch.Tensor):
                            X_tensor = torch.FloatTensor(X)
                        else:
                            X_tensor = X
                            
                        h, l, *rest = model(X_tensor)
                        
                        # If output is quantiles (batch, 3), take median (index 1)
                        if h.shape[1] == 3:
                            h = h[:, 1]
                            l = l[:, 1]
                            
                        high_preds_list.append(h.numpy().flatten())
                        low_preds_list.append(l.numpy().flatten())
                else:
                    # XGBoost or similar
                    h, l = model.predict(X)
                    high_preds_list.append(h.flatten())
                    low_preds_list.append(l.flatten())
                    
            except Exception as e:
                print(f"Error predicting with model {model}: {e}")
                
        # Weighted Average
        final_high = np.zeros_like(high_preds_list[0])
        final_low = np.zeros_like(low_preds_list[0])
        
        for i, w in enumerate(self.weights):
            final_high += high_preds_list[i] * w
            final_low += low_preds_list[i] * w
            
        return final_high, final_low
