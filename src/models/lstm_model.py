import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, hidden_dim)
        return context, weights

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        # Bidirectional doubles the hidden dimension
        self.attention = Attention(hidden_dim * 2)
        
        # Output heads for High and Low predictions
        # Predicting 3 quantiles for each: 0.1, 0.5 (median), 0.9
        self.fc_high = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) 
        )
        
        self.fc_low = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        context, attn_weights = self.attention(lstm_out)  # (batch_size, hidden_dim * 2)
        
        high_preds = self.fc_high(context)
        low_preds = self.fc_low(context)
        
        return high_preds, low_preds, attn_weights

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss
