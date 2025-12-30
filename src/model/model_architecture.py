from torch.nn import nn


from src.logger import get_logger
logger = get_logger(__name__, log_file='model_architecture.log')

class CNN_GRU_Model(nn.Module):
  """A CNN-GRU model for time series classification."""
  def __init__(self, num_features=41):
    super(CNN_GRU_Model, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    self.gru = nn.GRU(input_size=16, hidden_size=32, batch_first=True, dropout=0.3)
    self.fc = nn.Sequential(
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(16,1)
    )
    logger.info("Initialized CNN_GRU_Model architecture.")

  def forward(self, x):
    try:
        
          #our x: (batch, seq_len, num_features)
          # but cnn expects (batch, channels, seq_len) so we swap 1 & 2 index
        x = x.permute(0,2,1)
        x = self.cnn(x)
          # Back to (batch, seq_len, channels) for GRU
        x = x.permute(0,2,1)
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        output = self.fc(h_n)

        return output
    except Exception as e:
        logger.exception(f"Error in forward pass: {e}")
        raise

