from torch import nn
from src.config import load_params
from src.logger import get_logger

logger = get_logger(__name__, log_file='model_architecture.log')
PARAMS = load_params()

class CNN_GRU_Model(nn.Module):
  """A CNN-GRU model for time series classification."""
  def __init__(self, num_features=41):
    super(CNN_GRU_Model, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv1d(in_channels=num_features, out_channels=PARAMS['model']['cnn']['out_channels_1'], kernel_size=PARAMS['model']['cnn']['kernel_size'], padding=PARAMS['model']['cnn']['padding']),
        nn.BatchNorm1d(PARAMS['model']['cnn']['out_channels_1']),
        nn.ReLU(),
        nn.Dropout(PARAMS['model']['cnn']['dropout_1']),

        nn.Conv1d(in_channels=PARAMS['model']['cnn']['out_channels_1'], out_channels=PARAMS['model']['cnn']['out_channels_2'], kernel_size=3, padding=1),
        nn.BatchNorm1d(PARAMS['model']['cnn']['out_channels_2']),
        nn.ReLU(),
        nn.Dropout(PARAMS['model']['cnn']['dropout_2'])
    )
    self.gru = nn.GRU(input_size=PARAMS['model']['cnn']['out_channels_2'], hidden_size=PARAMS['model']['gru']['hidden_size'], batch_first=True, dropout=PARAMS['model']['gru']['dropout'])
    self.fc = nn.Sequential(
        nn.Linear(PARAMS['model']['gru']['hidden_size'], PARAMS['model']['fc']['hidden_size']),
        nn.ReLU(),
        nn.Dropout(PARAMS['model']['fc']['dropout']),
        nn.Linear(PARAMS['model']['fc']['hidden_size'], 1)
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

