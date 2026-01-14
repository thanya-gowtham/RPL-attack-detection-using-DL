# file: src/models/cnn_gru.py
import torch
import torch.nn as nn

class CNNGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5):
        """
        Initializes the Hybrid CNN-GRU model architecture.
        
        Args:
            input_dim (int): The number of features in the input data.
            hidden_dim (int): The number of hidden units in the GRU layer.
            num_layers (int): The number of layers in the GRU.
            num_classes (int): The number of output classes.
            dropout_prob (float): The dropout probability.
        """
        super(CNNGRUModel, self).__init__()
        
        # --- 1D Convolutional Layers ---
        # These layers act as feature extractors. They scan the input sequences
        # with small filters to identify local patterns (e.g., a sudden spike
        # in transmission rate that lasts for 3 time steps).
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # --- GRU Layer ---
        # The GRU layer processes the sequence of features extracted by the CNNs.
        # It has memory, allowing it to understand the temporal context of the patterns.
        # The input size to the GRU must match the output channels of the last CNN.
        # The Max-Pooling layers have reduced the sequence length.
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, # This makes handling batches more intuitive
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # --- Dropout and Fully Connected Layer ---
        # Dropout is a regularization technique to prevent overfitting by randomly
        # zeroing some of the neurons during training.
        self.dropout = nn.Dropout(dropout_prob)
        
        # The final fully connected (linear) layer maps the GRU's output
        # to the number of classes we want to predict.
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, input_dim).
        
        Returns:
            torch.Tensor: The output logits of shape (batch_size, num_classes).
        """
        # The CNN layers expect input of shape (batch_size, channels, length).
        # Our input is (batch_size, seq_length, input_dim), so we need to permute it.
        # Here, input_dim corresponds to channels and seq_length to length.
        x = x.permute(0, 2, 1) # -> (batch_size, input_dim, seq_length)
        
        # Pass through CNN layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Permute back for the GRU layer, which expects (batch_size, seq_length, features)
        x = x.permute(0, 2, 1) # -> (batch_size, new_seq_length, 128)
        
        # Pass through GRU layer
        # We only need the output of the last time step for classification
        _, h_n = self.gru(x) # h_n shape: (num_layers, batch_size, hidden_dim)
        
        # We take the hidden state of the last layer
        x = h_n[-1, :, :] # -> (batch_size, hidden_dim)
        
        # Apply dropout and the final classification layer
        x = self.dropout(x)
        out = self.fc(x)
        
        return out