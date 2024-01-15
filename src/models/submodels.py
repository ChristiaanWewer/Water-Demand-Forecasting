import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            batch_first,
            num_layers,
            dropout,
    ):  
        super(EncoderLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def forward(self, x): 
        
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_()
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_()
        _, (h0, c0) = self.lstm(x, (h0, c0))

        return h0, c0
    
class DecoderLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            batch_first,
            num_layers,
            dropout
    ):  
        super(DecoderLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # self.out_layer = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x, hx, cx): 
        
        x, (hx, cx) = self.lstm(x, (hx, cx))

        return x, hx, cx