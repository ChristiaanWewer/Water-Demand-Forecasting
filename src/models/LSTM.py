import torch
import torch.nn as nn
from src.core import TimeSeriesCore

class LSTM(nn.Module):
    def __init__(
            self,
            num_historic_features,
            num_future_features,
            historic_sequence_length,
            prediction_horizon_length,
            lstm_hidden_size,
            lstm_layers,
            lstm_dropout,
            decoder_size,
            decoder_dropout,
            activation_function_decoder
        ):
        super().__init__()
        """
        Description:
        LSTM model for time series forecasting. This model can be used to make predictions based on historic and future data.
        The model consists of an LSTM encoder for the historic data and a decoder that combines the LSTM encoded historic data with the future data.

        input:
        num_historic_features: int, number of features in the historic data
        num_future_features: int, number of features in the future data
        historic_sequence_length: int, length of the sequence of historic data used to make the prediction with
        prediction_horizon_length: int, length of the prediction horizon, how many time steps in the future are predicted with the model
        lstm_hidden_size: int, size of the hidden state of the LSTM encoder
        lstm_layers: int, number of LSTM layers in the LSTM encoder
        lstm_dropout: float, probability of dropout in the LSTM encoder
        decoder_size: list of ints, [breadth, depth], breadth is the size of the linear layers in the MLP decoder and depth is the number of layers
        decoder_dropout: float, probability of dropout in the MLP decoder
        activation_function_decoder: torch.nn activation function, activation function used in the MLP decoder

        output:
        None
        """

        # save needed info to initialize layers

        # historic and future data related params
        self.num_historic_features = num_historic_features # input size of lstm too
        self.historic_sequence_length = historic_sequence_length # length of historic features
        self.num_future_features = num_future_features 
        self.prediction_horizon_length = prediction_horizon_length

        # lstm hyper params
        self.lstm_hidden_size = lstm_hidden_size 
        self.num_lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout # probability of dropout

        # decoder hyper params
        self.decoder_breadth, self.decoder_depth = decoder_size # dimensions of decoder
        self.activation_function_decoder = activation_function_decoder # nn.ReLU() or nn.LeakyReLU() or whatever
        self.decoder_dropout = decoder_dropout # probability of dropout

        # first LSTM layer of model
        self.lstm = nn.LSTM(
            input_size=self.num_historic_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
            num_layers=self.num_lstm_layers,
            dropout=self.lstm_dropout
        )
        
        # combine historic and future data
        self.combine_historic_and_future = nn.Linear(
            in_features=self.historic_sequence_length * self.lstm_hidden_size + self.prediction_horizon_length * self.num_future_features, 
            out_features=self.decoder_breadth
        )

        # decoder as sequence of dropout(relu(linear)) layers

        decoder_layer = nn.Sequential(
            nn.Dropout(p=self.decoder_dropout),
            self.activation_function_decoder,
            nn.Linear(in_features=self.decoder_breadth, out_features=self.decoder_breadth)
        )
        self.mlp_decoder = nn.Sequential(
            *[decoder_layer for _ in range(self.decoder_depth)]
        )

        # linear layer
        self.out_layer = nn.Linear(in_features=self.decoder_breadth, out_features=self.prediction_horizon_length)

    def forward(self, x):
        """
        Description:
        Forward pass of the model. This function runs the model on the input data and returns the prediction.

        input:
        x: dict of tensors, contains the historic and future data

        output:
        x: tensor, contains the prediction
        """
        
        x_historic = x['historic']
        x_future = x['future']
        batch_size = x_historic.shape[0]
        # shape x: (batch_size, historic_sequence_length, in_features)
        # turn x's into shape (batch_size, historic_sequence_length x in_features)
        
        # initialize c0 and h0
        c0 = torch.zeros(self.num_lstm_layers, batch_size, 
                         self.lstm_hidden_size).requires_grad_()
        h0 = torch.zeros(self.num_lstm_layers, batch_size, 
                         self.lstm_hidden_size).requires_grad_()
        
        # 1 LSTM encoder on historic data
        rnn_out, (_, _) = self.lstm(x_historic, (h0, c0))

        # rnn_out shape: [batch_size, historic_sequence_length, hidden_size]
        # turn rnn_out to [batch_size, historic_sequence_length x hidden_size]
        rnn_out = rnn_out.reshape(batch_size, -1)

        # turn x_future size to [batch_size, prediction_horizon_length x future features]
        x_future = x_future.reshape(batch_size, -1) 

        # concatenate rnn_out and x_future [batch_size, (historic_sequence_length x hidden_size) + (prediction_horizon_length x future features)]
        x = torch.cat((rnn_out, x_future), dim=1)

        # combine the combination of LSTM encoded historic data and future data with a linear layer
        x = self.combine_historic_and_future(x)

        # run through a MLP decoder which is defined above
        x = self.mlp_decoder(x)

        # reshape this to the output shape [batch_size, prediction_horizon_length, 1] by the means of a linear layer
        x = self.out_layer(x)

        return x


class TimeSeriesLSTM(TimeSeriesCore):
    def __init__(
            self,  
            df_train,                                      
            df_val,                                        
            target_feature,                               
            historic_features,                            
            future_features,                             
            historic_sequence_length,                      
            prediction_horizon_length,                     
            batch_size,                                
            epochs,                                                
            learning_rate,                              
            loss_function, 
            lstm_hidden_size,                                   
            lstm_layers,
            lstm_dropout,
            decoder_size,
            decoder_dropout,
            activation_function_decoder
        ):
        super().__init__(
            df_train,                                      
            df_val,                                        
            target_feature,                               
            historic_features,                            
            future_features,                             
            historic_sequence_length,                      
            prediction_horizon_length,                     
            batch_size,                                
            epochs,                                                
            learning_rate,                              
            loss_function, 
        )

        # LSTM model (hyper) params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        # decoder params
        self.decoder_size = decoder_size # [breadth, depth]
        self.decoder_dropout = decoder_dropout # probability of dropout
        self.activation_function_decoder = activation_function_decoder # nn.ReLU() or nn.LeakyReLU() or whatever

        # initialize model 
        self.model = LSTM(
            num_historic_features=self.num_historic_features,
            num_future_features=self.num_future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout,
            decoder_size=self.decoder_size,
            decoder_dropout=self.decoder_dropout,
            activation_function_decoder=self.activation_function_decoder
        )

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)