import torch
import torch.nn as nn
from WewerForecast.core import TimeSeriesCore
from WewerForecast.models.submodels import EncoderLSTM

class LSTM(nn.Module):
    def __init__(
            self,
            num_historic_features,
            num_future_features,
            historic_sequence_length,
            prediction_horizon_length,
            quantiles,
            lstm_historic_encoder_hidden_size,
            num_historic_encoder_layers,
            lstm_historic_encoder_dropout,
            lstm_future_encoder_hidden_size,
            num_future_encoder_layers,
            lstm_future_encoder_dropout,
            FNN_hidden_size,
            FNN_layers,
            FNN_dropout
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

        output:
        None
        """

        # save needed info to initialize layers

        # historic and future data related params
        self.num_historic_features = num_historic_features # input size of lstm too
        self.historic_sequence_length = historic_sequence_length # length of historic features
        self.num_future_features = num_future_features 
        self.prediction_horizon_length = prediction_horizon_length
        self.quantiles = quantiles

        # encoder lstms hyper params
        self.lstm_historic_encoder_hidden_size = lstm_historic_encoder_hidden_size 
        self.num_historic_encoder_layers = num_historic_encoder_layers
        self.lstm_historic_encoder_dropout = lstm_historic_encoder_dropout # probability of dropout

        # encoder lstms hyper params
        self.lstm_future_encoder_hidden_size = lstm_future_encoder_hidden_size 
        self.num_future_encoder_layers = num_future_encoder_layers
        self.lstm_future_encoder_dropout = lstm_future_encoder_dropout # probability of dropout

        # FNN decoder hyper params
        self.FNN_hidden_size = FNN_hidden_size
        self.FNN_layers = FNN_layers
        self.FNN_dropout = FNN_dropout

        # lstms
        self.historic_lstm_encoder = EncoderLSTM(
            input_size=1,
            hidden_size=self.lstm_historic_encoder_hidden_size,
            batch_first=True,
            num_layers=self.num_historic_encoder_layers,
            dropout=self.lstm_historic_encoder_dropout
        )

        # self.future_lstm_encoder = EncoderLSTM(
        #     input_size=self.num_future_features,
        #     hidden_size=self.lstm_future_encoder_hidden_size,
        #     batch_first=True,
        #     num_layers=self.num_future_encoder_layers,
        #     dropout=self.lstm_future_encoder_dropout
        # )

        self.future_lstm_encoder = nn.LSTM(
            input_size=self.num_future_features,
            hidden_size=self.lstm_future_encoder_hidden_size,
            batch_first=True,
            num_layers=self.num_future_encoder_layers,
            dropout=self.lstm_future_encoder_dropout
        )

        self.combine_future_and_historic_hx = nn.Linear(
            in_features=self.lstm_historic_encoder_hidden_size * self.num_historic_encoder_layers + self.prediction_horizon_length,
            out_features=self.FNN_hidden_size
        )

        self.FNN_decoder = nn.Sequential(
            *[nn.Linear(self.FNN_hidden_size, self.FNN_hidden_size), nn.ReLU(), nn.Dropout(self.FNN_dropout)] * self.FNN_layers,
        )

        self.out_layer = nn.Linear(
            in_features=self.FNN_hidden_size, 
            out_features=self.prediction_horizon_length
        )
        

        
    def forward(self, x):
        """
        Description:
        Forward pass of the model. This function runs the model on the input data and returns the prediction.

        input:
        x: dict of tensors, contains the historic and future data

        output:
        x: tensor, contains the prediction
        """

        x_historic_target = x['historic_target']
        x_future = x['future']

        batch_size = x_historic_target.shape[0]

        historic_hx, _ = self.historic_lstm_encoder(x_historic_target)

        hf = torch.zeros(self.num_future_encoder_layers, batch_size, self.lstm_future_encoder_hidden_size)
        cf = torch.zeros(self.num_future_encoder_layers, batch_size, self.lstm_future_encoder_hidden_size)

        future_x, _ = self.future_lstm_encoder(x_future, (hf, cf))
        future_x = future_x[:, :, -1] # shape [batch size, sequence_length]

        
        # first reshape hx to (batch size, num layers, hidden size)
        historic_hx = historic_hx.permute(1, 0, 2)
        # then reshape hx to (batch size, num layers x hidden size)
        historic_hx = historic_hx.reshape(batch_size, -1)

        # combine historic and future hx
        combined_hx = torch.cat((historic_hx, future_x), dim=1)

        # pass combined hx through linear layer
        combined_hx = self.combine_future_and_historic_hx(combined_hx)
        
        # pass combined hx through FNN decoder
        combined_hx = self.FNN_decoder(combined_hx)

        # make empty placeholder for quantiles
        quantiles = torch.empty((batch_size, self.prediction_horizon_length, len(self.quantiles)))

        # pass combined hx through out layer for each quantile
        for i in range(len(self.quantiles)):
         
            out = self.out_layer(combined_hx) 
            quantiles[:, :, i] = out

        return quantiles


class TimeSeriesLSTMMLP(TimeSeriesCore):
    def __init__(
            self,  
            df_train,                                      
            df_val,                                        
            model_name,
            log_loss_interval,
            save_path,
            save_model_interval,
            target_feature,
            use_historic_target,   
            historic_features,                            
            future_features,  
            scaler_dict,                                                   
            historic_sequence_length,                      
            prediction_horizon_length,                     
            batch_size,                                
            epochs,                                                
            learning_rate,                              
            loss_function,
            quantiles,
            masked_loss,
            lstm_historic_encoder_hidden_size,
            num_historic_encoder_layers,
            lstm_historic_encoder_dropout,
            lstm_future_encoder_hidden_size,
            num_future_encoder_layers,
            lstm_future_encoder_dropout,
            FNN_hidden_size,
            FNN_layers,
            FNN_dropout 
    ):

        super().__init__(
            df_train,                                      
            df_val,
            model_name,
            log_loss_interval,
            save_path,
            save_model_interval,            
            target_feature,
            use_historic_target,                               
            historic_features,                            
            future_features,  
            scaler_dict,                                                   
            historic_sequence_length,                      
            prediction_horizon_length,                     
            batch_size,                                
            epochs,                                                
            learning_rate,                              
            loss_function, 
            quantiles,
            masked_loss
        )

        # LSTM model (hyper) params
        # historic and future data related params
        self.historic_sequence_length = historic_sequence_length # length of historic features
        self.prediction_horizon_length = prediction_horizon_length
        self.quantiles = quantiles

        # encoder lstms hyper params
        self.lstm_historic_encoder_hidden_size = lstm_historic_encoder_hidden_size 
        self.num_historic_encoder_layers = num_historic_encoder_layers
        self.lstm_historic_encoder_dropout = lstm_historic_encoder_dropout # probability of dropout

        # encoder lstms hyper params
        self.lstm_future_encoder_hidden_size = lstm_future_encoder_hidden_size 
        self.num_future_encoder_layers = num_future_encoder_layers
        self.lstm_future_encoder_dropout = lstm_future_encoder_dropout # probability of dropout

        # FNN decoder hyper params
        self.FNN_hidden_size = FNN_hidden_size
        self.FNN_layers = FNN_layers
        self.FNN_dropout = FNN_dropout

        # initialize model 
        self.model = LSTM(
            num_historic_features=self.num_historic_features,
            num_future_features=self.num_future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length,
            quantiles=self.quantiles,
            lstm_historic_encoder_hidden_size=self.lstm_historic_encoder_hidden_size,
            num_historic_encoder_layers=self.num_historic_encoder_layers,
            lstm_historic_encoder_dropout=self.lstm_historic_encoder_dropout,
            lstm_future_encoder_hidden_size=self.lstm_future_encoder_hidden_size,
            num_future_encoder_layers=self.num_future_encoder_layers,
            lstm_future_encoder_dropout=self.lstm_future_encoder_dropout,
            FNN_hidden_size=self.FNN_hidden_size,
            FNN_layers=self.FNN_layers,
            FNN_dropout=self.FNN_dropout
        )

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)