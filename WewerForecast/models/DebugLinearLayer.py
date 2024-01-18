import torch
import torch.nn as nn
from WewerForecast.core import TimeSeriesCore
from WewerForecast.models.submodels import EncoderLSTM

class MLP(nn.Module):
    def __init__(
            self,
            num_historic_features,
            num_future_features,
            historic_sequence_length,
            prediction_horizon_length,
            quantiles,

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

        # justone linear layer for testing
        self.linear = nn.Linear(in_features=self.prediction_horizon_length, out_features=self.prediction_horizon_length)
        
    def forward(self, x):
        """
        Description:
        Forward pass of the model. This function runs the model on the input data and returns the prediction.

        input:
        x: dict of tensors, contains the historic and future data

        output:
        x: tensor, contains the prediction
        """

        x_future = x['future']
        # get rid of last dimension
        x_future = x_future.squeeze(dim=-1)
        quantiles = torch.zeros((x_future.shape[0], x_future.shape[1], len(self.quantiles)))
        for i in range(len(self.quantiles)):
            quantiles[:, :, i] = self.linear(x_future)

        return quantiles


class TimeSeriesDebugLinearLayer(TimeSeriesCore):
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
            masked_loss
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

        # initialize model 
        self.model = MLP(
            num_historic_features=self.num_historic_features,
            num_future_features=self.num_future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length,
            quantiles=self.quantiles
        )

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)