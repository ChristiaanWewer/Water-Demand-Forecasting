import torch
import torch.nn as nn
from WewerForecast.core import TimeSeriesCore
from WewerForecast.models.submodels import EncoderLSTM, DecoderLSTM
from torch.nn.utils import weight_norm

class LSTM(nn.Module):
    def __init__(
            self,
            num_historic_features,
            num_future_features,
            historic_sequence_length,
            prediction_horizon_length,
            quantiles,
            lstm_encoder_decoder_hidden_size,
            lstm_encoder_layers,
            lstm_encoder_dropout,
            lstm_decoder_layers,
            lstm_decoder_dropout
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
        self.lstm_encoder_decoder_hidden_size = lstm_encoder_decoder_hidden_size 
        self.num_lstm_encoder_layers = lstm_encoder_layers
        self.lstm_encoder_dropout = lstm_encoder_dropout # probability of dropout

        # decoder lstms hyper params
        self.num_lstm_decoder_layers = lstm_decoder_layers
        self.lstm_decoder_dropout = lstm_decoder_dropout # probability of dropout
        
        # lstms
        self.historic_lstm_encoder = EncoderLSTM(
            input_size=self.num_historic_features+1,
            hidden_size=self.lstm_encoder_decoder_hidden_size,
            batch_first=True,
            num_layers=self.num_lstm_encoder_layers,
            dropout=self.lstm_encoder_dropout
        )

        self.lstm_decoder = DecoderLSTM(
            input_size=self.num_future_features+1,
            hidden_size=self.lstm_encoder_decoder_hidden_size,
            batch_first=True,
            num_layers=self.num_lstm_decoder_layers,
            dropout=self.lstm_decoder_dropout
        )

        self.quantile_layer = nn.Linear(
            in_features=self.lstm_encoder_decoder_hidden_size, 
            out_features=len(self.quantiles)
        )
        
        self.combine_quantiles_layer = weight_norm(
            nn.Linear(
            in_features=len(self.quantiles),
            out_features=1, 
            bias=False
            )
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
        x_historic = x['historic']
        x_future = x['future']

        batch_size = x_historic.shape[0]

        xy_historic = torch.concat((x_historic_target, x_historic), dim=-1)

        hx, cx = self.historic_lstm_encoder(xy_historic)

        xy = xy_historic[:, -1, :].unsqueeze(1)
        target_placeholders = torch.zeros(batch_size, self.prediction_horizon_length, len(self.quantiles))

        for i in range(self.prediction_horizon_length):

            xy, hx, cx = self.lstm_decoder(xy, hx, cx)
            
            # dimension of xy: [batch_size, 1, hidden_size]
            # squeeze to [batch_size, hidden_size], use hidden layer to get [batch_size, nr of quantiles]
            y_quantiles = self.quantile_layer(xy.squeeze(1))

            # add y to target_placeholders
            # compute expected value of the nr of quantiles over that dimension

            target_placeholders[:,i,:] = y_quantiles

            # combine quantiles to get y, make sure the weights of this layer are add up to 1
            y = self.combine_quantiles_layer(y_quantiles)


            # unsqueeze back to [batch_size, 1, nr_target_features], concat with x[future] features to get original size
            xy = torch.cat((y, x_future[:,i,:]), dim=-1).unsqueeze(1)

        return target_placeholders


class TimeSeriesLSTM(TimeSeriesCore):
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
            lstm_encoder_decoder_hidden_size,
            lstm_encoder_layers,
            lstm_encoder_dropout,
            lstm_decoder_layers,
            lstm_decoder_dropout
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
        self.lstm_encoder_decoder_hidden_size = lstm_encoder_decoder_hidden_size
        self.lstm_encoder_layers = lstm_encoder_layers
        self.lstm_encoder_dropout = lstm_encoder_dropout
        self.lstm_decoder_layers = lstm_decoder_layers
        self.lstm_decoder_dropout = lstm_decoder_dropout

        # initialize model 
        self.model = LSTM(
            num_historic_features=self.num_historic_features,
            num_future_features=self.num_future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length,
            quantiles=self.quantiles,
            lstm_encoder_decoder_hidden_size=self.lstm_encoder_decoder_hidden_size,
            lstm_encoder_layers=self.lstm_encoder_layers,
            lstm_encoder_dropout=self.lstm_encoder_dropout,
            lstm_decoder_layers=self.lstm_decoder_layers,
            lstm_decoder_dropout=self.lstm_decoder_dropout

        )

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)