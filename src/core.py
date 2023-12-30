
from src.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd


class TimeSeriesCore():
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
    ):
                
        # data
        self.df_train = df_train
        self.df_val = df_val

        # training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        # general advice params
        self.historic_sequence_length = historic_sequence_length
        self.prediction_horizon_length = prediction_horizon_length
        self.target_feature = target_feature

        self.historic_features = historic_features
        self.future_features = future_features

        self.num_historic_features = len(self.historic_features) if self.historic_features else 0
        self.num_future_features = len(self.future_features) if self.future_features else 0

        # make placeholder for model output data
        self.y_pred_col = self.target_feature + '_pred'

        # make train and val dataset
        self.train_dataset = TimeSeriesDataset(
            dataframes=self.df_train,
            target_feature=self.target_feature,
            historic_features=self.historic_features,
            future_features=self.future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length
        )

        self.val_dataset = TimeSeriesDataset(
            dataframes=self.df_val,
            target_feature=self.target_feature,
            historic_features=self.historic_features,
            future_features=self.future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length
        )

        # load train and validation dataset in DataLoader
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.val_loader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

    def __train_model(self):

        # initialize variables
        num_batches = len(self.train_loader)
        total_loss = 0
        
        # iterate through data from dataloader and run model and calculate loss
        self.model.train()
        for X, y in self.train_loader:
            output = self.model(X)
            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        print(f'Train loss: {avg_train_loss}')

        return avg_train_loss

    def __val_model(self):
        
        # initialize variables
        num_batches = len(self.val_loader)
        total_loss = 0

        # iterate through data from dataloader and val model and calculate loss
        self.model.eval()
        with torch.no_grad():
            for X, y in self.val_loader:
                output = self.model(X)
                total_loss += self.loss_function(output, y).item()

        avg_val_loss = total_loss / num_batches
        print(f'val loss: {avg_val_loss}')
        
        return avg_val_loss

    def train(self):

        # run untrained val and make loss lists
        print('Untrained val\n--------')
        self.train_losses = []
        self.val_losses = [self.__val_model()]
        print()

        # train model and calculate training and val loss
        for epoch_i in range(1,self.epochs+1):
            print(f'Epoch {epoch_i}\n-----')

            # epoch
            train_loss = self.__train_model()
            val_loss = self.__val_model()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def set_prediction_data(self, df_pred):
        """
        Description:
        Sets the data that is used to make predictions with the model. This data is used in the predict function.

        input:
        df_pred: dataframe, contains the data that is used to make predictions with the model

        output:
        None
        """

        # make prediction dataset
        self.df_pred = df_pred
            
        self.pred_dataset = TimeSeriesDataset(
            dataframes=df_pred,
            target_feature=self.target_feature,
            historic_features=self.historic_features,
            future_features=self.future_features,
            historic_sequence_length=self.historic_sequence_length,
            prediction_horizon_length=self.prediction_horizon_length,
            predict=True
        )
        
    def predict(self, index):
        """
        Description:
        Makes a prediction with the model for the data at the given index.

        input:
        index: int, index of the data to make a prediction for

        output:
        df_pred: dataframe, contains the prediction data
        """
        
        # get index belonging to prediction data and run model
        index_rows = self.df_pred.index[index+self.historic_sequence_length:index+self.historic_sequence_length+self.prediction_horizon_length]

        self.model.eval()
        with torch.no_grad():
        
            X = self.pred_dataset[index]

            # turn each key into a tensor with batch size by adding the batch dimension
            for key in X:
                X[key] = X[key].unsqueeze(0)             

            y_hat = self.model(X)[0]

            # make dataframe with prediction data
            df_pred = pd.DataFrame(y_hat.numpy(), index=index_rows, columns=[self.y_pred_col])
            return df_pred
