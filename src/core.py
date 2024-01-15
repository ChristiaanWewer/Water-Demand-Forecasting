
from src.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import wandb
import os
from src.utils import Scaler
from src import losses

class TimeSeriesCore():
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
    ):
        # model info
        self.model_name = model_name
        self.save_path = save_path
        self.save_model_interval = save_model_interval
        self.log_loss_interval = log_loss_interval

        # data
        self.df_train = df_train
        self.df_val = df_val
        self.scaler_dict = scaler_dict

        # training params
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.masked_loss = masked_loss

        if quantiles is None:
            self.quantiles = [0.5]
        else:
            self.quantiles = quantiles
        
        self.loss_function = getattr(losses, loss_function)(self.quantiles)


        # general advice params
        self.historic_sequence_length = historic_sequence_length
        self.prediction_horizon_length = prediction_horizon_length
        self.target_feature = target_feature
        self.use_historic_target = use_historic_target

        self.historic_features = historic_features
        self.future_features = future_features

        self.num_historic_features = len(self.historic_features) if self.historic_features else 0
        self.num_future_features = len(self.future_features) if self.future_features else 0

        # make placeholder for model output data

        # prediction 0.xth quantile
        self.y_pred_col = ['Prediction {}th quantile {}'.format(q, self.target_feature) for q in self.quantiles]
        self.epoch_i = 0

        # make scaler for training data
        self.scaler = Scaler(
            df=self.df_train,
            columns_to_scale=list(self.scaler_dict.keys()),
            scaling_methods=list(self.scaler_dict.values())
        )

        # scale training data
        self.df_train, self.scaling_coefficients = self.scaler.transform_df(self.df_train)
        self.df_val, _ = self.scaler.transform_df(self.df_val)


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
            y_hat = self.model(X)
            loss = self.loss_function(y=y, y_hat=y_hat, masked_loss=self.masked_loss)
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

                y_hat = self.model(X)

                loss = self.loss_function(y=y, y_hat=y_hat, masked_loss=self.masked_loss).item()
                
                if self.masked_loss and str(loss) == 'nan':
                    num_batches -= 1
                else:
                    total_loss += loss

        if num_batches == 0:
            avg_val_loss = float('nan')
        else:
            avg_val_loss = total_loss / num_batches
        print(f'Val loss: {avg_val_loss}')
        
        return avg_val_loss

    def train(self):
 
        # run untrained val and make loss lists
        self.train_losses = [None]
        self.val_losses = [self.__val_model()]

        # train model and calculate training and val loss
        for epoch_i in range(1,self.epochs+1):
            print(f'Epoch {epoch_i}\n-----')

            # epoch
            train_loss = self.__train_model()
            val_loss = self.__val_model()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print()
            
            if self.log_loss_interval > 0:
                if epoch_i % self.log_loss_interval:
                    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
            
            if self.save_model_interval > 0:
                if epoch_i % self.save_model_interval == 0:
                    self.save_model(epoch_i)

        if self.save_model_interval >= 0:
            self.save_model(epoch_i)
        
        if self.log_loss_interval >= 0:
            loss_table = wandb.Table(data=list(zip(self.train_losses, self.val_losses)), columns=["train_loss", "val_loss"])
            wandb.log({"losses": loss_table})


    def save_model(self, epoch_i):
        """
        Description:
        Saves the model as a wandb artifact.

        input:
        None

        output:
        None
        """

        # make model artifact
        model_artifact = wandb.Artifact(
            name=self.model_name,
            type='model',
            description='Model trained on the given data'
        )

        # save pytorch model
        file_path = r'{}/{}/{}_{}_epochs.pt'.format(self.save_path, self.model_name, self.model_name, epoch_i)
        
        # see if file path self.save_path exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if not os.path.exists(r'{}/{}'.format(self.save_path, self.model_name)):
            os.makedirs(r'{}/{}'.format(self.save_path, self.model_name))
        
        torch.save(self.model.state_dict(), file_path)
        model_artifact.add_file(file_path)
        wandb.run.log_artifact(model_artifact)

    def load_model(self, path):
        """
        Description:
        Loads a pytorch model from a given path

        input:

        """
        print('Loading model...')
        self.model.load_state_dict(torch.load(path))
        print('Model loaded')


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
            predict=True,
            use_historic_target=self.use_historic_target
        )

    # def compute_loss(self, y, y_hat):
    #     """
    #     Description:
    #     Calculates the loss between two tensors (y and y_hat).

    #     Input:
    #     y: torch tensor
    #     y: torch tensor

    #     Output:
    #     loss: torch tensor
    #     """


    #     loss_func =  getattr(losses, self.loss_function)
    #     if self.masked_loss:
    #         mask = ~torch.isnan(y)
    #     else:
    #         mask = None

    #     loss = loss_func(y, y_hat, self.quantiles, mask=mask)    
    #     return loss

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

            y_hat = self.model(X)
            # print(y_hat.shape) # [batch_size, prediction_horizon_length, nr_quantiles]


            # # get rid of batch dimension
            y_hat = y_hat.squeeze(0) # [prediction_horizon_length, nr_quantiles]

            # # turn into df
            df_pred = pd.DataFrame(y_hat.numpy(), index=index_rows, columns=self.y_pred_col)

            s1, s2 = self.scaling_coefficients[self.target_feature]
            df_pred = (df_pred - s2) / s1
            # # rescale data
            # df_pred = self.scaler.get_inverse_transform(df_pred, self.target_feature)

            # # rename df_pred column from self.target_feature to self.y_pred_col
            # df_pred = df_pred.rename(columns={self.target_feature: self.y_pred_col})

            return df_pred
