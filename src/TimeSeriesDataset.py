import torch
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(
            self, 
            dataframes, 
            target_feature, 
            historic_features=None, 
            future_features=None, 
            historic_sequence_length=5, 
            prediction_horizon_length=1, 
            predict=False
        ): 
        """
        Description:
        Dataset class for time series data. This class can be used to create a dataset from a dataframe with time series data.
        The dataset can be used in a dataloader to load batches of data for training a model.
        
        input:
        dataframes: list of dataframes or one dataframe that contains the columns given in historic_features, future_features and target_feature
        target_feature: string, name of the target feature
        historic_features: list of strings, names of the features that describe the past
        future_features: list of strings, names of the features that describe the future, these can be predictions of other models
        historic_sequence_length: int, length of the sequence of historic data used to make the prediction with
        prediction_horizon_length: int, length of the prediction horizon, how many time steps in the future are predicted with the model
        predict: bool, if True, the target feature is not needed and only the historic and future features are returned

        output:
        None
        """
        
        # save which are the features and target data and sequence length
        self.historic_features = historic_features
        self.future_features = future_features
        self.target_feature = target_feature
        self.historic_sequence_length = historic_sequence_length
        self.prediction_horizon_length = prediction_horizon_length
        self.predict = predict
        self.data_frames = dataframes

        self.lens = [] #len(self.dataframe) - self.historic_sequence_length - self.prediction_horizon_length + 1
        self.X_historic = []
        self.X_future = []
        self.y = []

        if type(dataframes) == pd.core.frame.DataFrame:
            self.__load_from_dataframe(dataframes)

        else:
            for dataframe in dataframes:
                self.__load_from_dataframe(dataframe)


        self.len = sum(self.lens) -1

    def __load_from_dataframe(
            self, 
            dataframe
        ):
        """
        Description:
        Loads data from dataframe into tensors and saves them in the right format to be used in the __getitem__ function.

        input:
        dataframe: dataframe that contains the columns given in historic_features, future_features and target_feature

        output:
        None
        """

        # load data from dataframe into tensors
        if not self.predict:
            self.y.append(torch.tensor(dataframe[self.target_feature].values).float())
        
        if self.historic_features:

            X_historic = torch.tensor(dataframe[self.historic_features].values).float()
            
            # make sure that X_historic is 2d
            if X_historic.dim() == 1:
                X_historic = X_historic.unsqueeze(1)
            
            self.X_historic.append(X_historic)
            
        if self.future_features:

            X_future = torch.tensor(dataframe[self.future_features].values).float()

            # make sure that X_future is 2d
            if X_future.dim() == 1:
                X_future = X_future.unsqueeze(1)

            self.X_future.append(X_future)
            
        self.lens.append(len(dataframe) - self.historic_sequence_length - self.prediction_horizon_length + 1)

    def __len__(
            self
        ):
        """
        Description:
        Returns the length of the dataset, in terms of number of training sampels that can be taken.

        input:
        None

        output:
        length of dataset
        """

        # return length shape
        return self.len

    def __getitem__(
            self, 
            index
        ): 
        """
        Description:
        Returns the data loaded using the dataframes at the given index

        input:
        index: int, index of the data to return

        output:
        x: dict of tensors, contains the historic and future data
        y: tensor, contains the target data (only if predict=False)
        """

        x = {}
        
        # get right dataset number and index and adjust index to dataset number
        for ds_nr, len_ in enumerate(self.lens):
            if index < len_:
                break
            index -= len_
        
        if self.historic_features is not None:
            x['historic'] = self.X_historic[ds_nr][index:index+self.historic_sequence_length,:]
        
        if self.future_features is not None:

            x['future'] = self.X_future[ds_nr][index+self.historic_sequence_length:index+self.historic_sequence_length+self.prediction_horizon_length,:]

        if self.predict:
            return x
        else:
            return x, self.y[ds_nr][index+self.historic_sequence_length:index+self.historic_sequence_length+self.prediction_horizon_length]
