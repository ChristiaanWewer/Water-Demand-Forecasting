import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class Scaler:
    def __init__(self, df, columns_to_scale, scaling_methods): 
        self.df = df
        self.columns_to_scale = columns_to_scale
        self.scaling_methods = scaling_methods

        if type(df) == pd.core.frame.DataFrame:
            self.scaling_coefficients = self.compute_scaling_coefficients(self.df)

        elif type(df) == list:
            total_lens = len(df[0])
            scaling_coeffs = self.compute_scaling_coefficients(df[0]) * len(df[0])
            for df_i in df[1:]:
                len_df_i = len(df_i)
                total_lens += len_df_i
                scaling_coeffs += self.compute_scaling_coefficients(df_i) * len_df_i
            scaling_coeffs /= total_lens
            self.scaling_coefficients = scaling_coeffs


    def compute_scaling_coefficients(self, df):
        scaling_coefficients = {}
        for i, column in enumerate(self.columns_to_scale):
            scaling_coefficients[column] = getattr(self, self.scaling_methods[i])(df[column])

        scaling_coefficients_df = pd.DataFrame.from_dict(data=scaling_coefficients)
        scaling_coefficients_df.index = ['s1', 's2']

        return scaling_coefficients_df

    @staticmethod
    def scaling(array):
        min_ = array.min()
        max_ = array.max()
        s1 = 1 / (max_ - min_)
        s2 = min_ / (min_ - max_)

        return s1, s2

    @staticmethod
    def normalization(array):
        mean_ = array.mean()
        std_ = array.std()
        s1 = 1 / std_
        s2 = - mean_ / std_

        return s1, s2
    
    def scaling_and_normalization(self, array):
        s1, s2 = self.scaling(array)
        new_arr = array * s1 + s2

        return self.normalization(new_arr)

    def transform_df(self, df, scaling_coefficients=None):
        if scaling_coefficients is None:
            scaling_coefficients = self.scaling_coefficients

        df_transformed = df.copy()
        for column in self.columns_to_scale:
            s1, s2 = self.scaling_coefficients[column]
            df_transformed[column] = df_transformed[column] * s1 + s2

        return df_transformed, self.scaling_coefficients



def fix_index_to_1h(df):


    df_copy = df.copy()

    # sort df_copy
    for i in range(len(df)-1):

        dt = df.index[i+1] - df.index[i]

        if dt > pd.Timedelta('1h'):

            df_copy.loc[df.index[i] + pd.Timedelta('1h')] = np.nan
            df_copy.sort_index(inplace=True)

            df_copy = fix_index_to_1h(df_copy)
            break

    df_copy[~df_copy.index.duplicated(keep='first')]
    
    return df_copy

    
    # def get_inverse_transform(self, array, column):

    #     s1, s2 = self.scaling_coefficients[column]
    #     return (array - s2) / s1
    
    # def get_scaling_coefficients(self, column):
    #     return self.scaling_coefficients[column]

    # def get_scaling_coefficients(self, column):
    #     return self.scaling_coefficients[column]
    
    