import numpy as np
import pandas as pd

class Standardization:

    def __init__(self):
        # Initialize mean and standard deviation attributes
        self.numerical_columns = None
        self.mean = None
        self.std = None

    def fit(self, data, numerical_columns):
        """Calculate the mean and standard deviation of the numerical data."""
        
        self.numerical_columns = numerical_columns
        self.mean = np.mean(data[numerical_columns], axis=0)
        self.std = np.std(data[numerical_columns], axis=0)

    def transform(self, data):
        """Standardize the numerical data using the pre-calculated mean and standard deviation."""

        data_scaled = (data[self.numerical_columns] - self.mean) / self.std
        categorical_data = data.drop(columns=self.numerical_columns)

        return pd.concat([data_scaled, categorical_data], axis=1)

        
    
    def fit_transform(self, data, numerical_columns):
        """Fit the data and then standardize it."""

        self.fit(data, numerical_columns)
        return self.transform(data)
