import pandas as pd

def createOneHot(df, encode_columns):
    """One-hot encode categorical columns in a DataFrame."""
    
    for column in encode_columns:
        # Create one-hot encoded columns with prefix
        dummies = pd.get_dummies(df[column], prefix=column)
        
        # Concatenate the one-hot encoded columns to the original DataFrame
        df = pd.concat([df, dummies], axis=1)

    # Drop the original categorical columns
    return df.drop(columns=encode_columns)
    