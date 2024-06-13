import numpy as np

def stratified_train_test_split(X, y, split=0.2):
    """Perform stratified train-test split on the data."""

    # get class names and counts
    label_counts = np.array(y.value_counts())
    label_names = np.array(y.unique())
    
    # calculate the number of samples for each class in the test set
    test_counts = (label_counts * split).astype(int)

    # dictionary to store indexes of train and test data
    train_test_index = {'train': [], 'test': []}

    for label, count in zip(label_names, test_counts):
        # get indexes of samples belonging to the current class
        indexes = y.loc[y == label].index

        # randomly select 'count' number of samples for the test set
        test_index = np.random.choice(indexes, count, replace=False)
        
        # remaining samples are for the train set
        train_index = indexes.difference(test_index)
        train_index = list(train_index)  # Convert to list

        # add the indexes to the dictionary
        train_test_index['train'].extend(train_index)
        train_test_index['test'].extend(test_index)

    X_train = X.loc[train_test_index['train']].reset_index(drop=True)
    y_train = y.loc[train_test_index['train']].reset_index(drop=True)
    X_test = X.loc[train_test_index['test']].reset_index(drop=True)
    y_test = y.loc[train_test_index['test']].reset_index(drop=True)

    return X_train, X_test, y_train, y_test