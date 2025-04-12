import numpy as np
import pandas as pd
from skmultilearn.dataset import load_from_arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## DATASET LOADING ##########################################################################################################################################

def load_csv(filename, columns):
    """Load specified columns from a CSV file and return as a NumPy array."""
    data = pd.read_csv(filename)
    return np.array(data.loc[:, columns])

def load_csv_and_train_test_split(file):
    data = pd.read_csv(file)

    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    y = y.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def load_arff_data(filename, label_count, sparse, to_array=True, return_labels_and_features=False, debug = False):
    """
    Load a multi-label ARFF file and return the features (X) and labels (y).

    Args:
        filename: Path to the ARFF file.
        label_count: Number of labels in the dataset.
        sparse: Boolean indicating whether the data is sparse.
        to_array: Convert the data to NumPy arrays if True.
        return_labels_and_features: If True, also return feature and label definitions.

    Returns:
        X: Features array.
        y: Labels array.
        (Optional) features, labels: Feature and label definitions if return_labels_and_features is True.
    """
    X, y, features, labels = load_from_arff(
        filename,
        label_count=label_count,
        label_location='end',
        load_sparse=sparse,
        return_attribute_definitions=True
    )

    if to_array:
        X = X.toarray()
        y = y.toarray()

    if debug:    

        print(f"""
            Multi-label Dataset Successfully Loaded
            File: {filename}
            Instances: {X.shape[0]}
            Features: {X.shape[1]}
            Number of Labels: {y.shape[1]}
        """)

    if return_labels_and_features:
        return X, y, features, labels
    else:
        return X, y
    
def load_multilabel_dataset(file_path, num_labels, sparse=False):
    
    X, Y, features, labels = load_arff_data(file_path, num_labels, sparse, return_labels_and_features=True)
    return X, Y, features, [label[0] for label in labels]

## DATA SCALING ##########################################################################################################################################

def scale_data(X_train, X_test):
    """
    Scale training and testing datasets using the same StandardScaler instance.
    
    Args:
        X_train: Training feature set.
        X_test: Testing feature set.

    Returns:
        Scaled training and testing feature sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def scale_training_set(X_train):
    """
    Scale the training dataset using StandardScaler.

    Args:
        X_train: Training feature set.

    Returns:
        Scaled training feature set.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X_train)


