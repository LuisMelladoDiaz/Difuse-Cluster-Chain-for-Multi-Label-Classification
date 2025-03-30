import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def load_dataset(filepath):

    if filepath.endswith(".arff"):
        from scipy.io import arff
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
    else:
        df = pd.read_csv(filepath)
    return df

def kfold_split(df, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    
    for train_index, test_index in kf.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        splits.append((train_data, test_data))
    
    return splits
