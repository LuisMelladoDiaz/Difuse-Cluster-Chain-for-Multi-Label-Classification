from load_dataset import load_arff_data
from LLC_MLC import *



## DATASET INFO
data_folder = 'datasets/Birds/birds'
train_file = f'{data_folder}-train.arff'
test_file = f'{data_folder}-test.arff'
q_labels = 19
n_clusters = 6
m = 2

# LLC-MLC
LLC_MLC(train_file, q_labels, num_clusters=n_clusters)