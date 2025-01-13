from load_dataset import load_arff_data



## DATASET INFO
data_folder = 'datasets/Birds/'
train_file = f'{data_folder}birds-train.arff'
test_file = f'{data_folder}birds-test.arff'
q_labels = 19 

## LOAD TRAIN DATASET
X_train, y_train = load_arff_data(train_file, q_labels, sparse=False)


# LOAD TEST DATASET
X_test, y_test = load_arff_data(test_file, q_labels, sparse=False)



