from LCC_MLC import *
from CC import train_CC, train_ECC
from preprocess import load_csv_and_train_test_split

## Clasifier Chain ##################################################################################################################3

file = 'datasets/binario/diabetes.csv'
X_train, X_test, y_train, y_test = load_csv_and_train_test_split(file)

Y_pred_CC = train_CC(X_train, y_train, X_test)
Y_pred_ECC_chains, Y_pred_ECC_ensemble = train_ECC(X_train, y_train, X_test)

print("Predicciones con Classifier Chain (CC):")
print(Y_pred_CC)

print("Predicciones con Ensemble of Classifier Chains (ECC):")
print("Predicciones por cada cadena:")
print(Y_pred_ECC_chains)

print("Predicciones del ensemble:")
print(Y_pred_ECC_ensemble)

## LLC-MLC ##################################################################################################################3

data_folder = 'datasets/multi_etiqueta/Birds/birds'
train_file = f'{data_folder}-train.arff'
test_file = f'{data_folder}-test.arff'
q_labels = 19
n_clusters = 6

#LLC_MLC(train_file, q_labels, num_clusters=n_clusters)