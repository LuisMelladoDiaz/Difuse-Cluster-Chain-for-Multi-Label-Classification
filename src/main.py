from models.LCC_MLC import LCC_MLC
from models.CC import train_CC, train_ECC
from models.LDCC_MLC import LCC_MLC_Fuzzy
from utils.preprocessing import load_csv_and_train_test_split


## Clasifier Chain ###########################################################################################################################################################################################

def example_CC():
    file = 'datasets/binario/diabetes.csv'
    X_train, X_test, y_train, y_test = load_csv_and_train_test_split(file)

    Y_pred_CC = train_CC(X_train, y_train, X_test)

    print("Predicciones con Classifier Chain (CC):")
    print(Y_pred_CC)

def example_ECC():
    file = 'datasets/binario/diabetes.csv'
    X_train, X_test, y_train, y_test = load_csv_and_train_test_split(file)

    Y_pred_ECC_chains, Y_pred_ECC_ensemble = train_ECC(X_train, y_train, X_test)

    print("Predicciones del ensemble ECC:")
    print(Y_pred_ECC_ensemble)

## LCC-MLC ###########################################################################################################################################################################################

def example_LCC_MLC():
    data_folder = 'datasets/multi_etiqueta/Birds/birds'
    train_file = f'{data_folder}-train.arff'
    test_file = f'{data_folder}-test.arff'
    q_labels = 19

    LCC_MLC(train_file, q_labels)

## LDCC-MLC ###########################################################################################################################################################################################

def example_LDCC_MLC():
    data_folder = 'datasets/multi_etiqueta/Birds/birds'
    train_file = f'{data_folder}-train.arff'
    test_file = f'{data_folder}-test.arff'
    q_labels = 19

    LCC_MLC_Fuzzy(train_file, q_labels)

## Examples of use ###########################################################################################################################################################################################
#example_CC()
#example_ECC()
example_LCC_MLC()
#example_LDCC_MLC()