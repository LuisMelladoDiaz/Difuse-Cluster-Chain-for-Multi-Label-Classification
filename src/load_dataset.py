import arff
import numpy as np
import pandas as pd
import time
from skmultilearn.dataset import load_from_arff

def load_arff_data(filename, q, sparse, to_array=True, return_labels_and_features = False):
    '''
    Carga un archivo ARFF multietiqueta y devuelve las características (X) y etiquetas (y).

    Parámetros:
    -----------
    filename : str
        Ruta al archivo ARFF que contiene el dataset.
    q : int
        Número de etiquetas presentes en el dataset.
    sparse : bool
        Indica si los datos están en formato disperso (True) o denso (False).
    to_array : bool, opcional (por defecto=True)
        Si es True, convierte las matrices dispersas a arrays densos.

    Retorna:
    --------
    X : array-like o scipy.sparse
        Matriz de características del dataset.
    y : array-like o scipy.sparse
        Matriz de etiquetas del dataset.
    
    Ejemplo:
    --------
    X, y = load_arff_data('datasets/Birds/birds-train.arff', q=19, sparse=False)
    '''
    X, y, features, labels = load_from_arff(
        filename,
        label_count=q,
        label_location='end',
        load_sparse=sparse,
        return_attribute_definitions=True
    )

    if to_array:
        X = X.toarray()
        y = y.toarray()

    print(f"""
Dataset Multietiqueta Cargado Exitosamente     
                      
       Archivo: {filename}                                       
       Instancias: {X.shape[0]}                                   
       Características: {X.shape[1]}                             
       Numero de etiquetas: {y.shape[1]}                                    
    """)

    
    
    if return_labels_and_features:
        return X, y, features, labels
    else:
        return X, y

def load_csv(filename, columns):
    """Load specified columns from a CSV file and return as a NumPy array."""
    data = pd.read_csv(filename)
    return np.array(data.loc[:, columns])





