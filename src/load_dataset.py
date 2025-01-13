import arff
import numpy as np
import pandas as pd
import time
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split

def load_arff_data(filename, q, sparse, to_array=True):
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
    X, y = load_from_arff(
        filename,
        label_count=q,
        label_location='end',
        load_sparse=sparse
    )

    if to_array:
        X = X.toarray()
        y = y.toarray()

    print(f"""
Dataset Multietiqueta Cargado Exitosamente     
                      
       Archivo: {filename}                                       
       Instancias: {X.shape[0]}                                   
       Características: {X.shape[1]}                             
       Etiquetas: {y.shape[1]}                                    
    """)

    return X, y






