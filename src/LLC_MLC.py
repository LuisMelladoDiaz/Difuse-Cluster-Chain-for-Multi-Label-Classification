from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from load_dataset import load_arff_data
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from classifier_chain import scale_data


# Clustering de Etiquetas en un Dataset Multietiqueta

# 1. Preprocesamiento de datos
# - Cargar el dataset multietiqueta.
# - Extraer la matriz de Y etiquetas y los matriz X atributos.

def preprocesar_datos(archivo, num_etiquetas, disperso):
    X, Y, características, etiquetas = load_arff_data(
        filename=archivo,
        q=num_etiquetas,
        sparse=disperso,
        return_labels_and_features=True
    )

    nombres_etiquetas = [etiqueta[0] for etiqueta in etiquetas]

    return X, Y, características, nombres_etiquetas

# 2. Construcción de una matriz de similitud entre etiquetas
# - Calcular el índice de Jaccard para cada par de etiquetas:
#   - Para cada par de etiquetas (y_i, y_j):
#     - Calcular la intersección y la unión de ocurrencias de etiquetas.
#     - Almacenar el valor de similitud en una matriz.

def indice_jaccard(y_i, y_j):
    intersección = np.sum((y_i == 1) & (y_j == 1))
    unión = np.sum((y_i == 1) | (y_j == 1))
    return intersección / unión if unión > 0 else 0.0

def construir_matriz_similitud(y):
    num_etiquetas = y.shape[1]
    matriz_similitud = np.zeros((num_etiquetas, num_etiquetas))
    
    for i in range(num_etiquetas):
        for j in range(num_etiquetas):
            matriz_similitud[i, j] = indice_jaccard(y[:, i], y[:, j])
    
    return matriz_similitud

# 3. Conversión a matriz de disimilitud
# - Transformar los valores de similitud en valores de disimilitud 
#   utilizando: dissimilarity = 1 - similarity.

def convertir_a_matriz_disimilitud(matriz_similitud):
    return 1 - matriz_similitud

# 4. Clustering jerárquico aglomerativo
# - Utilizar un algoritmo de clustering jerárquico con el método de enlace Ward.D2:
#   - Tomar la matriz de disimilitud como entrada.
#   - Generar un dendrograma.

def realizar_clustering_jerarquico(matriz_disimilitud, n_clusters, metodo='ward'):
    distancias_comprimidas = matriz_disimilitud[np.triu_indices_from(matriz_disimilitud, k=1)]
    matriz_enlace = linkage(distancias_comprimidas, method=metodo)
    etiquetas_clúster = fcluster(matriz_enlace, n_clusters, criterion='maxclust')
    
    return etiquetas_clúster, matriz_enlace

def dibujar_dendrograma(matriz_enlace, etiquetas):
    plt.figure(figsize=(10, 10))
    dendrogram(matriz_enlace, labels=etiquetas, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrograma")
    plt.xlabel("Etiquetas")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.show()

# 5. Selección de la partición óptima
# - Cortar el dendrograma en diferentes niveles para obtener múltiples particiones.
# - Evaluar cada partición utilizando el coeficiente de silueta.
# - Seleccionar la partición con el coeficiente de silueta más alto.

def seleccionar_mejor_particion(matriz_enlace, matriz_disimilitud, max_clusters, nombres_etiquetas):
    mejor_score = -1
    mejor_particion = None
    mejor_num_clusters = 0

    for num_clusters in range(2, max_clusters + 1):
        partición = fcluster(matriz_enlace, num_clusters, criterion='maxclust')
        score = silhouette_score(matriz_disimilitud, partición, metric="precomputed")

        if score > mejor_score:
            mejor_score = score
            mejor_particion = partición
            mejor_num_clusters = num_clusters

    print(f"\nPartición óptima encontrada con {mejor_num_clusters} clústeres.")
    print(f"Coeficiente de silueta: {mejor_score:.4f}\n")
    print("Resultados del clustering (Etiqueta -> Clúster):")
    for etiqueta, cluster in zip(nombres_etiquetas, mejor_particion):
        print(f"{etiqueta} -> Clúster {cluster}")

    return mejor_particion, mejor_num_clusters, mejor_score

# 6. Entrenamiento de clasificadores por clúster
# - Para cada clúster en la partición seleccionada:
#   - Entrenar un clasificador multietiqueta utilizando solo las etiquetas del clúster.
#   - Incorporar las etiquetas verdaderas del clúster como nuevas características para el siguiente clúster.

def entrenar_clasificadores_por_cluster(X_train, y_train, mejor_particion, nombres_etiquetas, number_of_chains=10):
    modelos = []

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    for cluster_id in np.unique(mejor_particion):
        print(f"\nEntrenando para el clúster {cluster_id}...")
        
        indices_cluster = np.where(mejor_particion == cluster_id)[0]
        y_train_cluster = y_train[:, indices_cluster]

        base_lr = LogisticRegression(max_iter=1000)
        chain_model = MultiOutputClassifier(base_lr)
        chain_model.fit(X_train, y_train_cluster)

        modelos.append(chain_model)

    print("\nEntrenamiento completado para todos los clústeres.")
    return modelos

# 7. Fase de predicción
# - Para un nuevo conjunto de datos:
#   - Predecir las etiquetas de cada clúster en orden.
#   - Usar las predicciones previas como características adicionales para los siguientes clústeres.
#   - Combinar predicciones

def predecir_y_combinar(modelos, X_test, mejor_particion, num_etiquetas):
    y_pred_final = np.zeros((X_test.shape[0], num_etiquetas))
    
    for cluster_id, modelo in enumerate(modelos):
        print(f"Realizando predicciones para el clúster {cluster_id + 1}...")
        
        indices_cluster = np.where(mejor_particion == cluster_id + 1)[0]
        
        y_pred_cluster = modelo.predict(X_test)
        
        y_pred_final[:, indices_cluster] = y_pred_cluster

    return y_pred_final


def LLC_MLC(archivo, num_etiquetas, disperso=False, num_clusters=5):
    X, y, características, nombres_etiquetas = preprocesar_datos(archivo, num_etiquetas, disperso)
    matriz_similitud = construir_matriz_similitud(y)
    matriz_disimilitud = convertir_a_matriz_disimilitud(matriz_similitud)

    etiquetas_clúster, matriz_enlace = realizar_clustering_jerarquico(matriz_disimilitud, num_clusters)

    print("Etiquetas de clúster asignadas:")
    print(etiquetas_clúster)

    dibujar_dendrograma(matriz_enlace, etiquetas=nombres_etiquetas)

    mejor_particion, mejor_num_clusters, mejor_score = seleccionar_mejor_particion(matriz_enlace, matriz_disimilitud, 10, nombres_etiquetas)
    modelos = entrenar_clasificadores_por_cluster(X, y, mejor_particion, nombres_etiquetas)
    y_pred_final = predecir_y_combinar(modelos, X, mejor_particion, num_etiquetas=num_etiquetas)

    print("Predicciones finales para todas las instancias:")
    print(y_pred_final)

    return y_pred_final
