import csv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss
)

def evaluate_multilabel_classification(y_true, y_pred):
    """Evaluates a multi-label classifier using common metrics."""
    
    # Calcular métricas
    metrics = {
        "Accuracy (Subset)": accuracy_score(y_true, y_pred),  # Exact match (todas las etiquetas correctas)
        "Hamming Loss": hamming_loss(y_true, y_pred),         # Errores a nivel de etiqueta
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Samples)": precision_score(y_true, y_pred, average="samples", zero_division=0),
        "Recall (Samples)": recall_score(y_true, y_pred, average="samples", zero_division=0),
        "F1-Score (Samples)": f1_score(y_true, y_pred, average="samples", zero_division=0)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def save_results_to_csv(metrics, y_pred, output_file):
    """Guarda las métricas y la matriz de predicciones en un archivo CSV."""
    
    # Primero, abre el archivo CSV en modo escritura
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Guardar las métricas
        header_metrics = ["Metric", "Value"]
        writer.writerow(header_metrics)  # Escribe la cabecera para las métricas
        
        # Escribir cada métrica con su valor
        for metric, value in metrics.items():
            writer.writerow([metric, value])
        
        # Separar las métricas de las predicciones
        writer.writerow([])  # Línea vacía para separar las métricas de las predicciones
        
        # Guardar las predicciones
        header_predictions = ["L_" + str(i+1) for i in range(y_pred.shape[1])]  # Encabezado para las etiquetas
        writer.writerow(header_predictions)  # Escribe la cabecera para las predicciones
        
        # Escribir cada fila de las predicciones
        for row in y_pred:
            writer.writerow(row)
        
    print(f"Results saved to {output_file}")

def run_evaluation_and_save_results(y_true, y_pred, output_file):
    metrics = evaluate_multilabel_classification(y_true, y_pred)
    save_results_to_csv(metrics, y_pred, output_file)
