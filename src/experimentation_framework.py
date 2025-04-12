import pandas as pd
from typing import Dict, Any, Callable
from utils.preprocessing import load_multilabel_dataset
import os
import time

class ExperimentationFramework:
    def __init__(
        self,
        model_name: str,
        model_function: Callable,
        output_dir: str,
        datasets: Dict[str, int],
        num_experiments: int = 10,
        seeds: list = None,
        num_folds: int = 5,  # Número de folds para cross-validation
        **model_params
    ):
        """
        Initialize the experimentation framework with cross-validation support.
        
        Args:
            model_name: Name of the model (for logging)
            model_function: Function to train and evaluate the model
            output_dir: Directory to save results
            datasets: Dictionary mapping dataset names to number of labels
            num_experiments: Total number of experiments to run
            seeds: List of random seeds to use
            num_folds: Number of folds for cross-validation
            **model_params: Additional parameters specific to the model
        """
        self.model_name = model_name
        self.model_function = model_function
        self.output_dir = output_dir
        self.datasets = datasets
        self.num_experiments = num_experiments
        self.seeds = seeds or [42, 123, 7, 13, 2023, 1995, 101, 555, 999, 314, 69, 1001, 21, 33, 404, 777, 666, 1234, 4321, 9876, 2468]
        self.num_folds = num_folds
        self.model_params = model_params
        self.multilabel_datasets_dir = "datasets/multi_etiqueta/"
        self.resultados = {}
        self.execution_times = {}

    def run_experiments(self):
        """Run all experiments for all datasets with cross-validation."""
        for dataset in self.datasets:
            self.resultados[dataset] = {}
            self.execution_times[dataset] = 0
            dataset_dir = f"{self.multilabel_datasets_dir}{dataset}/"
            
            # Loop through folds
            for fold in range(1, self.num_folds + 1):
                self.resultados[dataset][f"fold{fold}"] = {}
                
                # Load train and test data for the current fold
                train_file_path = os.path.join(dataset_dir, f"train_fold{fold}.arff")
                test_file_path = os.path.join(dataset_dir, f"test_fold{fold}.arff")
                
                X_train, y_train, _, _ = load_multilabel_dataset(train_file_path, self.datasets[dataset])
                X_test, y_test, _, _ = load_multilabel_dataset(test_file_path, self.datasets[dataset])
                
                # Split the total number of experiments into num_folds parts
                experiments_per_fold = self.num_experiments // self.num_folds
                for i in range(experiments_per_fold):
                    seed = self.seeds[(i + (fold - 1) * experiments_per_fold) % len(self.seeds)]
                    print(f"Ejecutando experimento {i+1} para el dataset {dataset}, fold {fold}...")
                    
                    start_time = time.time()
                    
                    # Prepare model parameters
                    current_params = self.model_params.copy()
                    current_params['random_state'] = seed
                    
                    # Add num_labels parameter if the model is FCCC
                    if self.model_name == "FCCC":
                        current_params['num_labels'] = self.datasets[dataset]
                    
                    # Train and evaluate model
                    prediction, metrics = self.model_function(
                        X_train, y_train, X_test, y_test,
                        **current_params
                    )
                    
                    end_time = time.time()
                    self.execution_times[dataset] += (end_time - start_time)
                    
                    self.resultados[dataset][f"fold{fold}"][i+1] = metrics

    def save_results(self):
        """Save results to a single CSV per dataset and a global CSV with dataset-level averages."""
        global_metrics = []
        
        for dataset, experimentos in self.resultados.items():
            all_folds = []
            
            for fold, fold_experiments in experimentos.items():
                df = pd.DataFrame.from_dict(fold_experiments, orient='index')
                df.index.name = "Experimento"
                all_folds.append(df)
            
            # Concatenate all folds for the dataset
            dataset_df = pd.concat(all_folds)
            dataset_df.loc['Promedio'] = dataset_df.mean()
            dataset_df.to_csv(f"{self.output_dir}/{dataset}_metrics.csv")
            
            # Save dataset-level average metrics
            avg_metrics = dataset_df.loc['Promedio']
            avg_metrics['Tiempo_promedio_por_experimento'] = self.execution_times[dataset] / self.num_experiments
            global_metrics.append(avg_metrics)
        
        # Create global summary table where each row is a dataset average
        final_metrics = pd.DataFrame(global_metrics)
        final_metrics.index = self.resultados.keys()
        
        # Add global average at the end
        final_metrics.loc['Global'] = final_metrics.mean()
        final_metrics.to_csv(f"{self.output_dir}/global_metrics.csv")
        
        print("Resultados guardados en CSVs.")
