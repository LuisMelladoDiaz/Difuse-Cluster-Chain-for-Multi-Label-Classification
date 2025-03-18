import pandas as pd
from typing import Dict, Any, Callable, Tuple
from utils.preprocessing import load_multilabel_dataset

class ExperimentationFramework:
    def __init__(
        self,
        model_name: str,
        model_function: Callable,
        output_dir: str,
        datasets: Dict[str, int],
        num_experiments: int = 10,
        seeds: list = None,
        **model_params
    ):
        """
        Initialize the experimentation framework.
        
        Args:
            model_name: Name of the model (for logging)
            model_function: Function to train and evaluate the model
            output_dir: Directory to save results
            datasets: Dictionary mapping dataset names to number of labels
            num_experiments: Number of experiments to run
            seeds: List of random seeds to use
            **model_params: Additional parameters specific to the model
        """
        self.model_name = model_name
        self.model_function = model_function
        self.output_dir = output_dir
        self.datasets = datasets
        self.num_experiments = num_experiments
        self.seeds = seeds or [42, 123, 7, 13, 2023, 1995, 101, 555, 999, 314, 69, 1001, 21, 33, 404, 777, 666, 1234, 4321, 9876, 2468]
        self.model_params = model_params
        self.multilabel_datasets_dir = "datasets/multi_etiqueta/"
        self.resultados = {}

    def run_experiments(self):
        """Run all experiments for all datasets."""
        for dataset in self.datasets:
            self.resultados[dataset] = {}
            
            file_path = f"{self.multilabel_datasets_dir}{dataset}/{dataset.lower()}"
            train_file_path = f"{file_path}-train.arff"
            test_file_path = f"{file_path}-test.arff"

            # Load data
            X_train, y_train, _, _ = load_multilabel_dataset(train_file_path, self.datasets[dataset])
            X_test, y_test, _, _ = load_multilabel_dataset(test_file_path, self.datasets[dataset])
            
            for i in range(self.num_experiments):
                seed = self.seeds[i]
                print(f"Ejecutando experimento {i+1} para el dataset {dataset}...")

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
                
                self.resultados[dataset][i+1] = metrics

    def save_results(self):
        """Save all results to CSV files."""
        all_metrics = []
        for dataset, experimentos in self.resultados.items():
            df = pd.DataFrame.from_dict(experimentos, orient='index')
            df.index.name = "Experimento"
            df.loc['Promedio'] = df.mean()
            df.to_csv(f"{self.output_dir}/{dataset}_metrics.csv")
            all_metrics.append(df.loc['Promedio'])

        # Create final table with mean of all metrics
        final_metrics = pd.DataFrame(all_metrics).mean().to_frame().T
        final_metrics.index = ["Global"]
        final_metrics.to_csv(f"{self.output_dir}/global_metrics.csv")

        print("Resultados guardados en CSVs.") 