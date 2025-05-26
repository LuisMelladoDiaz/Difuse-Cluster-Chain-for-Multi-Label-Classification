from models.CC import train_ECC
from experimentation_framework.experimentation_framework import ExperimentationFramework


def run_ecc_experiments(DATASETS, NUM_EXPERIMENTS, NUM_CHAINS, NUM_FOLDS, sparse = False):
    framework = ExperimentationFramework(
        model_name="ECC",
        model_function=train_ECC,
        output_dir="experimentos/ECC",
        datasets=DATASETS,
        num_experiments=NUM_EXPERIMENTS,
        number_of_chains=NUM_CHAINS,
        num_folds=NUM_FOLDS,
        sparse=sparse
    )

    framework.run_experiments()
    framework.save_results()
