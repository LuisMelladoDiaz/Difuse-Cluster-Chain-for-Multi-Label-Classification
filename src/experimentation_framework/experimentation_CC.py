from models.CC import train_CC
from experimentation_framework.experimentation_framework import ExperimentationFramework



def run_cc_experiments(DATASETS, NUM_EXPERIMENTS, NUM_FOLDS, sparse = False):

    framework = ExperimentationFramework(
        model_name="CC",
        model_function=train_CC,
        output_dir="experimentos/CC",
        datasets=DATASETS,
        num_experiments=NUM_EXPERIMENTS,
        num_folds=NUM_FOLDS,
        sparse=sparse

    )

    framework.run_experiments()
    framework.save_results()
