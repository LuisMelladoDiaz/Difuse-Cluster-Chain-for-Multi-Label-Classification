from load_dataset import load_arff_data
from classifier_chain import *
from implementación_inicial import implementacion_inicial



## DATASET INFO
data_folder = 'datasets/Birds/birds'
train_file = f'{data_folder}-train.arff'
test_file = f'{data_folder}-test.arff'
q_labels = 19
n_clusters = 6
m = 2

## LOAD DATASET
X_train, y_train = load_arff_data(train_file, q_labels, sparse=False)
X_test, y_test = load_arff_data(test_file, q_labels, sparse=False)

# CLASSIFIER CHAIN
num_chains = 10
chain_jaccard_scores, ensemble_jaccard_score = classifier_chain(X_train, y_train, X_test, y_test, number_of_chains=num_chains)

# ONE VS REST CLASSIFIER
ovr_jaccard_score = jaccard_similarity(y_test, one_vs_rest_classifier(X_train, y_train, X_test))

# JACCARD SIMILARITY
plot_jaccard_scores(ovr_jaccard_score, chain_jaccard_scores, ensemble_jaccard_score, num_chains)

# IMPLEMENTACIÓN INICIAL - FUZZY C MEANS
implementacion_inicial(train_file, q_labels, n_clusters)
