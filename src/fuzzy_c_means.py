import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv(filename, columns):
    """Load specified columns from a CSV file and return as a NumPy array."""
    data = pd.read_csv(filename)
    return np.array(data.loc[:, columns])

def initialize_membership_matrix(data_size, K):
    """Initialize the membership matrix with random values normalized to sum to 1 for each data point."""
    U = np.random.rand(data_size, K)
    U /= np.sum(U, axis=1)[:, np.newaxis]
    return U

def calculate_centroids(data, U, m):
    """Calculate centroids based on the membership matrix and the fuzziness parameter m."""
    K = U.shape[1]
    centroids = np.zeros((K, data.shape[1]))
    for i in range(K):
        numer = np.sum((U[:, i]**m)[:, np.newaxis] * data, axis=0)
        denom = np.sum(U[:, i]**m)
        centroids[i, :] = numer / denom
    return centroids

def calculate_membership(data, centroids, m):
    """Update the membership matrix based on current centroids and the fuzziness parameter m."""
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    U_new = 1 / (distances ** (2/(m-1)) * np.sum((1/distances) ** (2/(m-1)), axis=1)[:, np.newaxis])
    return U_new

def fuzzy_c_means(data, K=5, m=3, max_iterations=100, tol=1e-5):
    """Perform the Fuzzy C-Means clustering algorithm with specified parameters."""
    U = initialize_membership_matrix(data.shape[0], K)
    for _ in range(max_iterations):
        centroids = calculate_centroids(data, U, m)
        U_new = calculate_membership(data, centroids, m)
        if np.linalg.norm(U_new - U) <= tol:
            break
        U = U_new
    labels = np.argmax(U, axis=1)
    return labels, centroids, U

def plot_clusters(data, labels):
    """Plot the clustered data points with different colors based on labels."""
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='nipy_spectral')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()

# Example usage
filename = 'datasets\Housing\housing.csv'
data = load_csv(filename, ['latitude', 'longitude'])
labels, centroids, U = fuzzy_c_means(data)
plot_clusters(data, labels)
