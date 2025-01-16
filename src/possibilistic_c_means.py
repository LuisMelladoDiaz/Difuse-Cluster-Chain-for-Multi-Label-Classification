import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename, columns):
    """Load specified columns from a CSV file and return as a NumPy array."""
    data = pd.read_csv(filename)
    return np.array(data.loc[:, columns])

def initf(c, data_n):
    """Initialize the membership matrix with random values normalized to sum to 1 for each data point."""
    A = np.random.random(size=(c, data_n))
    col_sum = np.sum(A, axis=0)
    return A/col_sum

def pdistfcm(cntr, data):
    """Calculate the distance between each cluster center and data point."""
    out = np.zeros(shape=(cntr.shape[0], data.shape[0]))
    for k in range(cntr.shape[0]):
        out[k] = np.sqrt(np.sum((np.power(data-cntr[k], 2)).T, axis=0)) + 1e-10
    return out

def pstepfcm(data, cntr, U, T, expo, a, b, nc, ni):
    """Perform a single iteration step of the Possibilistic Fuzzy C-Means algorithm."""
    mf = np.power(U, expo)
    tf = np.power(T, nc)
    tfo = np.power((1 - T), nc)
    cntr = (np.dot(a * mf + b * tf, data).T / np.sum(a * mf + b * tf, axis=1).T).T
    dist = pdistfcm(cntr, data)
    obj_fcn = np.sum(np.sum(np.power(dist, 2) * (a * mf + b * tf), axis=0)) + np.sum(ni * np.sum(tfo, axis=0))
    ni = mf * np.power(dist, 2) / (np.sum(mf, axis=0))
    tmp = np.power(dist, (-2/(expo - 1)))
    U = tmp/(np.sum(tmp, axis=0))
    tmpt = np.power((b / ni) * np.power(dist, 2), (1 / (nc - 1)))
    T = 1 / (1 + tmpt)
    return U, T, cntr, obj_fcn, ni

def pfcm(data, c, expo=2, max_iter=1000, min_impro=0.005, a=1, b=4, nc=3):
    """Possibilistic Fuzzy C-Means Clustering Algorithm."""
    obj_fcn = np.zeros(shape=(max_iter, 1))
    ni = np.zeros(shape=(c, data.shape[0]))
    U = initf(c, data.shape[0])
    T = initf(c, data.shape[0])
    cntr = np.random.uniform(low=np.min(data), high=np.max(data), size=(c, data.shape[1]))
    for i in range(max_iter):
        current_cntr = cntr
        U, T, cntr, obj_fcn[i], ni = pstepfcm(data, cntr, U, T, expo, a, b, nc, ni)
        if i > 1:
            if abs(obj_fcn[i] - obj_fcn[i-1]) < min_impro:
                break
            elif np.max(abs(cntr - current_cntr)) < min_impro:
                break
    return cntr, U, T, obj_fcn

def plot_clusters(data, labels):
    """Plot the clustered data points with different colors based on labels."""
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='nipy_spectral')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()

# Example usage for the Housing dataset using Possibilistic Fuzzy C-Means
filename = 'datasets/Housing/housing.csv'
data = load_data(filename, ['latitude', 'longitude'])
cntr, U, T, obj_fcn = pfcm(data, c=5)
labels = np.argmax(U, axis=0)
plot_clusters(data, labels)
