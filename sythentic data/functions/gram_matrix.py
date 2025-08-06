import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

# gram_matrix is for scenario II when Y is matrices data

# gram_matrix2 is for scenario I when Y is distributional data

# gram_matrix3 is for Geodesic

# gram_matrix4 is Euclidean distance

def gram_matrix(X, phi, complexity=0.01):
    """
    Parameters:
    X: Random objects - A P by Q by N array for distributional matrices
    phi: A scaling factor (similar to the MATLAB implementation)
    complexity: Tuning parameter in the reproducing kernel (default is 0.01)

    Returns:
    N by N kernel Gram matrix
    """
    X = np.array(X, dtype=float)
    N = X.shape[0]  # Number of data points
    
    # Compute distance matrix using Frobenius norm
    k = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            k[i, j] = np.linalg.norm(X[i] - X[j], 'fro')
            k[j, i] = k[i, j]  # Symmetric matrix
    
    # Compute kernel Gram matrix
    sigma2 = np.sum(k ** 2) / (N * (N - 1) / 2)
    gamma = complexity / sigma2 * phi
    K = np.exp(-gamma * (k ** 2))
    
    return K





def wasserstein1d(a, b, p=2):
    """
    Compute the Wasserstein distance between two 1D distributions.
    
    Parameters:
        a (numpy.ndarray): 1D array of the first distribution.
        b (numpy.ndarray): 1D array of the second distribution.
        p (int): Power parameter for the Wasserstein distance. Default is 2.
    
    Returns:
        float: Wasserstein distance between a and b.
    """
    if len(a) != len(b):
        raise ValueError('Vectors must have the same length.')
    
    # Sort the arrays
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    
    # Compute the Wasserstein distance
    distance = np.mean(np.abs(b_sorted - a_sorted)**p)**(1/p)
    
    return distance

def dist(x):
    """
    Compute the Wasserstein distance matrix for a given matrix x.
    
    Parameters:
        x (numpy.ndarray): A n by m matrix where each row is an iid observation from a distribution.
    
    Returns:
        numpy.ndarray: Wasserstein distance matrix.
    """
    n = x.shape[0]
    dmatrix = np.zeros((n, n))
    
    # Generate all unique pairs of indices
    indices = list(combinations(range(n), 2))
    
    for i, k in indices:
        dmatrix[i, k] = wasserstein1d(x[i, :], x[k, :])
    
    # Make the matrix symmetric
    dmatrix = dmatrix + dmatrix.T
    
    return dmatrix

def gram_matrix2(x, lambda_, kernel='Gaussian'):
    """
    Compute the kernel Gram matrix.
    
    Parameters:
        x (numpy.ndarray): A n by m matrix for distributional objects.
        lambda_ (float): Tuning parameter in the reproducing kernel.
        kernel (str): Type of kernel - 'Gaussian' or 'Laplacian'. Default is 'Gaussian'.
    
    Returns:
        numpy.ndarray: n by n kernel Gram matrix.
    """
    # Convert input to matrix if needed
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Compute distance matrix for distributional data
    k = dist(x)
    complexity = 1
    
    # Compute normalization factor
    n_pairs = n * (n - 1) / 2
    
    # Compute kernel Gram matrix
    if kernel == 'Gaussian':
        sigma2 = np.sum(k**2) / n_pairs
        gamma = complexity / sigma2 *lambda_
        K = np.exp(-gamma * k**2)
    elif kernel == 'Laplacian':
        sigma = np.sum(k) / n_pairs
        gamma = complexity / sigma
        K = np.exp(-gamma * k)
    else:
        raise ValueError('Unknown kernel')
    
    return K


def gram_matrix3(x, lambda_, kernel='Gaussian'):
    """
    Compute the kernel Gram matrix.
    
    Parameters:
        x (numpy.ndarray): A n by m matrix for distributional objects.
        lambda_ (float): Tuning parameter in the reproducing kernel.
        kernel (str): Type of kernel - 'Gaussian' or 'Laplacian'. Default is 'Gaussian'.
    
    Returns:
        numpy.ndarray: n by n kernel Gram matrix.
    """
    # Convert input to matrix if needed
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Compute distance matrix for distributional data
    k = dist_sphere(x)
    complexity = 1
    
    # Compute normalization factor
    n_pairs = n * (n - 1) / 2
    
    # Compute kernel Gram matrix
    if kernel == 'Gaussian':
        sigma2 = np.sum(k**2) / n_pairs
        gamma = complexity / sigma2 *lambda_
        K = np.exp(-gamma * k**2)
    elif kernel == 'Laplacian':
        sigma = np.sum(k) / n_pairs
        gamma = complexity / sigma
        K = np.exp(-gamma * k)
    else:
        raise ValueError('Unknown kernel')
    
    return K




def dist_sphere(x):
    
    x = np.asarray(x)
    n = x.shape[0]

    
    dmatrix = np.zeros((n, n))

    
    def distance(i, k):
        
        dot_product = np.dot(x[i], x[k])
        clipped_value = np.clip(dot_product, -1.0, 1.0)
        return np.arccos(clipped_value)

    
    for i in range(n):
        for k in range(i + 1, n):
            dmatrix[i, k] = distance(i, k)

    
    dmatrix += dmatrix.T

    return dmatrix

def gram_matrix4(x, lambda_, kernel='Gaussian'):
    """
    Compute the kernel Gram matrix.
    
    Parameters:
        x (numpy.ndarray): A n by m matrix for distributional objects.
        lambda_ (float): Tuning parameter in the reproducing kernel.
        kernel (str): Type of kernel - 'Gaussian' or 'Laplacian'. Default is 'Gaussian'.
    
    Returns:
        numpy.ndarray: n by n kernel Gram matrix.
    """
    # Convert input to matrix if needed
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Compute distance matrix for distributional data
    k = dist_eu(x)
    complexity = 1
    
    # Compute normalization factor
    n_pairs = n * (n - 1) / 2
    
    # Compute kernel Gram matrix
    if kernel == 'Gaussian':
        sigma2 = np.sum(k**2) / n_pairs
        gamma = complexity / sigma2 *lambda_
        K = np.exp(-gamma * k**2)
    elif kernel == 'Laplacian':
        sigma = np.sum(k) / n_pairs
        gamma = complexity / sigma
        K = np.exp(-gamma * k)
    else:
        raise ValueError('Unknown kernel')
    
    return K

def dist_eu(x):
   
    n = x.shape[0]
    dmatrix = np.zeros((n, n))
    
    # Generate all unique pairs of indices
    indices = list(combinations(range(n), 2))
    
    for i, k in indices:
        dmatrix[i, k] = np.linalg.norm(x[i]- x[k])
    
    # Make the matrix symmetric
    dmatrix = dmatrix + dmatrix.T
    
    return dmatrix