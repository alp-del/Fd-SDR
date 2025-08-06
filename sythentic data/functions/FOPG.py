import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

def FOPG(x, y, d, isGram=True, rho=1, niter=5, h=None):
    # Compute variance of each column of x and construct diagonal matrix
    variance_x = np.var(x, axis=0, ddof=1)
    sig = np.diag(variance_x)
    
    # Standardize x
    z = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)
    
    # Get dimensions
    n, p = z.shape
    
    # Kernel bandwidth initialization
    c0 = 2.34
    p0 = max(p, 3)
    rn = n**(-1/(2*(p0 + 6)))
    
    if h is None:
        h = c0 * n**(-1/(p0 + 6))
    
    # Initialize B as identity matrix
    B = np.eye(p)
    
    for _ in range(niter):
        kmat = kern(np.dot(z, B), h, 'Gaussian')
        bmat_list = []
        
        for i in range(n):
            b = swls(y, kmat, z, i)
            bmat_list.append(b['b'])
        bmat = np.hstack(bmat_list)
        
        mat = np.dot(bmat, bmat.T) / (n**2)
        eigenvalues, eigenvectors = eigh(mat)
        sort_index = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_index]
        B = eigenvectors[:, :d]
        h = max(rn * h, c0 * n**(-1/(d + 4)))
    
    sig_sqrt_inv = np.diag(1 / np.sqrt(np.diag(sig)))
    beta_final = np.dot(sig_sqrt_inv, B)
    
    return beta_final

def swls(hmat, kmat, x, i):
    # Convert x and kmat to NumPy arrays
    x = np.array(x)
    kmat = np.array(kmat)
    
    # Extract the diagonal of kmat and ensure it's a 1D array
    wi = np.diag(kmat[:, i])
    
    # Compute the matrix of differences
    xdi = x - x[i, :]
    
    # Add a column of ones for the intercept term
    xdi1 = np.hstack([np.ones((xdi.shape[0], 1)), xdi])
    
    # Convert hmat to a NumPy array
    hmat = np.array(hmat,dtype=float)
    
    # Compute the matrix A and vector B
    A = xdi1.T @ wi @ xdi1
    B = xdi1.T @ wi @ hmat
    
    # Solve the linear system A * abmat = B
    abmat = np.linalg.solve(A, B)
    
    return {'a': abmat[0, :], 'b': abmat[1:, :], 'ab': abmat}

def kern(x, h, type):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    k2 = np.dot(x, x.T)
    k1 = np.repeat(np.diag(k2).reshape(-1, 1), n, axis=1)
    k3 = k1.T
    k = k1 - 2 * k2 + k3
    
    if type == 'Gaussian':
        K = (1 / h) * np.exp(-0.5 * (k / h**2))
    elif type == 'Epan':
        K = (1 / h) * 3 / 4 * np.maximum(1 - k / h**2, 0)
    else:
        raise ValueError('Unsupported kernel type')
    
    return K
