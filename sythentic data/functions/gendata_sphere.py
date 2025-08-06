import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.stats import norm

def orth_mat(x):
    temp = x.T @ x 
    C = temp**(-1/2)
    return x*C

def gendata_sphere(n, p, rho=0, model='model1'):
    # Validate inputs
    if p <= 5:
        raise ValueError("Dimension p must be greater than or equal to 6")
    if model not in ['model1', 'model2', 'model3']:
        raise ValueError("Model must be one of 'model1', 'model2', 'model3'")
    
    # Generate AR1(rho) predictors with fixed values
    if rho > 0:
        x = np.zeros((n, p))
        x[:, 0] = np.random.normal(size=n) # Set all values in the first column to 1
        for i in range(1, p):
            x[:, i] = rho * x[:, i - 1] + np.sqrt(1 - rho**2) * np.random.normal(size=n)
    else:
        x = np.random.normal(size=(n, p))  # Set all values to 1
    
    
    x = norm.cdf(x)

    data = {}

    if model == 'model1':
        delta = np.random.normal(0, 0.2, size=n)
        m_x = np.column_stack((np.cos(np.pi * x[:, 0]), np.sin(np.pi * x[:, 0])))
        eps = np.column_stack((-delta * np.sin(np.pi * x[:, 0]), delta * np.cos(np.pi * x[:, 0])))
        y = np.array([np.cos(np.abs(delta[i])) * m_x[i] + np.sin(np.abs(delta[i])) / np.abs(delta[i]) * eps[i] for i in range(n)])
        data['x'] = x
        data['y'] = y
        data['d0'] = 1
        data['b0'] = np.concatenate(([1], np.zeros(p - 1)))
    
    elif model == 'model2':
        def genone(x):
            delta = np.random.normal(0, 0.2, size=2)
            m_x = np.array([np.sqrt(1 - x[1]**2) * np.cos(np.pi * x[0]), np.sqrt(1 - x[1]**2) * np.sin(np.pi * x[0]), x[1]])
            temp = np.array([[-np.sqrt(1 - x[1]**2) * np.sin(np.pi * x[0]), np.sqrt(1 - x[1]**2) * np.cos(np.pi * x[0]), 0]])
            temp = temp.flatten()
            v1 = orth_mat(temp)
            v2 = orth_mat(np.cross(v1, m_x))
            eps = delta[0] * v1 + delta[1] * v2
            y = np.cos(np.sqrt(np.sum(eps**2))) * m_x + np.sin(np.sqrt(np.sum(eps**2))) / np.sqrt(np.sum(eps**2)) * eps
            return np.concatenate((x, y))
        
        data_full = np.array([genone(x[i]) for i in range(n)])
        data['x'] = data_full[:, :p]
        data['y'] = data_full[:, p:p+3]
        data['d0'] = 2
        col1 = np.concatenate(([1], np.zeros(p - 1)))
        col2 = np.concatenate(([0], [1], np.zeros(p - 2)))
        data['b0'] = np.vstack((col1, col2)).T
    
    elif model == 'model3':
        eps1 = np.random.normal(0, 0.2, size=n)
        eps2 = np.random.normal(0, 0.2, size=n)
        m_x = np.column_stack((np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                               np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
                               np.cos(np.pi * x[:, 0])))
        y = np.column_stack((np.sin(np.pi * x[:, 0] + eps1) * np.sin(np.pi * x[:, 1] + eps2),
                             np.sin(np.pi * x[:, 0] + eps1) * np.cos(np.pi * x[:, 1] + eps2),
                             np.cos(np.pi * x[:, 0] + eps1)))
        data['x'] = x
        data['y'] = y
        data['d0'] = 2
        col1 = np.concatenate(([1], np.zeros(p - 1)))
        col2 = np.concatenate(([0], [1], np.zeros(p - 2)))
        data['b0'] = np.vstack((col1, col2)).T

    return data
