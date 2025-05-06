import numpy as np
import scipy.linalg as la
from scipy.linalg import norm as linalg_norm
from scipy.stats import norm as stats_norm, gamma
from scipy import stats
from lib_fun import *  
from MAVE1 import *
from gram_matrix import *
from FOPG import *
from scipy.linalg import sqrtm
from numpy.linalg import inv, pinv
import time
from sklearn.covariance import GraphicalLassoCV
# Parameters
n = 400
p = 20
q = 100
d_0 = 2  
phi = 0.5
sigma_epsilon = np.sqrt(1 - phi**2)
beta_3 = np.concatenate([[1, 2], np.zeros(p - 3), [2]])
beta_4 = np.concatenate(([0, 0, 1, 2, 2], np.zeros(p - 5)))

beta = np.vstack([beta_3, beta_4])
beta_true = beta.T

metric = "Wasserstein"
num_repeats = 100
neigh = False
np.random.seed(123)

errors1 = np.zeros(num_repeats)
errors2 = np.zeros(num_repeats)
errors3 = np.zeros(num_repeats)
Time1 = np.zeros(num_repeats)
Time2 = np.zeros(num_repeats)
Time3 = np.zeros(num_repeats)



for repeat in range(num_repeats):
    # Data Generation
    X = np.zeros((n, p))

    for sample in range(n):
        U = np.zeros(p)
        U[0] = np.random.normal(0, 1)

        for t in range(1, p):
            epsilon_t = np.random.normal(0, sigma_epsilon)
            U[t] = phi * U[t-1] + epsilon_t

        x = stats_norm.cdf(U)
        X[sample, :] = x

    # Compute mu and mu_y
    mu_y = np.zeros(n)
    sigma_y = np.zeros(n)


    for i in range(n):
        x = X[i, :]
        alpha = (2 + 2 * np.dot(x, beta_4))**2 / 0.5
        beta = 0.5 / (2 + 2 * np.dot(x, beta_4))
        sigma_y[i] = gamma.rvs(alpha, scale=beta)
    
        mu = 3 * np.sin(np.dot(x, beta_3))
        mu_y[i] = np.random.normal(mu, 0.5**2)

    # Adjust sigma_y values
    sigma_y = np.clip(sigma_y, 0.1, 10)

    # Generate y matrix
    sigma_matrix = 0.4 * np.diag(sigma_y) #alpha =0.2/0.4
    y = np.random.randn(n, q)
    y = np.dot(sigma_matrix, y) + mu_y[:, np.newaxis]
    Nb=[]
    

    if not neigh:
        # Neighborhood is unknown
    
        graphical_lasso_model = GraphicalLassoCV()
        graphical_lasso_model.fit(X)
        omega = graphical_lasso_model.precision_
        np.where(np.sum(omega!=0, axis = 1) > 1)[0]
        
    for j in range(p):
        Ni = (np.nonzero(omega[j, :])[0]).tolist()
        Nb.append(Ni)
    ygram = gram_matrix2(y,10)
    ygram = sqrtm(ygram)
    start_time = time.time()
    beta_gwire, _ = gwire_cv(X, y, Nb, metric, d_0, fold=5)
    gwire_time = time.time() - start_time
    Time1[repeat] = gwire_time
    b1 = beta_gwire
    ygram2 = gram_matrix2(y,1)
    start_time = time.time()
    beta_fopg = FOPG(X,ygram2,d_0)
    fopg_time = time.time() - start_time
    Time2[repeat] = fopg_time
    b3 = beta_fopg
    initial = beta_fopg
    start_time = time.time()
    beta_dcov, _,_ = MAVE1(X.T,ygram,initial)
    mave1_time = time.time() - start_time
    Time3[repeat] = mave1_time
    b2 = beta_dcov
   

    error2 = linalg_norm(b2 @ inv(b2.T @ b2) @ b2.T - beta_true @ inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    error3 = linalg_norm(b3 @ inv(b3.T @ b3) @ b3.T - beta_true @ inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    error1 = linalg_norm(b1 @ pinv(b1.T @ b1) @ b1.T - beta_true @ pinv(beta_true.T @ beta_true) @ beta_true.T, 'fro')

    errors1[repeat] = error1
    errors2[repeat] = error2
    errors3[repeat] = error3

# Compute mean and standard deviation of the errors
mean_errorG = np.mean(errors1)
sd_errorG = np.std(errors1)
mean_errorD = np.mean(errors2)
sd_errorD = np.std(errors2)
mean_errorF = np.mean(errors3)
sd_errorF = np.std(errors3)

meantimeG = np.mean(Time1)
sdtimeG = np.std(Time1)
meantimeF = np.mean(Time2)
sdtimeF = np.std(Time2)
meantimeD = np.mean(Time3)
sdtimeD = np.std(Time3)

print("GWIRE Mean Error:", mean_errorG)
print("GWIRE Standard Deviation of Error:", sd_errorG)
print("GWIRE mean Time:", meantimeG)
print("GWIRE std of Time:", sdtimeG)

print("DCOV Mean Error:", mean_errorD)
print("DCOV Standard Deviation of Error:", sd_errorD)
print("DCOV mean Time",meantimeD)
print("DCOV std of Time:", sdtimeD)

print("FOPG Mean Error:", mean_errorF)
print("FOPG Standard Deviation of Error:", sd_errorF)
print("FOPG mean Time:",meantimeF)
print("FOPG std of Time:",sdtimeF)
