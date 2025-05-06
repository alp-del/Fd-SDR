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
from numpy.linalg import inv
import time

# Parameters
n = 400
p = 20
q = 100
d_0 = 1  
sigma_y = 0.5
metric = "Wasserstein"
num_repeats = 100
neigh = False
np.random.seed(123)



beta_1 = np.concatenate([np.array([1, 1]), np.zeros(p - 2)])
beta_true = beta_1.reshape((p,1))

errors1 = np.zeros(num_repeats)
errors2 = np.zeros(num_repeats)
errors3 = np.zeros(num_repeats)
Time1 = np.zeros(num_repeats)
Time2 = np.zeros(num_repeats)
Time3 = np.zeros(num_repeats)

for i in range(num_repeats):
    # Data Generation
    X = np.random.randn(n, p)
    

    mu = np.exp(X @ beta_1)
    noise = 0.1 * np.random.randn(n)
    mu_y = mu + noise

    y = np.random.randn(n, q)
    y = y * sigma_y + mu_y[:, np.newaxis]
    Nb = []
    if not neigh:
        # Neighborhood is unknown
        from sklearn.covariance import GraphicalLassoCV
        graphical_lasso_model = GraphicalLassoCV()
        graphical_lasso_model.fit(X)
        omega = graphical_lasso_model.precision_
        np.where(np.sum(omega!=0, axis = 1) > 1)[0]
        
    for j in range(p):
        Ni = (np.nonzero(omega[j, :])[0]).tolist()
        Nb.append(Ni)

    
    

   
    ygram = gram_matrix2(y,10)
    xgram = gram_matrix2(X,10)
    start_time = time.time()
    beta_gwire, _ = gwire_cv(X, y, Nb, metric, d_0, fold=5)
    gwire_time = time.time() - start_time
    Time1[i] = gwire_time
    b1 = beta_gwire

    ygram2 = gram_matrix2(y,1)
    start_time = time.time()
    beta_fopg = FOPG(X,ygram2,d_0)
    fopg_time = time.time() - start_time
    Time2[i] = fopg_time
    b3 = beta_fopg
    initial = b3
    ygram = sqrtm(ygram)
    start_time = time.time()
    beta_dcov, _,_ = MAVE1(X.T,ygram,initial)
    mave1_time = time.time() - start_time
    Time3[i] = mave1_time
    b2 = beta_dcov
    

    error2 = linalg_norm(b2@inv(b2.T@b2)@b2.T-beta_true@inv(beta_true.T@beta_true)@beta_true.T)
    errors2[i] =error2

    error3 = linalg_norm(b3@inv(b3.T@b3)@b3.T-beta_true@inv(beta_true.T@beta_true)@beta_true.T)
    errors3[i] = error3

    
    
  
    #error1 = linalg_norm(b1@inv(b1.T@b1)@b1.T-beta_true@inv(beta_true.T@beta_true)@beta_true.T)
    #errors1[i] = error1
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