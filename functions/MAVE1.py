import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import sqrtm
from scipy.linalg import orth

def MAVE1(X, y, init, option=None):
    verbosity = 0
    maxiter = 2000
    tolgradnorm = 1e-6
    d = 1  # default target dimension is d=1
    alpha = 1

    # Update parameters from options
    if option is not None:
        maxiter = option.get('maxiter', maxiter)
        tolgradnorm = option.get('tolgradnorm', tolgradnorm)
        verbosity = option.get('verbosity', verbosity)
        alpha = option.get('alpha', alpha)

    p, n = X.shape  # dimension and # of samples

    # Precompute necessary matrices
    KY = CompKernel(y) ** alpha
    B = KY - KY.mean(axis=1, keepdims=True) - KY.mean(axis=0, keepdims=True) + KY.mean()
    N = np.cov(X)
    N1 = sqrtm(N).real
   

    Z = np.linalg.inv(N1) @ X
   

    # Create a random starting point if no starting point is provided
    if init is None or init.size == 0:
        C = Projector(np.random.randn(p, d))
    else:
        C = orth(init)
        d = init.shape[1]

    if verbosity:
        print('   iter     cost val \t  \t   relative err \t grad. norm ')

    error = 1
    controller = 0
    fcur = dCovLoss(C, Z, B, alpha)
    start_time = time.time()
    lr_ini = 1
    object_vals = []

    for iter in range(maxiter):
        #print(iter)
        if iter == maxiter - 1:
            if verbosity:
                print('MAVE terminates: Achieved maximum iteration.')
            break

        grad = CompGradient(C, Z, B, alpha)
        fold = fcur
        Cold = C

        if controller:
            lr_ini *= 1.5
        else:
            lr_ini = 1

        lr = lr_ini

        # Line search
        sub_iter = 0
        normDsquare = np.trace(grad.T @ grad)

        C = Projector(Cold + lr * grad - lr * Cold @ (Cold.T @ grad))
        f_arm = dCovLoss(C, Z, B, alpha)

        while f_arm <= fcur + (1e-10) * lr * normDsquare:
            lr *= 0.5
            sub_iter += 1
            if lr < 1e-10:
                print(lr)
                break
            C = Projector(Cold + lr * grad - lr * Cold @ (Cold.T @ grad))
            f_arm = dCovLoss(C, Z, B, alpha)

        C = Projector(Cold + lr * grad - lr * Cold @ (Cold.T @ grad))
        
        fcur = f_arm

        error_prev = error
        error = abs((fcur - fold) / fcur)
        controller = error_prev < error

        gradnorm = np.linalg.norm(grad, 'fro')
        if verbosity:
            print(f'{iter:5d} \t  {fcur:.8e}  \t  {error:.8e} \t {gradnorm:.3f}')

        object_vals.append(fcur)

        if error < tolgradnorm:
            if verbosity:
                print(f'MAVE terminates: converged iteration: {iter:4d}')
            break

        if gradnorm < tolgradnorm:
            break

    elapsed_time = time.time() - start_time

    if verbosity:
        print(f'Total time is {elapsed_time:.6f} [s]')

    beta = np.linalg.inv(N1) @ C
    return beta, object_vals, elapsed_time

def CompKernel(x):
    
    # Compute the pairwise Euclidean distance matrix
    K = squareform(pdist(x.T, 'euclidean'))
    
    return K

def dCovLoss(C, Z, B, alpha):
    return np.mean(CompKernel(C.T @ Z) ** alpha * B)

def CompGradient(C, Z, B, alpha):
    n = B.shape[0]
    CZ = CompKernel(C.T @ Z)
    Idx = np.abs(CZ) > 1e-12
    CZinv = np.zeros_like(CZ)
    CZinv[Idx] = CZ[Idx] ** (alpha - 2)
    coef = alpha * CZinv * B
    G = Z @ (2 * (np.diag(coef.sum(axis=1)) - coef) @ Z.T @ C) / n**2
    return G

def Projector(C):
    # Compute the Gram matrix
    temp = C.T @ C
    
    # Compute the inverse square root of the Gram matrix
    inv_sqrt = np.linalg.inv(sqrtm(temp))
    
    # Project C to ensure orthonormal columns
    P = C @ np.real(inv_sqrt)
    
    return P


