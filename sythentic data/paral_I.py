import time
import pickle
import numpy as np
import scipy
from scipy.linalg import sqrtm, inv
from sklearn.covariance import GraphicalLassoCV
from functions.gram_matrix import gram_matrix2
from functions.lib_fun import gwire_cv
from functions.FOPG import FOPG
from functions.FD_SDR import FD_SDR
from multiprocessing import Pool, cpu_count
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def direction_error_angle(b, beta_true):
    """Calculate the directional error between estimated and true coefficients."""
    proj_b = b @ inv(b.T @ b) @ b.T
    proj_true = beta_true @ inv(beta_true.T @ beta_true) @ beta_true.T
    return np.linalg.norm(proj_b - proj_true, 'fro')

def generate_X(n, p, mode_X = '(a)'):
    """Data generation for X"""
    if mode_X == '(a)':
        X = np.random.randn(n, p)
    elif mode_X == '(b)':
        X = np.zeros((n, p))
        # Generate X
        for sample in range(n):
            U = np.zeros(p)
            U[0] = np.random.normal(0, 1)
        
            for t in range(1, p):
                epsilon_t = np.random.normal(0, sigma_epsilon)
                U[t] = phi * U[t-1] + epsilon_t
        
            # Modify U[0] and U[1]
            U[0] = np.sin(U[0])
            U[1] = np.abs(U[1])
        
            X[sample, :] = U

    elif mode_X == '(c)':
        X = np.zeros((n, p))
        phi = 0.5
        sigma_epsilon = np.sqrt(1 - phi**2)
        for sample in range(n):
            U = np.zeros(p)
            U[0] = np.random.normal(0, 1)

            for t in range(1, p):
                epsilon_t = np.random.normal(0, sigma_epsilon)
                U[t] = phi * U[t-1] + epsilon_t

            x = scipy.stats.norm.cdf(U)
            X[sample, :] = x

    return X

def generate_Y(X, alpha = 0.2, q = 100, mode_y = '(1)', IF_GWIRE = False, neigh = None):
    n = X.shape[0]
    p = X.shape[1]

    beta_1 = np.concatenate([np.array([1, 1]), np.zeros(p - 2)])
    beta_2 = np.concatenate([np.zeros(p - 2), np.array([1, 1])])
    beta_3 = np.concatenate([[1, 2], np.zeros(p - 3), [2]])
    beta_4 = np.concatenate(([0, 0, 1, 2, 2], np.zeros(p - 5)))
    
    """Neighborhood estimation for GWIRE"""
    Nb = []
    if IF_GWIRE:
        if not neigh:
            # Neighborhood is unknown
            graphical_lasso_model = GraphicalLassoCV()
            graphical_lasso_model.fit(X)
            omega = graphical_lasso_model.precision_
            np.where(np.sum(omega!=0, axis = 1) > 1)[0]
            
        for j in range(p):
            Ni = (np.nonzero(omega[j, :])[0]).tolist()
            Nb.append(Ni)

    """Data generation for Y"""
    if mode_y == '(1)':
        # generate beta_true
        d_0 = 1
        beta_true = beta_1.reshape((p,1))
        
        # generate y
        sigma_y = 0.5
        mu = np.exp(X @ beta_1)
        noise = 0.1 * np.random.randn(n)
        mu_y = mu + noise
        y = np.random.randn(n, q)
        y = y * sigma_y + mu_y[:, np.newaxis]
    elif mode_y == '(2)':
        # generate beta_true
        d_0 = 2
        beta_true = np.vstack([beta_1,beta_2]).T

        # generate y
        mu = np.exp(X @ beta_1)
        noise = 0.1 * np.random.randn(n)
        mu_y = mu + noise
        mu2 = np.exp(X @ beta_2)
        mu2 = np.clip(mu2, 0.1, 10)
        sigma_matrix = np.diag(mu2)
        y = np.random.randn(n, q)
        y = (sigma_matrix @ y) + mu_y[:, np.newaxis]
    elif mode_y == '(3)':
        # generate beta_true
        d_0 = 2
        beta_true = np.vstack([beta_3, beta_4]).T

        # generate y
        nu = 0.5
        mu = 3 * np.dot(X, beta_3)
        mu_y = mu + 0.5 * np.random.randn(n)
        sigma_y = np.zeros(n)
        for sample_idx in range(n):
            x = X[sample_idx, :]
            gamma_alpha = (2 + 2 * np.dot(x, beta_4))**2 / nu
            gamma_beta = nu / (2 + 2 * np.dot(x, beta_4))
            sigma_y[sample_idx] = scipy.stats.gamma.rvs(gamma_alpha, scale=gamma_beta)

        # Adjust sigma_y values
        sigma_y[sigma_y < 0.1] = 0.1
        sigma_y[sigma_y > 10] = 10

        # Generate y matrix
        sigma_matrix = alpha * np.diag(sigma_y)
        y = np.random.randn(n, q)
        y = np.dot(sigma_matrix, y) + mu_y[:, np.newaxis]
    elif mode_y == '(4)':
        # generate beta_true
        d_0 = 2
        beta_true = np.vstack([beta_3, beta_4]).T

        # generate y
        nu = 0.5
        mu_y = np.zeros(n)
        sigma_y = np.zeros(n)

        for i in range(n):
            x = X[i, :]
            gamma_alpha = (2 + 2 * np.dot(x, beta_4))**2 / nu
            gamma_beta = nu / (2 + 2 * np.dot(x, beta_4))
            sigma_y[i] = scipy.stats.gamma.rvs(gamma_alpha, scale=gamma_beta)
        
            mu = 3 * np.sin(np.dot(x, beta_3))
            mu_y[i] = np.random.normal(mu, 0.5**2)

        # Adjust sigma_y values
        sigma_y = np.clip(sigma_y, 0.1, 10)

        # Generate y matrix
        sigma_matrix = alpha * np.diag(sigma_y) #alpha =0.2/0.4
        y = np.random.randn(n, q)
        y = np.dot(sigma_matrix, y) + mu_y[:, np.newaxis]

    return {'X': X, 'y': y, 'beta_true': beta_true, 'Nb': Nb, 'd_0': d_0}

def run_single_iteration(iter_num, config):
    """Run a single iteration of the simulation (for parallel processing)"""
    np.random.seed(123 + iter_num)  # Different seed for each iteration
    
    n = config['n']
    p = config['p']
    q = config['q']
    alpha = config['alpha']
    mode_X = config['mode_X']
    mode_y = config['mode_y']
    IF_GWIRE = config['IF_GWIRE']
    neigh = config['neigh']
    metric = config['metric']
    verbose = config['verbose']
    
    result = {
        'gwire_error': None,
        'fopg_error': None,
        'fd_sdr_error': None
    }
    
    if verbose:
        print(f"Starting iteration {iter_num + 1}")
    
    X = generate_X(n, p, mode_X)
    DATA_XY = generate_Y(X, alpha, q, mode_y, IF_GWIRE, neigh)
    y = DATA_XY['y']
    beta_true = DATA_XY['beta_true']
    Nb = DATA_XY['Nb']
    d_0 = DATA_XY['d_0']
    
    ygram = sqrtm(gram_matrix2(y, 10))
    ygram2 = gram_matrix2(y, 1)

    # GWIRE method
    if IF_GWIRE:
        beta_gwire, _ = gwire_cv(X, y, Nb, metric, d_0, fold=5)
        result['gwire_error'] = direction_error_angle(beta_gwire, beta_true)
    
    # FOPG method
    beta_fopg = FOPG(X, ygram2, d_0)
    result['fopg_error'] = direction_error_angle(beta_fopg, beta_true)

    # FD-SDR method
    beta_fd_sdr, _, _ = FD_SDR(X.T, ygram, beta_fopg)
    result['fd_sdr_error'] = direction_error_angle(beta_fd_sdr, beta_true)
    
    if verbose:
        print(f"Completed iteration {iter_num + 1}")
    
    return result

def run_simulation(config):
    num_repeats = config['num_repeats']
    verbose = config['verbose']
    
    # Initialize result storage
    results = {
        'gwire_errors': [],
        'fd_sdr_errors': [],
        'fopg_errors': []
    }

    total_start = time.time()

    """Run simulation"""
    if verbose:
        print("\n" + "="*60)
        print(f"Starting Simulation".center(60))
        print("="*60)
        print(f"Configuration:")
        print(f"- Repeats: {num_repeats}")
        print(f"- Dimensions: n={config['n']}, p={config['p']}, q={config['q']}")
        print(f"- X mode: {config['mode_X']}, Y mode: {config['mode_y']}")
        print(f"- Metric: {config['metric']}")
        print(f"- GWIRE: {'Enabled' if config['IF_GWIRE'] else 'Disabled'}")
        print(f"- Using {cpu_count()} CPU cores")
        print("="*60 + "\n")

    # Create a pool of workers
    with Pool(processes=cpu_count()) as pool:
        # Prepare arguments for each iteration
        args = [(i, config) for i in range(num_repeats)]
        
        # Run iterations in parallel
        iter_results = pool.starmap(run_single_iteration, args)
    
    # Aggregate results
    for res in iter_results:
        if config['IF_GWIRE']:
            results['gwire_errors'].append(res['gwire_error'])
        
        results['fopg_errors'].append(res['fopg_error'])
        results['fd_sdr_errors'].append(res['fd_sdr_error'])

    total_time = time.time() - total_start

    if verbose:
        print("\n" + "="*60)
        print(" Simulation Summary ".center(60, '='))
        print("="*60)
        print(f"Total simulation time: {total_time:.2f} seconds")
        print(f"Average per iteration: {total_time/num_repeats:.2f} seconds")
        print("-"*60)
        print(" Average Performance ".center(60))
        print("-"*60)
        
        if config['IF_GWIRE']:
            print(f"{'':<10} Error = {np.mean(results['gwire_errors']):.4f} ± {np.std(results['gwire_errors']):.4f}")
            print("-"*60)
        
        print(f"{'':<10} Error = {np.mean(results['fopg_errors']):.4f} ± {np.std(results['fopg_errors']):.4f}")
        print("-"*60)
        
        print(f"{'':<10} Error = {np.mean(results['fd_sdr_errors']):.4f} ± {np.std(results['fd_sdr_errors']):.4f}")
        print("="*60 + "\n")

    return results

def print_summary(results, config=None):
    """
    Print a well-formatted summary of the simulation results.
    """
    # Header
    print("\n" + "="*80)
    print(" SIMULATION SUMMARY ".center(80, '='))
    print("="*80)
    
    # Print configuration if provided
    if config:
        print("\nCONFIGURATION:")
        for key, value in config.items():
            print(f"- {key}: {value}")
        print("-"*80)
    
    # Calculate statistics
    stats = {}
    methods = []
    
    if 'gwire_errors' in results and len(results['gwire_errors']) > 0:
        methods.append('GWIRE')
        stats['GWIRE'] = {
            'error_mean': np.mean(results['gwire_errors']),
            'error_std': np.std(results['gwire_errors'])
        }
    
    methods.extend(['FOPG', 'FD-SDR'])
    
    stats['FOPG'] = {
        'error_mean': np.mean(results['fopg_errors']),
        'error_std': np.std(results['fopg_errors'])
    }
    
    stats['FD-SDR'] = {
        'error_mean': np.mean(results['fd_sdr_errors']),
        'error_std': np.std(results['fd_sdr_errors'])
    }
    
    # Print performance table
    print("\nPERFORMANCE METRICS:")
    print("-"*80)
    print(f"{'Method':<10}{'Time (mean ± std)':<30}{'Error (mean ± std)':<30}")
    print("-"*80)
    
    for method in methods:
        m = stats[method]
        time_str = f"{m['time_mean']:.4f}s ± {m['time_std']:.4f}"
        error_str = f"{m['error_mean']:.4f} ± {m['error_std']:.4f}"
        print(f"{method:<10}{time_str:<30}{error_str:<30}")
    
    # Footer
    print("="*80 + "\n")

if __name__ == '__main__':
    np.random.seed(123)

    config = {
        'n': 200,
        'p': 10,
        'q': 100,
        'alpha': 0.2,
        'mode_X': '(c)',
        'mode_y': '(3)',
        'num_repeats': 100,
        'IF_GWIRE': False,
        'neigh': None,
        'metric': 'Wasserstein',
        'verbose': True
    }

    results_200_10 = run_simulation(config=config)
    print_summary(results_200_10, config)