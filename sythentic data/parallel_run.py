import time
import pickle
import numpy as np
from scipy.linalg import sqrtm, inv
from sklearn.covariance import GraphicalLassoCV
from multiprocessing import Pool
import functools

# Import your custom functions (make sure these are available in your Python path)
from functions.gram_matrix import gram_matrix2
from functions.lib_fun import gwire_cv
from functions.FOPG import FOPG
from functions.FD_SDR import FD_SDR

def direction_error_angle(b, beta_true):
    """Calculate the directional error between estimated and true coefficients."""
    proj_b = b @ inv(b.T @ b) @ b.T
    proj_true = beta_true @ inv(beta_true.T @ beta_true) @ beta_true.T
    return np.linalg.norm(proj_b - proj_true, 'fro')

def run_single_result(repeat_idx, n, p, q, metric, d_0, neigh, beta_true, verbose = False):
    """Run a single iteration of the simulation."""
    single_result = {}
    
    # Data Generation
    X = np.random.randn(n, p)
    mu = np.exp(X @ beta_true[:, 0])
    noise = 0.1 * np.random.randn(n)
    mu_y = mu + noise
    mu2 = np.clip(np.exp(X @ beta_true[:, 1]), 0.1, 10)
    y = (np.diag(mu2) @ np.random.randn(n, q)) + mu_y[:, np.newaxis]
    
    # Neighborhood estimation
    if not neigh:
        graphical_lasso_model = GraphicalLassoCV()
        graphical_lasso_model.fit(X)
        omega = graphical_lasso_model.precision_
        Nb = [np.nonzero(omega[j, :])[0].tolist() for j in range(p)]
    else:
        Nb = neigh
    
    # GWIRE method
    ygram = sqrtm(gram_matrix2(y, 10))
    start_time = time.time()
    beta_gwire, _ = gwire_cv(X, y, Nb, metric, d_0, fold=5)
    single_result['gwire_time'] = time.time() - start_time
    single_result['gwire_error'] = direction_error_angle(beta_gwire, beta_true)
    
    # FOPG method
    ygram2 = gram_matrix2(y, 1)
    start_time = time.time()
    beta_fopg = FOPG(X, ygram2, d_0)
    single_result['fopg_time'] = time.time() - start_time
    single_result['fopg_error'] = direction_error_angle(beta_fopg, beta_true)
    
    # FD-SDR method
    start_time = time.time()
    beta_fd_sdr, _, _ = FD_SDR(X.T, ygram, beta_fopg)
    single_result['fd_sdr_time'] = time.time() - start_time
    single_result['fd_sdr_error'] = direction_error_angle(beta_fd_sdr, beta_true)

    # Final summary
    if verbose:
        print("\nSimulation completed!")
        print(f"Average method times (seconds):")
        print(f"GWIRE: {single_result['gwire_time']:.3f}")
        print(f"FOPG: {single_result['fopg_time']:.3f}")
        print(f"FD-SDR: {single_result['fd_sdr_time']:.3f}")
        print(f"Average method errors:")
        print(f"GWIRE: {single_result['gwire_error']:.3f}")
        print(f"FOPG: {single_result['fopg_error']:.3f}")
        print(f"FD-SDR: {single_result['fd_sdr_error']:.3f}")
    
    return single_result

def run_simulation(num_repeats, n, p, q, metric="Wasserstein", d_0=1, neigh=False, num_processes=None):
    """
    Run simulation comparing GWIRE, FOPG, and FD-SDR methods in parallel.
    """
    # Set up true coefficients
    beta_1 = np.concatenate([np.array([1, 1]), np.zeros(p - 2)])
    beta_2 = np.concatenate([np.zeros(p - 2), np.array([1, 1])])
    beta_true = np.vstack([beta_1, beta_2]).T

    print(f"Starting parallel simulation with {num_repeats} repeats")
    print(f"Parameters: n={n}, p={p}, q={q}, d_0={d_0}")
    print(f"Using {num_processes if num_processes else 'all available'} processes")

    # Create partial function with fixed parameters
    partial_func = functools.partial(
        run_single_result,
        n=n, p=p, q=q, metric=metric, d_0=d_0, neigh=neigh, beta_true=beta_true
    )

    # Run simulations in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial_func, range(num_repeats))

    # Aggregate results
    aggregated = {
        'gwire_times': [r['gwire_time'] for r in results],
        'fopg_times': [r['fopg_time'] for r in results],
        'fd_sdr_times': [r['fd_sdr_time'] for r in results],
        'gwire_errors': [r['gwire_error'] for r in results],
        'fopg_errors': [r['fopg_error'] for r in results],
        'fd_sdr_errors': [r['fd_sdr_error'] for r in results]
    }

    # Final summary
    print("\nSimulation completed!")
    print(f"Average method times (seconds):")
    print(f"GWIRE: {np.mean(aggregated['gwire_times']):.3f} ± {np.std(aggregated['gwire_times']):.3f}")
    print(f"FOPG: {np.mean(aggregated['fopg_times']):.3f} ± {np.std(aggregated['fopg_times']):.3f}")
    print(f"FD-SDR: {np.mean(aggregated['fd_sdr_times']):.3f} ± {np.std(aggregated['fd_sdr_times']):.3f}")
    
    print(f"\nAverage direction errors:")
    print(f"GWIRE: {np.mean(aggregated['gwire_errors']):.4f} ± {np.std(aggregated['gwire_errors']):.4f}")
    print(f"FOPG: {np.mean(aggregated['fopg_errors']):.4f} ± {np.std(aggregated['fopg_errors']):.4f}")
    print(f"FD-SDR: {np.mean(aggregated['fd_sdr_errors']):.4f} ± {np.std(aggregated['fd_sdr_errors']):.4f}")

    return aggregated

def main():
    # Set parameters
    n = 200
    p = 10
    q = 100
    d_0 = 2
    num_repeats = 2
    
    # Run simulation
    start_time = time.time()
    results = run_simulation(
        num_repeats=num_repeats,
        n=n,
        p=p,
        q=q,
        d_0=d_0,
        num_processes=4  # You can adjust this based on your CPU cores
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Optional: Save results to file
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()