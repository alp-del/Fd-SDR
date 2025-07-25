# Fréchet Sufficient Dimension Reduction – Simulation Code

This repository contains the Python code used for the numerical experiments in our paper:

**Fréchet Sufficient Dimension Reduction for Metric Space-Valued Data via Distance Covariance**

The simulations reproduce the main results presented in the paper. All core methods and algorithmic implementations are located in the `functions/` directory.

---

## Included Methods

1. **Fd-SDR**  
   *Our proposed method.*  
   Implemented in [`functions/MAVE1.py`](functions/MAVE1.py)

2. **FOPG**  
   Implemented based on:  
   *Qi Zhang, Lingzhou Xue, and Bing Li. “Dimension Reduction for Fréchet Regression.”*  
   *Journal of the American Statistical Association*, 2023, pp. 1–15.  
   Code in [`functions/fopg.py`](functions/fopg.py)

3. **GWIRE**  
   Provided by the authors of:  
   *Jiaying Weng, Kai Tan, Cheng Wang, and Zhou Yu. “Sparse Fréchet Sufficient Dimension Reduction with Graphical Structure Among Predictors.”*  
   *arXiv preprint*, arXiv:2310.19114, 2023.  
   Code in [`functions/gwire.py`](functions/gwire.py)

---



