"""
Negative Binomial GLM implementation for differential expression analysis.

This module provides an implementation of negative binomial generalized
linear models using Iteratively Reweighted Least Squares (IRLS).

Classes:
    - NegativeBinomialGLM: Negative Binomial GLM implementation using IRLS

:Created: May 20, 2021
:Updated: March 28, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import numpy as np
from typing import Dict, Optional
from scipy.stats import nbinom


class NegativeBinomialGLM:
    """
    Negative Binomial GLM implementation using IRLS.

    This class implements iteratively reweighted least squares (IRLS) for
    fitting negative binomial generalized linear models.
    """

    def __init__(self, minmu: float = 0.5, tol: float = 1e-8, maxit: int = 100):
        """
        Initialize NegativeBinomialGLM.

        Parameters
        ----------
        minmu : float
            Minimum value for fitted means
        tol : float
            Convergence tolerance
        maxit : int
            Maximum number of iterations
        """
        self.minmu = minmu
        self.tol = tol
        self.maxit = maxit

    def _dnbinom_mu(self, x: np.ndarray, size: float, mu: np.ndarray, log: bool = False) -> np.ndarray:
        """
        Negative binomial probability mass function with mu parameterization.

        Parameters
        ----------
        x : ndarray
            Count values
        size : float
            Size parameter (1/dispersion)
        mu : ndarray
            Mean values
        log : bool, default=False
            Whether to return log probabilities

        Returns
        -------
        ndarray
            Probability mass function values
        """
        # Convert to prob parameterization
        prob = size / (size + mu)

        if log:
            return nbinom.logpmf(x, size, prob)
        else:
            return nbinom.pmf(x, size, prob)

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        nf: np.ndarray,
        alpha_hat: np.ndarray,
        beta_init: Optional[np.ndarray] = None,
        lambda_: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        use_weights: bool = False,
        use_qr: bool = True,
        contrast: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fit a negative-binomial GLM.

        Parameters
        ----------
        y : np.ndarray
            Count matrix, shape (N, M) where N=samples, M=genes
        x : np.ndarray
            Design matrix, shape (N, K) where K=coefficients
        nf : np.ndarray
            Size factors matrix, shape (N, M)
        alpha_hat : np.ndarray
            Dispersion estimates, shape (M,)
        beta_init : np.ndarray, optional
            Initial beta coefficients, shape (M, K)
        lambda_ : np.ndarray, optional
            Ridge penalty vector, shape (K,)
        weights : np.ndarray, optional
            Observation weights, shape (N, M)
        use_weights : bool, default=False
            Whether to use observation weights
        use_qr : bool, default=True
            Whether to use QR decomposition
        contrast : np.ndarray, optional
            Contrast vector, shape (K,)

        Returns
        -------
        dict
            Dictionary containing:
            - beta_mat: fitted coefficients (M, K)
            - beta_var_mat: variance of coefficients (M, K)
            - mu: fitted means (N, M)
            - iter: number of iterations per gene (M,)
            - hat_diagonals: hat matrix diagonals (N, M)
            - contrast_num: contrast numerator (M,)
            - contrast_denom: contrast denominator (M,)
            - deviance: deviance values (M,)
        """
        y_n, y_m = y.shape  # samples, genes
        x_p = x.shape[1]  # number of coefficients

        # Input validation
        assert alpha_hat.shape == (y_m,), f"alpha_hat shape {alpha_hat.shape} != ({y_m},)"
        assert nf.shape == y.shape, f"nf shape {nf.shape} != {y.shape}"
        assert x.shape[0] == y_n, f"x shape {x.shape} != ({y_n}, {x_p})"

        # Initialize beta if not provided
        if beta_init is None:
            # Use QR decomposition to get initial estimates
            if np.linalg.matrix_rank(x) == x_p:
                q, r = np.linalg.qr(x)
                y_log = np.log(y + 0.1)
                beta_init = np.linalg.solve(r, q.T @ y_log).T
            else:
                # Fallback to simple initialization
                beta_init = np.zeros((y_m, x_p))
                log_base_mean = np.log(np.mean(y, axis=0))
                beta_init[:, 0] = log_base_mean  # Intercept

        # Set default ridge penalty
        if lambda_ is None:
            lambda_ = np.repeat(1e-6, x_p)

        # Set default contrast
        if contrast is None:
            contrast = np.zeros(x_p)
            contrast[0] = 1.0

        # Initialize arrays
        beta_mat = beta_init.copy()
        beta_var_mat = np.zeros(beta_mat.shape)
        contrast_num = np.zeros(y_m)
        contrast_denom = np.zeros(y_m)
        large = 30.0
        iter_ = np.zeros(y_m)
        ridge = np.diag(lambda_)

        # Initial mu calculation
        mu_hat = np.maximum(nf * np.exp(x @ beta_mat.T), self.minmu)
        dev = np.zeros(y_m)
        dev_old = np.zeros(y_m)
        idx = np.repeat(True, y_m)

        # IRLS iterations
        for t in range(self.maxit):
            idx_n = np.sum(idx)
            if idx_n == 0:
                break

            iter_[idx] += 1

            mu_hat_idx = mu_hat[:, idx]

            # Calculate working weights
            if use_weights:
                w_vec = weights[:, idx] * mu_hat_idx / (1.0 + alpha_hat[idx] * mu_hat_idx)
            else:
                w_vec = mu_hat_idx / (1.0 + alpha_hat[idx] * mu_hat_idx)

            w_sqrt_vec = np.sqrt(w_vec)

            # Standard matrix inversion approach
            z = np.log(mu_hat_idx / nf[:, idx]) + (y[:, idx] - mu_hat_idx) / mu_hat_idx
            zwtx = (z * w_vec).T @ x
            beta_hat = np.linalg.solve(x.T @ (x * w_vec.T[:, :, None]) + ridge, zwtx[:, :, None]).squeeze(-1)

            # Update beta coefficients
            beta_mat[idx, :] = beta_hat

            # Check for large coefficients
            subidx = ~(np.sum(np.abs(beta_hat) > large, axis=1) > 0)
            newidx = idx.copy()
            newidx[idx] = subidx
            iter_[idx & ~newidx] = self.maxit

            idx = newidx

            # Update mu
            mu_hat[:, idx] = np.maximum(nf[:, idx] * np.exp(x @ beta_hat[subidx, :].T), self.minmu)

            # Calculate deviance
            dev[idx] = 0.0
            if use_weights:
                dev[idx] -= 2.0 * np.sum(
                    weights[:, idx] * self._dnbinom_mu(y[:, idx], 1.0 / alpha_hat[idx], mu_hat[:, idx], True),
                    axis=0,
                )
            else:
                dev[idx] -= 2.0 * np.sum(
                    self._dnbinom_mu(y[:, idx], 1.0 / alpha_hat[idx], mu_hat[:, idx], True),
                    axis=0,
                )

            # Convergence check
            conv_test = np.abs(dev - dev_old) / (np.abs(dev) + 0.1)
            nanidx = np.isnan(conv_test)
            iter_[idx & nanidx] = self.maxit
            idx &= ~nanidx
            if t > 0:
                idx &= ~(conv_test < self.tol)
            dev_old = dev.copy()

        # Recalculate weights for final variance calculation
        if use_weights:
            w_vec = weights * mu_hat / (1.0 + alpha_hat * mu_hat)
        else:
            w_vec = mu_hat / (1.0 + alpha_hat * mu_hat)

        w_sqrt_vec = np.sqrt(w_vec)
        x * w_sqrt_vec.T[:, :, None]
        np.linalg.inv(x.T @ (x * w_vec.T[:, :, None]) + ridge)

        # Calculate hat diagonals
        hat_diagonals = np.zeros(y.shape)

        # Calculate covariance matrix and extract variances
        beta_var_mat = np.zeros((y_m, x_p))
        beta_cov_mat = np.zeros((y_m, x_p, x_p))  # Full covariance matrix for each gene
        contrast_num = np.zeros(y_m)
        contrast_denom = np.zeros(y_m)

        for i in range(y_m):
            if not np.isnan(beta_mat[i, 0]):  # Only for valid genes
                # Calculate variance for this gene
                w_gene = w_vec[:, i]
                xtwx = x.T @ (x * w_gene[:, None]) + ridge
                xtwx_inv = np.linalg.inv(xtwx)
                beta_var_mat[i, :] = np.diagonal(xtwx_inv)
                beta_cov_mat[i, :, :] = xtwx_inv  # Store full covariance matrix

                # Calculate contrast statistics if contrast is provided
                if contrast is not None:
                    contrast_num[i] = contrast @ beta_mat[i, :]
                    contrast_denom[i] = np.sqrt(contrast.T @ xtwx_inv @ contrast)

        return {
            "beta_mat": beta_mat,
            "beta_var_mat": beta_var_mat,
            "beta_cov_mat": beta_cov_mat,  # Add full covariance matrix
            "mu": mu_hat,
            "iter": iter_,
            "hat_diagonals": hat_diagonals,
            "contrast_num": contrast_num,
            "contrast_denom": contrast_denom,
            "deviance": dev,
        }

    @staticmethod
    def fit_negative_binomial_glm(
        y: np.ndarray,
        x: np.ndarray,
        nf: np.ndarray,
        alpha_hat: np.ndarray,
        beta_init: Optional[np.ndarray] = None,
        lambda_: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        use_weights: bool = False,
        use_qr: bool = True,
        contrast: Optional[np.ndarray] = None,
        minmu: float = 0.5,
        tol: float = 1e-8,
        maxit: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Fit a negative-binomial GLM.

        This is a static method that creates a NegativeBinomialGLM instance
        and calls its fit method.

        Parameters
        ----------
        y : np.ndarray
            Count matrix, shape (N, M) where N=samples, M=genes
        x : np.ndarray
            Design matrix, shape (N, K) where K=coefficients
        nf : np.ndarray
            Size factors matrix, shape (N, M)
        alpha_hat : np.ndarray
            Dispersion estimates, shape (M,)
        beta_init : np.ndarray, optional
            Initial beta coefficients, shape (M, K)
        lambda_ : np.ndarray, optional
            Ridge penalty vector, shape (K,)
        weights : np.ndarray, optional
            Observation weights, shape (N, M)
        use_weights : bool, default=False
            Whether to use observation weights
        use_qr : bool, default=True
            Whether to use QR decomposition
        contrast : np.ndarray, optional
            Contrast vector, shape (K,)
        minmu : float, default=0.5
            Minimum value for fitted means
        tol : float, default=1e-8
            Convergence tolerance
        maxit : int, default=100
            Maximum number of iterations

        Returns
        -------
        dict
            Dictionary containing GLM results
        """
        glm = NegativeBinomialGLM(minmu=minmu, tol=tol, maxit=maxit)
        return glm.fit(
            y=y,
            x=x,
            nf=nf,
            alpha_hat=alpha_hat,
            beta_init=beta_init,
            lambda_=lambda_,
            weights=weights,
            use_weights=use_weights,
            use_qr=use_qr,
            contrast=contrast,
        )
