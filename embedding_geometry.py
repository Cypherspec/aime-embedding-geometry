"""
embedding_geometry.py
─────────────────────
Riemannian geometry analysis of the TF-IDF / LSA embedding space.

Implements:
  - Local sectional curvature estimation via the Brehmer–Cranmer estimator
  - Geodesic distance approximation on the unit hypersphere (cosine manifold)
  - Class-conditional covariance structure (anisotropy ratio)
  - Fisher information geometry of the softmax classifier
  - Within-class intrinsic dimensionality via TWO-NN estimator

Reference:
  Brehmer & Cranmer (2020). "Flows for simultaneous manifold learning
  and density estimation." NeurIPS.
  Facco et al. (2017). "Estimating the intrinsic dimension of datasets by a
  minimal neighborhood information." Scientific Reports.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigvalsh
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Geodesic distance on the unit sphere (cosine manifold)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Cosine distance d_cos(u, v) = 1 - cos(u, v) in [0, 2].
    X: (n, d), Y: (m, d) — both should be L2-normalized.
    Returns (n, m) distance matrix.
    """
    X_n = normalize(X, norm='l2')
    Y_n = normalize(Y, norm='l2')
    return 1.0 - X_n @ Y_n.T


def geodesic_distance_sphere(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Great-circle geodesic distance on S^{d-1}:
        d_geo(u, v) = arccos(u · v)  in [0, π].
    This is the true geodesic; cosine distance is a monotone proxy.
    """
    X_n = normalize(X, norm='l2')
    Y_n = normalize(Y, norm='l2')
    # Clip for numerical stability
    cos_sim = np.clip(X_n @ Y_n.T, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(cos_sim)


# ─────────────────────────────────────────────────────────────────────────────
# Sectional curvature estimation (Brehmer–Cranmer local estimator)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_sectional_curvature(
    Z: np.ndarray,
    n_samples: int = 200,
    k_neighbors: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Estimate local sectional curvature of the embedding manifold.

    Uses the metric comparison approach: on a manifold of constant curvature κ,
    the ratio of geodesic distances to Euclidean distances in a small ball
    determines κ. Positive κ → spherical geometry; negative κ → hyperbolic.

    For the unit-sphere TF-IDF space, we expect κ > 0 globally (since we
    embed on S^{d-1}), but local curvature in the LSA subspace may differ.

    Returns:
        dict with keys: mean_curvature, std_curvature, min_curvature,
                        max_curvature, fraction_positive
    """
    rng = np.random.default_rng(random_state)
    Z_n = normalize(Z, norm='l2')
    n = len(Z_n)
    indices = rng.choice(n, size=min(n_samples, n), replace=False)

    curvatures = []
    for idx in indices:
        # Find k nearest neighbors
        dists = cosine_distance(Z_n[[idx]], Z_n)[0]
        dists[idx] = np.inf
        nn_idx = np.argpartition(dists, k_neighbors)[:k_neighbors]

        # For each pair of neighbors, estimate local curvature via
        # the law of cosines on the manifold vs. Euclidean space
        neighbors = Z_n[nn_idx]
        center = Z_n[idx]

        local_curvs = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = np.arccos(np.clip(center @ neighbors[i], -1+1e-7, 1-1e-7))
                b = np.arccos(np.clip(center @ neighbors[j], -1+1e-7, 1-1e-7))
                c_geo = np.arccos(np.clip(neighbors[i] @ neighbors[j], -1+1e-7, 1-1e-7))

                # Euclidean distances for same points
                a_e = np.linalg.norm(center - neighbors[i])
                b_e = np.linalg.norm(center - neighbors[j])
                c_e = np.linalg.norm(neighbors[i] - neighbors[j])

                if a_e < 1e-9 or b_e < 1e-9:
                    continue

                # Curvature proxy: ratio of angle excess
                # On flat space: c^2 = a^2 + b^2 - 2ab*cos(C)
                # Excess/deficit relative to Euclidean gives curvature sign
                cos_C_geo = np.clip(
                    (np.cos(c_geo) - np.cos(a) * np.cos(b)) /
                    (np.sin(a) * np.sin(b) + 1e-9), -1, 1
                )
                cos_C_euc = np.clip(
                    (c_e**2 - a_e**2 - b_e**2) / (-2 * a_e * b_e + 1e-9),
                    -1, 1
                )
                # κ ≈ (C_geo - C_euc) / (area) — sign is what matters here
                kappa = np.arccos(cos_C_geo) - np.arccos(cos_C_euc)
                local_curvs.append(kappa)

        if local_curvs:
            curvatures.append(np.mean(local_curvs))

    curvatures = np.array(curvatures)
    return {
        'mean_curvature': float(np.mean(curvatures)),
        'std_curvature': float(np.std(curvatures)),
        'min_curvature': float(np.min(curvatures)),
        'max_curvature': float(np.max(curvatures)),
        'fraction_positive': float(np.mean(curvatures > 0)),
        'n_estimates': len(curvatures),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic dimensionality: TWO-NN estimator (Facco et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────

def twonn_intrinsic_dim(Z: np.ndarray, fraction: float = 0.9) -> Tuple[float, float]:
    """
    TWO-NN intrinsic dimensionality estimator.

    For each point, computes the ratio μ_i = r2_i / r1_i (second to first
    nearest-neighbor distances). Under the assumption that data lies on a
    d-dimensional manifold, μ follows a Pareto distribution with shape d.
    MLE of d is n / sum(log μ_i).

    Args:
        Z: embedding matrix (n, D)
        fraction: fraction of points to use (discard largest μ to reduce
                  boundary effects)

    Returns:
        (d_hat, d_ci_half_width) — point estimate and half-width of 95% CI
    """
    Z_n = normalize(Z, norm='l2')
    n = len(Z_n)
    dists = cosine_distance(Z_n, Z_n)
    np.fill_diagonal(dists, np.inf)

    mu_vals = []
    for i in range(n):
        row = dists[i]
        sorted_d = np.sort(row)
        r1, r2 = sorted_d[0], sorted_d[1]
        if r1 > 1e-9:
            mu_vals.append(r2 / r1)

    mu_vals = np.sort(mu_vals)
    # Keep only fraction to reduce boundary effects
    cutoff = int(len(mu_vals) * fraction)
    mu_vals = mu_vals[:cutoff]

    # MLE: d = n / sum(log mu)
    log_mu = np.log(mu_vals)
    d_hat = len(mu_vals) / np.sum(log_mu)

    # Fisher information CI for Pareto MLE
    d_se = d_hat / np.sqrt(len(mu_vals))
    d_ci = 1.96 * d_se

    return float(d_hat), float(d_ci)


def twonn_by_class(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Compute TWO-NN intrinsic dim separately for each class."""
    results = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        Z_cls = Z[mask]
        if len(Z_cls) < 10:
            results[name] = (np.nan, np.nan)
            continue
        d_hat, d_ci = twonn_intrinsic_dim(Z_cls)
        results[name] = (d_hat, d_ci)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Anisotropy: class-conditional covariance structure
# ─────────────────────────────────────────────────────────────────────────────

def class_covariance_anisotropy(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_components: int = 20,
) -> Dict[str, Dict]:
    """
    Compute the anisotropy ratio of each class's point cloud:
        anisotropy = λ_max / λ_min (ratio of largest to smallest eigenvalue
        of the within-class covariance matrix, restricted to top-k components)

    High anisotropy → elongated cluster along one direction (structure).
    Low anisotropy → isotropic / spherical cluster (no preferred direction).

    Also returns the participation ratio (effective dimensionality):
        PR = (sum λ_i)^2 / sum λ_i^2
    """
    results = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        Z_cls = Z[mask]
        if len(Z_cls) < 5:
            continue

        Z_centered = Z_cls - Z_cls.mean(axis=0)
        # Compute sample covariance restricted to top-k dims
        U, s, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        eigenvalues = s[:n_components]**2 / (len(Z_cls) - 1)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        anisotropy = eigenvalues[0] / eigenvalues[-1] if len(eigenvalues) > 1 else 1.0
        pr = (eigenvalues.sum()**2) / (eigenvalues**2).sum()

        results[name] = {
            'anisotropy': float(anisotropy),
            'participation_ratio': float(pr),
            'top_eigenvalue': float(eigenvalues[0]),
            'eigenvalue_spectrum': eigenvalues[:10].tolist(),
            'n_points': int(mask.sum()),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fisher Information geometry of the softmax classifier
# ─────────────────────────────────────────────────────────────────────────────

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fisher_information_metric(
    Z: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute the Fisher information matrix G(z) of the softmax model at each
    embedding point z.

    For a multinomial logistic model with parameters W (C × d) and bias b (C,),
    the FIM at a point z ∈ R^d is:

        G(z) = W^T diag(p(z)) W - (W^T p(z))(W^T p(z))^T

    where p(z) = softmax(Wz + b).  This is the Riemannian metric induced by
    the model's statistical manifold structure.

    Args:
        Z: (n, d) embedding matrix
        W: (C, d) logistic regression weight matrix
        b: (C,) bias vector

    Returns:
        G: (n, d, d) array of Fisher information matrices
    """
    n, d = Z.shape
    C = W.shape[0]

    logits = Z @ W.T + b[None, :]   # (n, C)
    P = softmax(logits)              # (n, C)

    G = np.zeros((n, d, d))
    for i in range(n):
        p = P[i]                     # (C,)
        # G = W^T diag(p) W - (W^T p)(W^T p)^T
        Wp = W.T @ p                 # (d,)
        G[i] = W.T @ (p[:, None] * W) - np.outer(Wp, Wp)

    return G


def fisher_geodesic_distance(
    z1: np.ndarray,
    z2: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    n_steps: int = 100,
) -> float:
    """
    Approximate the Fisher-Rao geodesic distance between two embedding points
    via linear interpolation and numerical integration:

        d_F(z1, z2) ≈ ∫_0^1 sqrt(v^T G(z(t)) v) dt

    where z(t) = (1-t)z1 + t*z2 (straight-line path) and v = z2 - z1.

    This approximates the true geodesic distance to first order; for small
    ||z2 - z1|| it is tight.
    """
    ts = np.linspace(0, 1, n_steps)
    v = z2 - z1
    integrand = []
    for t in ts:
        z_t = (1 - t) * z1 + t * z2
        G_t = fisher_information_metric(z_t[None], W, b)[0]
        speed = float(np.sqrt(np.clip(v @ G_t @ v, 0, None)))
        integrand.append(speed)
    return float(np.trapz(integrand, ts))


# ─────────────────────────────────────────────────────────────────────────────
# Within/Between distance ratio with bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def wb_ratio_bootstrap(
    Z: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute the within/between cosine distance ratio with a bootstrap 95% CI.

    W/B = mean(d(zi, zj) for same-class i,j) /
          mean(d(zi, zj) for different-class i,j)

    Returns: {observed, ci_lower, ci_upper, se}
    """
    rng = np.random.default_rng(random_state)
    Z_n = normalize(Z, norm='l2')
    D = cosine_distance(Z_n, Z_n)
    n = len(Z_n)

    def compute_wb(labs):
        same_mask = labs[:, None] == labs[None, :]
        np.fill_diagonal(same_mask, False)
        diff_mask = ~same_mask
        np.fill_diagonal(diff_mask, False)
        within = D[same_mask].mean()
        between = D[diff_mask].mean()
        return within / (between + 1e-12)

    observed = compute_wb(labels)

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_vals.append(compute_wb(labels[idx]))

    boot_vals = np.array(boot_vals)
    return {
        'observed': float(observed),
        'ci_lower': float(np.percentile(boot_vals, 2.5)),
        'ci_upper': float(np.percentile(boot_vals, 97.5)),
        'se': float(np.std(boot_vals)),
        'n_bootstrap': n_bootstrap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full geometry report
# ─────────────────────────────────────────────────────────────────────────────

def full_geometry_report(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    W: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict:
    """
    Run all geometry analyses and return a structured report.
    """
    print("Running geometry analysis...")
    report = {}

    print("  [1/5] Intrinsic dimensionality (TWO-NN)...")
    report['global_intrinsic_dim'] = twonn_intrinsic_dim(Z)
    report['classwise_intrinsic_dim'] = twonn_by_class(Z, labels, class_names)

    print("  [2/5] Class covariance anisotropy...")
    report['anisotropy'] = class_covariance_anisotropy(Z, labels, class_names)

    print("  [3/5] Within/between ratio with bootstrap CI...")
    report['wb_ratio'] = wb_ratio_bootstrap(Z, labels, random_state=random_state)

    print("  [4/5] Sectional curvature estimation...")
    report['curvature'] = estimate_sectional_curvature(Z, random_state=random_state)

    if W is not None and b is not None:
        print("  [5/5] Fisher information geometry...")
        # Sample 20 points for tractability
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(Z), size=min(20, len(Z)), replace=False)
        G_sample = fisher_information_metric(Z[sample_idx], W, b)
        # Trace of G = effective Fisher dimension
        traces = [float(np.trace(G_sample[i])) for i in range(len(sample_idx))]
        report['fisher'] = {
            'mean_trace': float(np.mean(traces)),
            'std_trace': float(np.std(traces)),
            'sample_size': len(sample_idx),
        }
    else:
        report['fisher'] = None

    return report


if __name__ == '__main__':
    # Smoke test with synthetic data
    import json
    np.random.seed(42)
    n_per_class = 40
    n_classes = 4
    d = 80

    # Simulate 4 class clusters in 80-d LSA space
    Z_list = []
    labels_list = []
    for c in range(n_classes):
        center = np.random.randn(d) * 0.3
        center /= np.linalg.norm(center)
        pts = center[None] + np.random.randn(n_per_class, d) * 0.4
        Z_list.append(pts)
        labels_list.extend([c] * n_per_class)

    Z = np.vstack(Z_list)
    labels = np.array(labels_list)
    class_names = ['Number Theory', 'Combinatorics', 'Algebra', 'Geometry']

    report = full_geometry_report(Z, labels, class_names, random_state=42)

    print("\n=== GEOMETRY REPORT ===")
    print(f"Global intrinsic dim: {report['global_intrinsic_dim'][0]:.2f} "
          f"± {report['global_intrinsic_dim'][1]:.2f}")
    print(f"Sectional curvature:  {report['curvature']['mean_curvature']:.4f} "
          f"± {report['curvature']['std_curvature']:.4f}")
    print(f"W/B ratio:            {report['wb_ratio']['observed']:.4f} "
          f"[{report['wb_ratio']['ci_lower']:.4f}, {report['wb_ratio']['ci_upper']:.4f}]")
    for cls in class_names:
        d_hat, d_ci = report['classwise_intrinsic_dim'][cls]
        anis = report['anisotropy'][cls]['anisotropy']
        pr = report['anisotropy'][cls]['participation_ratio']
        print(f"  {cls[:12]:12s}: d_int={d_hat:.1f}±{d_ci:.1f}, "
              f"anisotropy={anis:.2f}, PR={pr:.1f}")
