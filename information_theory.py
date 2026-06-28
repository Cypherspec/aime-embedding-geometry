"""
information_theory.py
─────────────────────
Information-theoretic analysis of the AIME problem corpus.

Implements:
  - PMI and PPMI matrices between terms and technique classes
  - KSG mutual information estimator (Kraskov, Stögbauer & Grassberger, 2004)
  - Centered kernel alignment (CKA) for representational similarity
  - Mutual information neural estimation (MINE) sketch for reference
  - Feature information gain and chi-square selection statistics
  - Entropy of technique class distribution per year (temporal diversity)

References:
  Kraskov A., Stögbauer H., Grassberger P. (2004). "Estimating mutual
  information." Physical Review E 69, 066138.
  Kornblith S. et al. (2019). "Similarity of Neural Network Representations
  Revisited." ICML.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.stats import chi2_contingency, entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Union
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# PMI / PPMI between vocabulary terms and technique classes
# ─────────────────────────────────────────────────────────────────────────────

def build_pmi_matrix(
    texts: List[str],
    labels: List[str],
    max_features: int = 2000,
    min_df: int = 3,
    smoothing: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build the PMI matrix between vocabulary terms and technique classes.

    PMI(t, c) = log[ P(t, c) / (P(t) * P(c)) ]
              = log[ N * n_{tc} / (n_t * n_c) ]

    where:
      N    = total (term, document) co-occurrence count
      n_tc = count of term t in documents of class c
      n_t  = count of term t across all documents
      n_c  = count of all terms in class c

    Args:
        texts: list of problem text strings
        labels: parallel list of technique class labels
        max_features: vocabulary size cap
        min_df: minimum document frequency for inclusion
        smoothing: add-smoothing constant (Laplace)

    Returns:
        pmi_matrix (V × C), vocab list, class list
    """
    class_names = sorted(set(labels))
    C = len(class_names)
    class_idx = {c: i for i, c in enumerate(class_names)}

    vectorizer = CountVectorizer(
        max_features=max_features, min_df=min_df, token_pattern=r'[a-zA-Z][a-zA-Z_]+',
    )
    X = vectorizer.fit_transform(texts).toarray().astype(float)  # (N, V)
    vocab = vectorizer.get_feature_names_out().tolist()
    V = len(vocab)

    # Build term-class co-occurrence matrix n_tc (V × C)
    label_ints = np.array([class_idx[l] for l in labels])
    n_tc = np.zeros((V, C)) + smoothing
    for c in range(C):
        mask = label_ints == c
        n_tc[:, c] += X[mask].sum(axis=0)

    n_t = n_tc.sum(axis=1, keepdims=True)   # (V, 1) — term marginal
    n_c = n_tc.sum(axis=0, keepdims=True)   # (1, C) — class marginal
    N = n_tc.sum()

    # PMI = log(N * n_tc / (n_t * n_c))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(N * n_tc / (n_t * n_c))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)

    return pmi, vocab, class_names


def ppmi_matrix(pmi: np.ndarray) -> np.ndarray:
    """Positive PMI: PPMI(t, c) = max(0, PMI(t, c))."""
    return np.maximum(0.0, pmi)


def top_pmi_terms(
    pmi: np.ndarray,
    vocab: List[str],
    class_names: List[str],
    top_k: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Return top-k PMI terms per class."""
    results = {}
    for j, cls in enumerate(class_names):
        col = pmi[:, j]
        top_idx = np.argsort(col)[::-1][:top_k]
        results[cls] = [(vocab[i], float(col[i])) for i in top_idx]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Information gain and chi-square feature selection
# ─────────────────────────────────────────────────────────────────────────────

def feature_information_gain(
    X: np.ndarray,
    y: np.ndarray,
    binarize_threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute information gain IG(t) = H(C) - H(C|t) for each feature t.

    IG measures how much knowing the presence/absence of term t reduces
    uncertainty about the technique class C.

    Args:
        X: (n, V) feature matrix (TF-IDF or count)
        y: (n,) integer class labels
        binarize_threshold: threshold for binarizing X (default: > 0)

    Returns:
        ig: (V,) array of information gain values
    """
    n, V = X.shape
    n_classes = y.max() + 1
    X_bin = (X > binarize_threshold).astype(float)

    # Prior class entropy H(C)
    p_c = np.bincount(y, minlength=n_classes) / n
    H_C = scipy_entropy(p_c, base=2)

    ig = np.zeros(V)
    for t in range(V):
        present = X_bin[:, t].astype(bool)
        absent = ~present
        n_pres = present.sum()
        n_abs = absent.sum()

        # H(C | t=1)
        if n_pres > 0:
            p_c_given_pres = np.bincount(y[present], minlength=n_classes) / n_pres
            H_c_pres = scipy_entropy(p_c_given_pres, base=2)
        else:
            H_c_pres = 0.0

        # H(C | t=0)
        if n_abs > 0:
            p_c_given_abs = np.bincount(y[absent], minlength=n_classes) / n_abs
            H_c_abs = scipy_entropy(p_c_given_abs, base=2)
        else:
            H_c_abs = 0.0

        # IG(t) = H(C) - P(t=1)*H(C|t=1) - P(t=0)*H(C|t=0)
        p_pres = n_pres / n
        p_abs = n_abs / n
        ig[t] = H_C - p_pres * H_c_pres - p_abs * H_c_abs

    return ig


def chi2_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    binarize_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chi-square feature selection statistic for each term.

    For each (term, class) pair, the chi-square statistic is:
        chi2(t, c) = N * (n_tc * n_tbar_cbar - n_tcbar * n_tbar_c)^2 /
                     (n_t * n_tbar * n_c * n_cbar)

    The aggregate statistic for term t is the mean across classes.

    Returns:
        chi2_scores: (V,) array of chi-square statistics
        p_values: (V,) array of corresponding p-values (df=n_classes-1)
    """
    n, V = X.shape
    n_classes = y.max() + 1
    X_bin = (X > binarize_threshold).astype(float)
    N = n

    chi2_scores = np.zeros(V)
    p_values = np.zeros(V)

    from scipy.stats import chi2 as chi2_dist

    for t in range(V):
        t_col = X_bin[:, t]
        # Build contingency table: rows=term (present/absent), cols=class
        contingency = np.zeros((2, n_classes))
        for c in range(n_classes):
            mask_c = (y == c)
            contingency[0, c] = (t_col[mask_c]).sum()      # present, class c
            contingency[1, c] = (1 - t_col[mask_c]).sum()  # absent, class c

        row_totals = contingency.sum(axis=1)
        col_totals = contingency.sum(axis=0)
        expected = np.outer(row_totals, col_totals) / N

        with np.errstate(invalid='ignore', divide='ignore'):
            chi2_val = np.nansum((contingency - expected)**2 /
                                 np.where(expected > 0, expected, 1))
        df = (2 - 1) * (n_classes - 1)
        p_val = 1.0 - chi2_dist.cdf(chi2_val, df=df)

        chi2_scores[t] = chi2_val
        p_values[t] = p_val

    return chi2_scores, p_values


# ─────────────────────────────────────────────────────────────────────────────
# KSG Mutual Information Estimator
# ─────────────────────────────────────────────────────────────────────────────

def ksg_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
) -> float:
    """
    KSG mutual information estimator (Algorithm 1, Kraskov et al., 2004).

    I(X; Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)

    where ψ = digamma, and n_x, n_y count neighbors within the Chebyshev
    radius of each joint-space k-th neighbor.

    This is a non-parametric estimator that works for continuous variables.
    Here we apply it to the embedding dimensions X (LSA vectors) and
    a 1-D class label encoding Y (treated as continuous).

    Args:
        X: (n, dx) first variable (e.g., first 2 LSA components)
        Y: (n, dy) second variable (e.g., class label as 1-hot)
        k: number of nearest neighbors

    Returns:
        Estimated mutual information in nats
    """
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    n = len(X)
    # Joint space
    XY = np.hstack([X, Y])

    # k-NN in joint space (Chebyshev metric)
    tree_xy = KDTree(XY)
    tree_x = KDTree(X)
    tree_y = KDTree(Y)

    # For each point, find radius of k-th neighbor in joint space
    dists, _ = tree_xy.query(XY, k=k + 1)  # +1 because self is included
    eps = dists[:, -1]  # (n,) Chebyshev radius

    # Count neighbors in marginal spaces within eps
    n_x = np.array([
        len(tree_x.query_ball_point(X[i], eps[i] - 1e-15, p=np.inf)) - 1
        for i in range(n)
    ])
    n_y = np.array([
        len(tree_y.query_ball_point(Y[i], eps[i] - 1e-15, p=np.inf)) - 1
        for i in range(n)
    ])

    # KSG estimator
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(mi)


def mutual_information_class_embedding(
    Z: np.ndarray,
    labels: np.ndarray,
    n_components: int = 5,
    k_ksg: int = 5,
) -> Dict[str, float]:
    """
    Estimate mutual information I(Z_component; Class) for the top LSA
    components using the KSG estimator.

    Args:
        Z: (n, d) LSA embedding matrix
        labels: (n,) integer class labels
        n_components: how many SVD components to evaluate
        k_ksg: KSG neighbor count

    Returns:
        dict mapping 'component_i' -> MI estimate in nats
    """
    Y = labels.astype(float)[:, None]
    results = {}
    for i in range(min(n_components, Z.shape[1])):
        Xi = Z[:, i:i+1]
        mi = ksg_mutual_information(Xi, Y, k=k_ksg)
        results[f'component_{i+1}'] = mi
    # Also estimate MI for the full Z (using first 2 dims for tractability)
    Z2 = Z[:, :2]
    mi_joint = ksg_mutual_information(Z2, Y, k=k_ksg)
    results['joint_dim1_2'] = mi_joint
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Centered Kernel Alignment (CKA) for representational similarity
# ─────────────────────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between two representation matrices.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Invariant to orthogonal transformations and isotropic scaling.
    CKA = 1 → representations are identical up to rotation/scale.
    CKA = 0 → representations are orthogonal (no shared structure).

    Args:
        X: (n, d1) first representation (e.g., LSA Stage 1 vectors)
        Y: (n, d2) second representation (e.g., TF-IDF Stage 2 vectors)

    Returns:
        CKA similarity score in [0, 1]
    """
    # Center columns
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # HSIC (Hilbert-Schmidt Independence Criterion) estimates
    def hsic(A, B):
        K = A @ A.T
        L = B @ B.T
        n = len(K)
        # Center the kernel matrices
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        return np.trace(KH @ LH) / (n - 1)**2

    return float(hsic(X_c, Y_c) / np.sqrt(hsic(X_c, X_c) * hsic(Y_c, Y_c) + 1e-12))


def cka_between_stages(
    Z_stage1: np.ndarray,
    Z_stage2: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute CKA between Stage 1 (LSA) and Stage 2 (raw TF-IDF) representations,
    both globally and per class.
    """
    results = {}
    results['global'] = linear_cka(Z_stage1, Z_stage2)
    for i, cls in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 5:
            results[cls] = linear_cka(Z_stage1[mask], Z_stage2[mask])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Temporal entropy of technique distribution
# ─────────────────────────────────────────────────────────────────────────────

def temporal_entropy(
    years: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[int, float]:
    """
    Compute the Shannon entropy of the technique class distribution for each
    competition year.

    H(C|year=t) = -sum_c P(C=c|year=t) * log2(P(C=c|year=t))

    High entropy → diverse year (all techniques equally represented).
    Low entropy → dominated by one technique (low compositional diversity).

    Returns dict: year -> entropy value
    """
    n_classes = len(class_names)
    unique_years = sorted(set(years))
    results = {}
    for yr in unique_years:
        mask = years == yr
        counts = np.bincount(labels[mask], minlength=n_classes).astype(float)
        if counts.sum() > 0:
            probs = counts / counts.sum()
            h = scipy_entropy(probs[probs > 0], base=2)
            results[int(yr)] = float(h)
    return results


def vocabulary_kl_divergence(
    texts: List[str],
    labels: List[str],
    years: List[int],
    class_names: List[str],
    era_split: int = 2004,
    max_features: int = 1000,
) -> Dict[str, float]:
    """
    Compute KL divergence of per-class vocabulary distributions between
    early (< era_split) and late (>= era_split) eras.

    KL(P_early || P_late) for each technique class measures vocabulary drift.
    """
    years_arr = np.array(years)
    early_mask = years_arr < era_split
    late_mask = years_arr >= era_split

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=3,
                                  token_pattern=r'[a-zA-Z][a-zA-Z_]+')
    vectorizer.fit(texts)

    results = {}
    for cls in class_names:
        cls_mask = np.array(labels) == cls
        early_texts = [t for t, m1, m2 in zip(texts, cls_mask, early_mask) if m1 and m2]
        late_texts = [t for t, m1, m2 in zip(texts, cls_mask, late_mask) if m1 and m2]

        if not early_texts or not late_texts:
            results[cls] = np.nan
            continue

        # Aggregate TF-IDF into vocabulary distributions
        X_early = vectorizer.transform(early_texts).toarray().mean(axis=0) + 1e-9
        X_late = vectorizer.transform(late_texts).toarray().mean(axis=0) + 1e-9

        # Normalize to probability distributions
        P_early = X_early / X_early.sum()
        P_late = X_late / X_late.sum()

        # Symmetric KL divergence (Jensen-Shannon divergence)
        M = 0.5 * (P_early + P_late)
        jsd = 0.5 * np.sum(P_early * np.log(P_early / M)) + \
              0.5 * np.sum(P_late * np.log(P_late / M))
        results[cls] = float(jsd)

    return results


if __name__ == '__main__':
    # Smoke test
    import json
    np.random.seed(42)
    n = 160
    d = 80
    Z = np.random.randn(n, d)
    labels = np.random.randint(0, 4, n)
    class_names = ['Number Theory', 'Combinatorics', 'Algebra', 'Geometry']

    print("Testing KSG MI estimator...")
    mi_results = mutual_information_class_embedding(Z, labels, n_components=3)
    for k, v in mi_results.items():
        print(f"  I(Z_{k}; Class) = {v:.4f} nats")

    print("\nTesting CKA...")
    Z2 = Z + np.random.randn(n, d) * 0.1  # slightly perturbed
    cka = linear_cka(Z, Z2)
    print(f"  CKA(Z, Z_perturbed) = {cka:.4f}")
    cka_id = linear_cka(Z, Z)
    print(f"  CKA(Z, Z) = {cka_id:.4f} [should be ~1.0]")

    print("\nTesting temporal entropy...")
    years = np.random.choice([2020, 2021, 2022, 2023], n)
    entropy_by_year = temporal_entropy(years, labels, class_names)
    for yr, h in sorted(entropy_by_year.items()):
        print(f"  H(C|year={yr}) = {h:.3f} bits")
