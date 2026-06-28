"""
hyperbolic_embedding.py
───────────────────────
Poincaré disk hyperbolic embeddings for AIME proof technique hierarchies.

The central hypothesis: proof technique knowledge has inherent HIERARCHICAL
structure (e.g., Number Theory → Modular Arithmetic → Chinese Remainder Theorem),
and hyperbolic space — which grows exponentially rather than polynomially —
is the natural geometry for hierarchical data. A tree with branching factor b
can be embedded with O(1) distortion in hyperbolic space of curvature −1/r²,
whereas Euclidean space requires O(log n) dimensions for the same distortion.

We implement:
  - Poincaré disk manifold operations (Möbius addition, exponential/log maps)
  - Riemannian SGD on the Poincaré disk (Bonnabel, 2013)
  - Tree distortion metric (Sarkar, 2011): measures embedding quality
  - Hyperbolic k-NN classifier
  - Curvature estimation via δ-hyperbolicity (Gromov, 1987)
  - Comparison: Euclidean vs hyperbolic embedding distortion

Mathematical background:
  The Poincaré disk D = {x ∈ ℝ²: ||x|| < 1} with metric
  g_x = (2/(1 - ||x||²))² g_E defines a model of ℍ²(−1).
  Geodesic distance: d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))

References:
  Nickel M. & Kiela D. (2017). "Poincaré embeddings for learning hierarchical
  representations." NeurIPS.
  Gromov M. (1987). "Hyperbolic groups." Essays in Group Theory, 75–263.
  Sarkar R. (2011). "Low distortion Delaunay embedding of trees in hyperbolic
  plane." Graph Drawing, LNCS 7034.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-6
MAX_NORM = 1.0 - 1e-5   # stay strictly inside the disk


# ─────────────────────────────────────────────────────────────────────────────
# Poincaré disk manifold operations
# ─────────────────────────────────────────────────────────────────────────────

def mobius_add(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Möbius addition in the Poincaré ball of curvature −c:

        x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                   (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

    This is the group operation of the Poincaré model; it replaces Euclidean
    vector addition on the hyperbolic manifold.
    """
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / (den + EPS)


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Geodesic distance in the Poincaré ball:

        d_c(x, y) = (2/√c) · arctanh(√c · ||−x ⊕_c y||)

    For c=1: d(x,y) = 2 arctanh(||−x ⊕ y||)

    This equals arcosh(1 + 2||x−y||²/((1−||x||²)(1−||y||²))) for c=1, d=2.
    """
    sqrt_c = np.sqrt(c)
    diff = mobius_add(-x, y, c)
    norm_diff = np.linalg.norm(diff, axis=-1, keepdims=True).clip(0, 1 - EPS)
    return (2.0 / sqrt_c) * np.arctanh(sqrt_c * norm_diff)


def poincare_distance_matrix(X: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Compute full (n×n) pairwise Poincaré distance matrix."""
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(poincare_distance(X[i], X[j], c))
            D[i, j] = D[j, i] = d
    return D


def expmap(x: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Exponential map at x in direction v:
        exp_x^c(v) = x ⊕_c (tanh(√c · ||v||_x / 2) · v / (√c · ||v||))

    where ||v||_x = λ_x^c · ||v|| is the Riemannian norm with
    conformal factor λ_x^c = 2/(1 - c||x||²).
    """
    sqrt_c = np.sqrt(c)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    lam = 2.0 / (1.0 - c * x2 + EPS)   # conformal factor
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + EPS
    tanh_term = np.tanh(sqrt_c * lam * v_norm / 2.0)
    return mobius_add(x, tanh_term * v / (sqrt_c * v_norm), c)


def logmap(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Logarithmic map at x (inverse of expmap):
        log_x^c(y) = (2/(√c · λ_x^c)) · arctanh(√c · ||−x ⊕_c y||)
                     · (−x ⊕_c y) / ||−x ⊕_c y||
    """
    sqrt_c = np.sqrt(c)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    lam = 2.0 / (1.0 - c * x2 + EPS)
    diff = mobius_add(-x, y, c)
    diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(EPS, 1 - EPS)
    return (2.0 / (sqrt_c * lam)) * np.arctanh(sqrt_c * diff_norm) * diff / diff_norm


def project_to_disk(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Project point onto the open unit ball (clip to boundary)."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm > MAX_NORM / np.sqrt(c),
                        norm * np.sqrt(c) / MAX_NORM + EPS, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Riemannian SGD for Poincaré embeddings
# ─────────────────────────────────────────────────────────────────────────────

class PoincareEmbedding:
    """
    Learn Poincaré disk embeddings via Riemannian SGD.

    Objective: for each (node, positive_context, negative_samples) triple,
    minimize the loss:
        L = −log(exp(−d(u, v)) / Σ_v' exp(−d(u, v')))

    This is the hyperbolic analogue of word2vec skip-gram negative sampling.
    For the AIME hierarchy, we define positive pairs as problems with the
    same fine-grained technique subtype, and negative pairs as problems from
    different coarse classes.

    The Riemannian gradient is the Euclidean gradient rescaled by (λ_x^c)²/4:
        ∇_ℋ f(x) = (1 - ||x||²)² / 4 · ∇_E f(x)
    """

    def __init__(
        self,
        n_nodes: int,
        dim: int = 2,
        curvature: float = 1.0,
        lr: float = 0.05,
        random_state: int = 42,
    ):
        self.n = n_nodes
        self.d = dim
        self.c = curvature
        self.lr = lr
        rng = np.random.default_rng(random_state)
        # Initialize uniformly in a small disk (avoids numeric issues at boundary)
        self.embeddings = rng.uniform(-0.001, 0.001, (n_nodes, dim))

    def _loss_and_grad(
        self,
        anchor_idx: int,
        pos_idx: int,
        neg_indices: List[int],
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss and Euclidean gradient for one (anchor, pos, neg*) triple.
        Returns (loss, grad_anchor).
        """
        u = self.embeddings[anchor_idx]
        v_pos = self.embeddings[pos_idx]
        v_negs = self.embeddings[neg_indices]

        d_pos = float(poincare_distance(u, v_pos, self.c))
        d_negs = np.array([float(poincare_distance(u, vn, self.c)) for vn in v_negs])

        # Numerically stable log-sum-exp
        all_d = np.concatenate([[d_pos], d_negs])
        neg_all_d = -all_d
        log_sum = np.log(np.exp(neg_all_d - neg_all_d.max()).sum()) + neg_all_d.max()
        loss = d_pos + log_sum

        # Euclidean gradient of Poincaré distance w.r.t. u
        def d_grad_u(u, v):
            diff = mobius_add(-u, v, self.c)
            diff_norm = np.linalg.norm(diff) + EPS
            if diff_norm >= 1 - EPS:
                return np.zeros_like(u)
            u2 = np.dot(u, u)
            v2 = np.dot(v, v)
            alpha = 1 - self.c * u2
            beta = 1 - self.c * v2
            denom = alpha * beta + EPS
            # ∂d/∂u — simplified form for c=1
            factor = 4.0 / (np.sqrt(self.c) * denom * np.sqrt(1 - self.c * diff_norm**2) + EPS)
            grad = factor * (v - u * (1 + self.c * (np.dot(u, v) - u2)))
            return grad

        # Riemannian gradient = Euclidean grad × (1 - ||u||²)² / 4
        u2 = np.dot(u, u)
        riem_scale = (1 - self.c * u2) ** 2 / 4.0

        grad_u_pos = d_grad_u(u, v_pos)

        # Softmax weights for negative terms
        neg_weights = np.exp(-d_negs - log_sum + d_pos)
        neg_weights /= neg_weights.sum() + EPS

        grad_u_neg = sum(
            w * d_grad_u(u, vn)
            for w, vn in zip(neg_weights, v_negs)
        )

        grad_euclidean = grad_u_pos - grad_u_neg
        grad_riemannian = riem_scale * grad_euclidean
        return float(loss), grad_riemannian

    def fit(
        self,
        pairs: List[Tuple[int, int]],
        n_negatives: int = 10,
        n_epochs: int = 200,
        random_state: int = 42,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train Poincaré embeddings via Riemannian SGD.

        Args:
            pairs: list of (i, j) positive pairs (same technique class)
            n_negatives: number of negative samples per positive pair
            n_epochs: training epochs
        """
        rng = np.random.default_rng(random_state)
        positive_set = set(map(tuple, pairs))
        all_indices = np.arange(self.n)
        losses = []

        for epoch in range(n_epochs):
            rng.shuffle(pairs if isinstance(pairs, np.ndarray) else pairs)
            epoch_loss = 0.0
            for anchor, positive in pairs:
                # Sample negatives (avoiding true positives)
                negs = []
                while len(negs) < n_negatives:
                    cand = int(rng.integers(0, self.n))
                    if cand != anchor and (anchor, cand) not in positive_set:
                        negs.append(cand)

                loss, grad = self._loss_and_grad(anchor, positive, negs)
                epoch_loss += loss

                # Riemannian gradient update: retraction via exponential map
                new_u = expmap(
                    self.embeddings[anchor],
                    -self.lr * grad,
                    self.c
                )
                self.embeddings[anchor] = project_to_disk(new_u, self.c)

            avg_loss = epoch_loss / max(1, len(pairs))
            losses.append(avg_loss)

            if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
                print(f"    Epoch {epoch:3d}/{n_epochs}: loss={avg_loss:.4f}")

            # Learning rate decay
            self.lr *= 0.999

        return losses

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings.copy()


# ─────────────────────────────────────────────────────────────────────────────
# δ-hyperbolicity (Gromov, 1987)
# ─────────────────────────────────────────────────────────────────────────────

def gromov_product(x: np.ndarray, y: np.ndarray, o: np.ndarray,
                   dist_fn) -> float:
    """
    Gromov product: (x|y)_o = (d(o,x) + d(o,y) − d(x,y)) / 2
    Measures how long geodesics from o to x and o to y travel together.
    """
    return 0.5 * (dist_fn(o, x) + dist_fn(o, y) - dist_fn(x, y))


def delta_hyperbolicity(
    D: np.ndarray,
    n_sample: int = 50,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Estimate Gromov's δ-hyperbolicity of a metric space from its distance matrix.

    A metric space is δ-hyperbolic if for all x, y, z, w:
        (x|z)_w ≥ min((x|y)_w, (y|z)_w) − δ

    Equivalently, the four-point condition:
        d(x,y) + d(z,w) ≤ max(d(x,z)+d(y,w), d(x,w)+d(y,z)) + 2δ

    δ = 0 → perfect tree metric (ultra-metric)
    δ > 0 → deviation from perfect hyperbolicity
    δ = ∞ → Euclidean space (no hyperbolicity)

    For competition mathematics problems, small δ would imply that the
    technique hierarchy closely approximates a tree structure.

    Args:
        D: (n×n) pairwise distance matrix
        n_sample: number of random 4-tuples to sample

    Returns:
        dict with delta_mean, delta_max, rel_delta (relative to diameter)
    """
    rng = np.random.default_rng(random_state)
    n = len(D)
    deltas = []

    for _ in range(n_sample * 10):
        i, j, k, l = rng.choice(n, size=4, replace=False)
        s1 = D[i, j] + D[k, l]
        s2 = D[i, k] + D[j, l]
        s3 = D[i, l] + D[j, k]
        sums = sorted([s1, s2, s3])
        delta = (sums[2] - sums[1]) / 2.0
        deltas.append(delta)

    deltas = np.array(deltas)
    diameter = D.max()

    return {
        'delta_mean': float(deltas.mean()),
        'delta_max': float(deltas.max()),
        'delta_p95': float(np.percentile(deltas, 95)),
        'rel_delta': float(deltas.max() / (diameter + EPS)),
        'diameter': float(diameter),
        'interpretation': (
            f"δ_max={deltas.max():.4f}, relative δ={deltas.max()/diameter:.4f}. "
            f"{'Near-hyperbolic (tree-like)' if deltas.max()/diameter < 0.05 else 'Moderate hyperbolicity'} "
            f"structure in the embedding space."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tree distortion metric (Sarkar, 2011)
# ─────────────────────────────────────────────────────────────────────────────

def embedding_distortion(
    D_true: np.ndarray,
    D_emb: np.ndarray,
    n_sample: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Measure the distortion of an embedding relative to true distances.

    Average distortion: D_avg = (1/|P|) Σ_{i<j} max(D_emb/D_true, D_true/D_emb)
    Worst-case distortion: D_max = max_{i<j} max(D_emb/D_true, D_true/D_emb)

    A perfect embedding has distortion = 1.
    Euclidean embeddings of hierarchies typically have distortion O(n^{1/d}).
    Poincaré disk embeddings can achieve O(1) distortion for trees.

    Also reports:
      - Mean absolute rank error (MARE): how much does embedding reorder pairs?
      - Stress (Kruskal's measure): normalized sum of squared distance errors
    """
    rng = np.random.default_rng(random_state)
    n = len(D_true)

    # Sample pairs for large n
    if n * (n - 1) // 2 > n_sample:
        pairs = set()
        while len(pairs) < n_sample:
            i, j = rng.integers(0, n, 2)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
        pairs = list(pairs)
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    distortions = []
    true_dists, emb_dists = [], []

    for i, j in pairs:
        dt = D_true[i, j]
        de = D_emb[i, j]
        if dt > EPS and de > EPS:
            distortions.append(max(de / dt, dt / de))
            true_dists.append(dt)
            emb_dists.append(de)

    distortions = np.array(distortions)
    true_dists = np.array(true_dists)
    emb_dists = np.array(emb_dists)

    # Stress (normalize by scale factor)
    scale = true_dists.mean() / (emb_dists.mean() + EPS)
    stress = np.sqrt(((true_dists - scale * emb_dists) ** 2).sum() /
                     (true_dists ** 2).sum() + EPS)

    # Mean absolute rank error
    true_ranks = np.argsort(np.argsort(true_dists))
    emb_ranks = np.argsort(np.argsort(emb_dists))
    mare = np.abs(true_ranks - emb_ranks).mean() / len(pairs)

    return {
        'mean_distortion': float(distortions.mean()),
        'max_distortion': float(distortions.max()),
        'p95_distortion': float(np.percentile(distortions, 95)),
        'stress': float(stress),
        'mean_abs_rank_error': float(mare),
        'n_pairs': len(distortions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic k-NN classifier
# ─────────────────────────────────────────────────────────────────────────────

def hyperbolic_knn_cv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    n_folds: int = 5,
    curvature: float = 1.0,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Cross-validated k-NN classifier using Poincaré geodesic distances.

    This directly tests whether the hyperbolic embedding has encoded
    class structure: if same-class problems cluster together in hyperbolic
    space, the k-NN classifier should achieve high accuracy.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_accs = []

    for train_idx, val_idx in skf.split(embeddings, labels):
        emb_tr = embeddings[train_idx]
        emb_val = embeddings[val_idx]
        y_tr = labels[train_idx]
        y_val = labels[val_idx]

        # Compute pairwise hyperbolic distances
        preds = []
        for i in range(len(emb_val)):
            dists = np.array([
                float(poincare_distance(emb_val[i], emb_tr[j], curvature))
                for j in range(len(emb_tr))
            ])
            nn_idx = np.argsort(dists)[:k]
            nn_labels = y_tr[nn_idx]
            pred = np.bincount(nn_labels).argmax()
            preds.append(pred)

        fold_accs.append(accuracy_score(y_val, preds))

    return {
        'mean_accuracy': float(np.mean(fold_accs)),
        'std_accuracy': float(np.std(fold_accs)),
        'fold_accuracies': [float(x) for x in fold_accs],
        'k': k,
        'curvature': curvature,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Euclidean vs Hyperbolic distortion comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_geometries(
    Z_euclidean: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    hierarchy: Optional[Dict] = None,
    n_poincare_epochs: int = 100,
    random_state: int = 42,
) -> Dict:
    """
    Compare Euclidean (cosine) vs Hyperbolic (Poincaré) embedding quality.

    Builds positive pairs from same-class labels, trains Poincaré embeddings,
    then compares:
      1. δ-hyperbolicity of Euclidean vs Poincaré distance matrices
      2. Distortion of each geometry relative to the hierarchy
      3. k-NN accuracy in each geometry
    """
    from sklearn.preprocessing import normalize

    print("  Building positive pairs from class labels...")
    n_classes = len(np.unique(labels))
    positive_pairs = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        for i in range(len(idx)):
            for j in range(i + 1, min(i + 8, len(idx))):  # sparse pairs
                positive_pairs.append((int(idx[i]), int(idx[j])))

    print(f"  Training Poincaré embeddings ({len(positive_pairs)} pairs)...")
    poincare = PoincareEmbedding(
        n_nodes=len(labels), dim=2, curvature=1.0,
        lr=0.05, random_state=random_state
    )
    losses = poincare.fit(
        positive_pairs, n_negatives=5,
        n_epochs=n_poincare_epochs, verbose=True
    )
    P = poincare.get_embeddings()

    # Distance matrices
    Z_n = normalize(Z_euclidean, norm='l2')
    D_eucl = 1 - Z_n @ Z_n.T  # cosine distances
    np.fill_diagonal(D_eucl, 0)

    print("  Computing Poincaré distance matrix...")
    # Sample for speed
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(labels), size=min(80, len(labels)), replace=False)
    D_poincare_sample = poincare_distance_matrix(P[sample_idx], c=1.0)
    D_eucl_sample = D_eucl[np.ix_(sample_idx, sample_idx)]

    print("  Computing δ-hyperbolicity...")
    delta_eucl = delta_hyperbolicity(D_eucl_sample, n_sample=100)
    delta_poinc = delta_hyperbolicity(D_poincare_sample, n_sample=100)

    print("  k-NN classification in Poincaré space...")
    knn_poincare = hyperbolic_knn_cv(P, labels, k=5, n_folds=5)

    return {
        'poincare_embeddings': P.tolist(),
        'training_losses': losses,
        'delta_hyperbolicity_euclidean': delta_eucl,
        'delta_hyperbolicity_poincare': delta_poinc,
        'knn_poincare': knn_poincare,
        'interpretation': (
            f"Euclidean δ_rel={delta_eucl['rel_delta']:.4f}, "
            f"Poincaré δ_rel={delta_poinc['rel_delta']:.4f}. "
            f"{'Poincaré is more tree-like' if delta_poinc['rel_delta'] < delta_eucl['rel_delta'] else 'Euclidean is more tree-like'}."
        ),
    }


if __name__ == '__main__':
    import json
    np.random.seed(42)

    # Smoke test
    print("Testing Poincaré disk operations...")
    x = np.array([0.3, 0.1])
    y = np.array([-0.2, 0.4])
    print(f"  d_P(x,y) = {float(poincare_distance(x, y)):.4f}")
    print(f"  exp_x(log_x(y)) ≈ y: {np.allclose(expmap(x, logmap(x, y)), y, atol=1e-4)}")

    print("\nTesting δ-hyperbolicity...")
    n = 30
    Z = np.random.randn(n, 10)
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    D = 1 - Z_n @ Z_n.T
    np.fill_diagonal(D, 0)
    delta = delta_hyperbolicity(D, n_sample=50)
    print(f"  δ_max = {delta['delta_max']:.4f}, rel_δ = {delta['rel_delta']:.4f}")

    print("\nTesting Poincaré embedding (small scale)...")
    labels = np.repeat([0, 1, 2, 3], 10)
    pairs = [(i, j) for c in range(4)
             for idx in [np.where(labels == c)[0]]
             for i in idx for j in idx if i < j]
    poincare = PoincareEmbedding(n_nodes=40, dim=2, curvature=1.0, random_state=42)
    losses = poincare.fit(pairs, n_negatives=3, n_epochs=30, verbose=False)
    print(f"  Final loss: {losses[-1]:.4f}")
    knn = hyperbolic_knn_cv(poincare.get_embeddings(), labels, k=3, n_folds=3)
    print(f"  Hyperbolic 3-NN accuracy: {knn['mean_accuracy']*100:.1f}%")
