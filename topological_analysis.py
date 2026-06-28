"""
topological_analysis.py
───────────────────────
Topological Data Analysis (TDA) of AIME embedding space.

Implements:
  - Persistent homology via Vietoris-Rips filtration (ripser)
  - Betti numbers β0, β1, β2 as topological invariants of technique classes
  - Persistence diagrams and bottleneck/Wasserstein distances between classes
  - Mapper algorithm: low-dimensional topological skeleton of problem space
  - Persistent entropy: information content of persistence diagrams
  - Class-conditional topology: does each technique class have distinct topology?

Mathematical background:
  Given a point cloud X ⊂ ℝ^d, the Vietoris-Rips complex VR(X, ε) at scale ε
  is the simplicial complex with:
    - 0-simplices (vertices): all points in X
    - 1-simplices (edges): all pairs {x,y} with d(x,y) ≤ ε
    - k-simplices: all (k+1)-cliques in the 1-skeleton

  Varying ε from 0 to ∞ gives a filtration. Persistent homology tracks the
  birth and death of topological features:
    β0 = connected components (born at ε=0, die when components merge)
    β1 = 1-dimensional holes / loops (born when a cycle forms, die when filled)
    β2 = 2-dimensional voids (rare in low-dimensional embeddings)

  A persistence diagram D is a multiset of (birth, death) pairs for each
  topological feature. The persistence (lifetime) of a feature is death − birth.
  Long-lived features are topologically significant; short-lived ones are noise.

References:
  Edelsbrunner H. & Harer J. (2010). Computational Topology. AMS.
  Carlsson G. (2009). "Topology and data." Bulletin of the AMS, 46(2): 255–308.
  Chazal F. & Michel B. (2021). "An introduction to topological data analysis."
    Frontiers in Artificial Intelligence.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from ripser import ripser
    from persim import plot_diagrams, wasserstein, bottleneck
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("WARNING: ripser/persim not installed. Using fallback implementations.")


# ─────────────────────────────────────────────────────────────────────────────
# Persistent homology computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_persistence(
    Z: np.ndarray,
    max_dim: int = 2,
    metric: str = 'cosine',
    max_edge_length: float = 2.0,
    n_subsample: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """
    Compute persistent homology of a point cloud.

    Args:
        Z: (n, d) point cloud (embedding matrix)
        max_dim: maximum homology dimension (1=loops, 2=voids)
        metric: distance metric ('cosine', 'euclidean')
        max_edge_length: maximum filtration value (2.0 = full cosine range)
        n_subsample: subsample to this many points for speed (None=all)

    Returns:
        dict with:
          diagrams: list of (birth, death) arrays per dimension
          betti_numbers: estimated Betti numbers at mid-filtration
          persistent_entropy: Shannon entropy of persistence lifetimes
          n_significant: features with persistence > median + 2*std
    """
    rng = np.random.default_rng(random_state)

    if n_subsample is not None and len(Z) > n_subsample:
        idx = rng.choice(len(Z), size=n_subsample, replace=False)
        Z_use = Z[idx]
    else:
        Z_use = Z

    Z_n = normalize(Z_use, norm='l2') if metric == 'cosine' else Z_use

    if HAS_RIPSER:
        # Full persistent homology via ripser
        result = ripser(Z_n, maxdim=max_dim, metric=metric,
                        thresh=max_edge_length)
        diagrams = result['dgms']
    else:
        # Fallback: compute only H0 via single-linkage dendrogram
        diagrams = _fallback_h0(Z_n, metric)
        diagrams = [diagrams] + [np.array([]).reshape(0, 2)] * max_dim

    # Process diagrams
    processed = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            processed[f'H{dim}'] = {
                'n_features': 0,
                'mean_persistence': 0.0,
                'max_persistence': 0.0,
                'persistent_entropy': 0.0,
                'n_significant': 0,
                'diagram': [],
            }
            continue

        # Remove infinite death values (use max_edge_length as proxy)
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]
        if len(dgm_finite) == 0:
            dgm_finite = dgm.copy()
            dgm_finite[~finite_mask, 1] = max_edge_length

        persistences = dgm_finite[:, 1] - dgm_finite[:, 0]
        persistences = persistences[persistences > 1e-6]

        if len(persistences) == 0:
            processed[f'H{dim}'] = {'n_features': 0, 'mean_persistence': 0.0,
                                     'max_persistence': 0.0, 'persistent_entropy': 0.0,
                                     'n_significant': 0, 'diagram': []}
            continue

        # Persistent entropy: H = -Σ (p_i log p_i) where p_i = L_i / L_total
        L_total = persistences.sum()
        p = persistences / L_total
        p = p[p > 0]
        pers_entropy = float(-np.sum(p * np.log(p)))

        # Significant features: persistence > mean + 1 std
        threshold = persistences.mean() + persistences.std()
        n_significant = int((persistences > threshold).sum())

        processed[f'H{dim}'] = {
            'n_features': len(persistences),
            'mean_persistence': float(persistences.mean()),
            'max_persistence': float(persistences.max()),
            'std_persistence': float(persistences.std()),
            'persistent_entropy': pers_entropy,
            'n_significant': n_significant,
            'diagram': dgm_finite.tolist(),
        }

    return {
        'homology': processed,
        'n_points': len(Z_use),
        'metric': metric,
        'max_dim': max_dim,
    }


def _fallback_h0(Z: np.ndarray, metric: str) -> np.ndarray:
    """
    Fallback H0 computation via single-linkage when ripser unavailable.
    Returns (n, 2) array of (birth=0, death=merge_distance) for each component.
    """
    from scipy.cluster.hierarchy import linkage
    D = cdist(Z, Z, metric=metric)
    n = len(Z)
    # Use single-linkage dendrogram: merge distances = H0 death values
    try:
        Z_link = linkage(D[np.triu_indices(n, k=1)], method='single')
        # Each merge kills one component; n-1 merges for n points
        deaths = np.sort(Z_link[:, 2])
        births = np.zeros(len(deaths))
        # Add the last component (never dies → infinite)
        dgm = np.column_stack([births, deaths])
        dgm = np.vstack([dgm, [0, np.inf]])
    except Exception:
        dgm = np.array([[0, np.inf]])
    return dgm


# ─────────────────────────────────────────────────────────────────────────────
# Class-conditional topology
# ─────────────────────────────────────────────────────────────────────────────

def class_persistence(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    max_dim: int = 1,
    metric: str = 'cosine',
    n_subsample: int = 40,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """
    Compute persistent homology separately for each technique class.

    This tests whether different technique classes have distinct topological
    signatures — different Betti numbers, persistence entropies, or numbers
    of significant topological features.

    A class with high β1 (many H1 loops) has a more "annular" or ring-shaped
    point cloud, suggesting internal subgroup structure.
    A class with high persistent entropy has a more complex, multi-scale topology.
    """
    results = {}
    for c_idx, cls in enumerate(class_names):
        mask = labels == c_idx
        Z_cls = Z[mask]
        print(f"    Computing H0, H1 for {cls} (n={mask.sum()})...")
        result = compute_persistence(
            Z_cls, max_dim=max_dim, metric=metric,
            n_subsample=min(n_subsample, mask.sum()),
            random_state=random_state,
        )
        results[cls] = result

    return results


def persistence_diagram_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    metric: str = 'bottleneck',
) -> float:
    """
    Compute distance between two persistence diagrams.

    Bottleneck distance: d_B(D1, D2) = inf_γ sup_{x∈D1} ||x - γ(x)||_∞
    This is the most common diagram distance; it measures the cost of the
    optimal matching between diagram points.

    Wasserstein distance (p=1): sum of matched distances (less extreme than bottleneck).
    """
    if not HAS_RIPSER:
        # Fallback: compare mean persistence
        if len(dgm1) == 0 or len(dgm2) == 0:
            return float('inf')
        p1 = (dgm1[:, 1] - dgm1[:, 0]).mean() if len(dgm1) > 0 else 0
        p2 = (dgm2[:, 1] - dgm2[:, 0]).mean() if len(dgm2) > 0 else 0
        return float(abs(p1 - p2))

    try:
        if metric == 'bottleneck':
            return float(bottleneck(dgm1, dgm2))
        else:
            return float(wasserstein(dgm1, dgm2))
    except Exception:
        return 0.0


def topology_distance_matrix(
    class_diagrams: Dict[str, Dict],
    class_names: List[str],
    homology_dim: int = 1,
    metric: str = 'bottleneck',
) -> np.ndarray:
    """
    Build (C×C) matrix of topological distances between class persistence diagrams.

    Small off-diagonal values → classes have similar topological structure.
    Large values → topologically distinct classes.
    """
    n = len(class_names)
    D = np.zeros((n, n))

    for i, cls_i in enumerate(class_names):
        for j, cls_j in enumerate(class_names):
            if i >= j:
                continue
            h_key = f'H{homology_dim}'
            dgm_i = np.array(class_diagrams[cls_i].get('homology', {}).get(
                h_key, {}).get('diagram', []))
            dgm_j = np.array(class_diagrams[cls_j].get('homology', {}).get(
                h_key, {}).get('diagram', []))

            if len(dgm_i) == 0 or len(dgm_j) == 0:
                d = 0.0
            else:
                d = persistence_diagram_distance(dgm_i, dgm_j, metric)

            D[i, j] = D[j, i] = d

    return D


# ─────────────────────────────────────────────────────────────────────────────
# Permutation test for topological class separation
# ─────────────────────────────────────────────────────────────────────────────

def topological_permutation_test(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_permutations: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Permutation test for whether class-conditional persistent entropy
    differs significantly from chance.

    Observed statistic: variance of per-class persistent entropy values.
    High variance → classes have topologically distinct structures.
    Under H0 (labels random), this variance should be near zero.
    """
    rng = np.random.default_rng(random_state)

    def entropy_variance(labs):
        entropies = []
        for c_idx in range(len(class_names)):
            mask = labs == c_idx
            if mask.sum() < 5:
                continue
            Z_cls = Z[mask][:min(25, mask.sum())]
            result = compute_persistence(Z_cls, max_dim=1, metric='cosine',
                                         n_subsample=20, random_state=42)
            h1 = result['homology'].get('H1', {})
            entropies.append(h1.get('persistent_entropy', 0.0))
        return float(np.var(entropies)) if len(entropies) > 1 else 0.0

    print("    Computing observed topological statistic...")
    observed = entropy_variance(labels)

    print(f"    Running {n_permutations} permutations...")
    null_dist = []
    for i in range(n_permutations):
        perm = rng.permutation(labels)
        null_dist.append(entropy_variance(perm))
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{n_permutations} done")

    null_arr = np.array(null_dist)
    p_value = float((null_arr >= observed).mean())
    z_score = float((observed - null_arr.mean()) / (null_arr.std() + 1e-9))

    return {
        'observed': observed,
        'null_mean': float(null_arr.mean()),
        'null_std': float(null_arr.std()),
        'z_score': z_score,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mapper algorithm (topological skeleton)
# ─────────────────────────────────────────────────────────────────────────────

class MapperGraph:
    """
    Mapper algorithm (Singh, Mémoli & Carlsson, 2007).

    Produces a graph (nerve of a cover) that captures the topology of
    high-dimensional data projected through a filter function.

    For AIME problems:
      Filter: first LSA component (primary technique axis)
      Cover: overlapping intervals on [min_filter, max_filter]
      Clustering: k-means within each interval (DBSCAN-like)

    The resulting graph nodes correspond to clusters of similar problems,
    and edges connect clusters that share problems (overlap).
    Cycles in the graph correspond to H1 features.
    """

    def __init__(
        self,
        n_intervals: int = 10,
        overlap_frac: float = 0.3,
        n_clusters_per_interval: int = 3,
        random_state: int = 42,
    ):
        self.n_intervals = n_intervals
        self.overlap = overlap_frac
        self.k = n_clusters_per_interval
        self.rs = random_state
        self.nodes_ = None
        self.edges_ = None
        self.node_labels_ = None

    def fit(
        self,
        Z: np.ndarray,
        filter_values: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> 'MapperGraph':
        """
        Build the Mapper graph.

        Args:
            Z: (n, d) point cloud
            filter_values: (n,) filter function values; if None, use first PCA component
            labels: (n,) class labels for node coloring
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        if filter_values is None:
            pca = PCA(n_components=1, random_state=self.rs)
            filter_values = pca.fit_transform(Z).flatten()

        f_min, f_max = filter_values.min(), filter_values.max()
        interval_width = (f_max - f_min) / self.n_intervals
        step = interval_width * (1 - self.overlap)

        nodes = {}   # node_id -> set of point indices
        node_id = 0

        for i in range(self.n_intervals):
            lo = f_min + i * step
            hi = lo + interval_width
            # Points in this interval
            mask = (filter_values >= lo) & (filter_values <= hi)
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            # Cluster within interval
            k_actual = min(self.k, len(idx))
            if k_actual < 2:
                nodes[node_id] = set(idx.tolist())
                node_id += 1
                continue

            km = KMeans(n_clusters=k_actual, random_state=self.rs, n_init=3)
            try:
                cluster_ids = km.fit_predict(Z[idx])
                for c in range(k_actual):
                    cluster_pts = idx[cluster_ids == c]
                    if len(cluster_pts) >= 1:
                        nodes[node_id] = set(cluster_pts.tolist())
                        node_id += 1
            except Exception:
                nodes[node_id] = set(idx.tolist())
                node_id += 1

        # Build edges: connect nodes sharing at least 1 point
        node_list = list(nodes.items())
        edges = []
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                if node_list[i][1] & node_list[j][1]:
                    edges.append((node_list[i][0], node_list[j][0]))

        self.nodes_ = {nid: pts for nid, pts in nodes.items()}
        self.edges_ = edges

        # Compute node label distributions
        if labels is not None:
            self.node_labels_ = {}
            for nid, pts in self.nodes_.items():
                pt_labels = labels[list(pts)]
                self.node_labels_[nid] = {
                    int(c): int((pt_labels == c).sum())
                    for c in np.unique(labels)
                }

        return self

    def graph_stats(self) -> Dict:
        """Compute topological statistics of the Mapper graph."""
        if self.nodes_ is None:
            return {}

        n_nodes = len(self.nodes_)
        n_edges = len(self.edges_)

        # Euler characteristic: χ = V - E (for a graph)
        euler = n_nodes - n_edges

        # Connected components (β0) via union-find
        parent = {nid: nid for nid in self.nodes_}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for u, v in self.edges_:
            union(u, v)

        components = len(set(find(nid) for nid in self.nodes_))

        # β1 = E - V + components (from Euler: χ = V - E = components - β1)
        beta1 = n_edges - n_nodes + components

        # Node size statistics
        sizes = [len(pts) for pts in self.nodes_.values()]

        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'euler_characteristic': euler,
            'beta0_components': components,
            'beta1_cycles': max(0, beta1),
            'mean_node_size': float(np.mean(sizes)),
            'max_node_size': int(np.max(sizes)),
            'density': n_edges / max(1, n_nodes * (n_nodes - 1) / 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full TDA report
# ─────────────────────────────────────────────────────────────────────────────

def full_tda_report(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_permutations: int = 100,
    random_state: int = 42,
) -> Dict:
    """Run all TDA analyses and return structured report."""
    report = {}

    print("[TDA 1/4] Global persistent homology...")
    report['global_persistence'] = compute_persistence(
        Z, max_dim=1, metric='cosine',
        n_subsample=min(120, len(Z)), random_state=random_state
    )

    print("[TDA 2/4] Class-conditional persistent homology...")
    report['class_persistence'] = class_persistence(
        Z, labels, class_names, max_dim=1,
        n_subsample=35, random_state=random_state
    )

    print("[TDA 3/4] Topological permutation test...")
    report['topology_permtest'] = topological_permutation_test(
        Z, labels, class_names,
        n_permutations=n_permutations, random_state=random_state
    )

    print("[TDA 4/4] Mapper graph...")
    mapper = MapperGraph(n_intervals=8, overlap_frac=0.35,
                         n_clusters_per_interval=3, random_state=random_state)
    mapper.fit(Z, labels=labels)
    report['mapper'] = mapper.graph_stats()
    if mapper.node_labels_:
        report['mapper']['node_label_distributions'] = mapper.node_labels_

    return report


if __name__ == '__main__':
    import json
    np.random.seed(42)

    print("=== TOPOLOGICAL DATA ANALYSIS SMOKE TEST ===\n")

    # Simulate 4-class embedding
    n_per = 35
    d = 20
    Z = np.vstack([
        np.random.randn(n_per, d) * 0.4 + np.array([2, 0] + [0]*(d-2)),
        np.random.randn(n_per, d) * 0.4 + np.array([0, 2] + [0]*(d-2)),
        np.random.randn(n_per, d) * 0.4 + np.array([-2, 0] + [0]*(d-2)),
        np.random.randn(n_per, d) * 0.4 + np.array([0, -2] + [0]*(d-2)),
    ])
    labels = np.repeat([0, 1, 2, 3], n_per)
    class_names = ['Number Theory', 'Combinatorics', 'Algebra', 'Geometry']

    print("Global persistence:")
    result = compute_persistence(Z, max_dim=1, metric='euclidean', n_subsample=60)
    for dim_key, stats in result['homology'].items():
        print(f"  {dim_key}: n_features={stats['n_features']}, "
              f"entropy={stats['persistent_entropy']:.3f}, "
              f"n_significant={stats['n_significant']}")

    print("\nMapper graph:")
    mapper = MapperGraph(n_intervals=6, random_state=42)
    mapper.fit(Z, labels=labels)
    stats = mapper.graph_stats()
    print(f"  Nodes={stats['n_nodes']}, Edges={stats['n_edges']}, "
          f"β0={stats['beta0_components']}, β1={stats['beta1_cycles']}")
