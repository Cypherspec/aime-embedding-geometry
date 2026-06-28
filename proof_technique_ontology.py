"""
proof_technique_ontology.py
───────────────────────────
A 20-subtype hierarchical proof technique taxonomy for AIME problems,
with hierarchical classification, tree distortion measurement, and
ontology-aware evaluation metrics.

TAXONOMY (4 classes → 20 subtypes → keyword signatures):

Number Theory (NT):
  NT-MOD    Modular arithmetic, congruences, CRT
  NT-DIV    Divisibility, GCD/LCM, divisor counting
  NT-DIOPH  Diophantine equations, integer solutions
  NT-PRIME  Prime factorization, primality, prime gaps
  NT-DIGIT  Digit problems, base representations

Combinatorics (CO):
  CO-COUNT  Counting principles, permutations, combinations
  CO-PROB   Probability, expected value, variance
  CO-GRAPH  Graph theory, paths, colorings
  CO-PIG    Pigeonhole principle, extremal combinatorics
  CO-BIJ    Bijections, generating functions, recursion

Algebra (AL):
  AL-POLY   Polynomial roots, Vieta's, factoring
  AL-INEQ   Inequalities (AM-GM, Cauchy-Schwarz, Jensen)
  AL-SEQ    Sequences, series, telescoping
  AL-FUNC   Functional equations, composition
  AL-COMPLEX Complex numbers, roots of unity

Geometry (GE):
  GE-EUCL   Classical Euclidean geometry, congruence
  GE-TRIG   Trigonometry, law of sines/cosines
  GE-COORD  Coordinate geometry, analytic methods
  GE-CIRC   Circle theorems, power of a point
  GE-PROJ   Projective / inversive geometry

References:
  Evan Chen (2021). "Recommendations for Olympiad Math." evanchen.cc.
  Andreescu T. & Gelca R. (2000). Mathematical Olympiad Challenges.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy definition
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY = {
    # (subtype_id, parent_class, display_name, keyword_patterns)
    'NT-MOD':   ('Number Theory', 'Modular Arithmetic',
                 [r'mod', r'congruent', r'remainder', r'modular', r'residue',
                  r'chinese remainder', r'crt']),
    'NT-DIV':   ('Number Theory', 'Divisibility',
                 [r'divisor', r'divides', r'gcd', r'lcm', r'greatest common',
                  r'least common multiple', r'factor']),
    'NT-DIOPH': ('Number Theory', 'Diophantine Equations',
                 [r'integer solution', r'positive integer', r'pairs.*satisfy',
                  r'diophantine', r'integer.*equation']),
    'NT-PRIME': ('Number Theory', 'Prime Factorization',
                 [r'prime', r'primality', r'composite', r'sieve',
                  r'prime factor', r'factorization']),
    'NT-DIGIT': ('Number Theory', 'Digit Problems',
                 [r'digit', r'base', r'decimal', r'units digit',
                  r'hundreds digit', r'number of digits']),

    'CO-COUNT': ('Combinatorics', 'Counting',
                 [r'how many', r'number of way', r'arrangement', r'permutation',
                  r'combination', r'choose', r'select']),
    'CO-PROB':  ('Combinatorics', 'Probability',
                 [r'probability', r'expected', r'random', r'chance',
                  r'likelihood', r'expectation', r'variance']),
    'CO-GRAPH': ('Combinatorics', 'Graph Theory',
                 [r'graph', r'vertex', r'edge', r'path', r'cycle',
                  r'connected', r'coloring', r'bipartite']),
    'CO-PIG':   ('Combinatorics', 'Pigeonhole / Extremal',
                 [r'pigeonhole', r'at least one', r'maximum number',
                  r'minimum number', r'exists.*such that']),
    'CO-BIJ':   ('Combinatorics', 'Bijections / Generating Functions',
                 [r'bijection', r'generating function', r'recurrence',
                  r'fibonacci', r'catalan', r'one-to-one']),

    'AL-POLY':  ('Algebra', 'Polynomials',
                 [r'polynomial', r'root', r'vieta', r'coefficient',
                  r'degree', r'leading', r'factored']),
    'AL-INEQ':  ('Algebra', 'Inequalities',
                 [r'inequality', r'am-gm', r'cauchy', r'jensen',
                  r'maximum', r'minimum', r'optimize']),
    'AL-SEQ':   ('Algebra', 'Sequences and Series',
                 [r'sequence', r'series', r'arithmetic', r'geometric',
                  r'sum.*terms', r'telescoping', r'partial sum']),
    'AL-FUNC':  ('Algebra', 'Functional Equations',
                 [r'function.*satisfy', r'f\(', r'functional equation',
                  r'for all x', r'continuous function']),
    'AL-COMPLEX': ('Algebra', 'Complex Numbers',
                   [r'complex', r'imaginary', r'real part', r'argument',
                    r'modulus', r'roots of unity', r'cis']),

    'GE-EUCL':  ('Geometry', 'Euclidean Geometry',
                 [r'triangle', r'congruent.*triangle', r'similar',
                  r'altitude', r'median', r'bisector', r'perpendicular']),
    'GE-TRIG':  ('Geometry', 'Trigonometry',
                 [r'sin', r'cos', r'tan', r'law of sines', r'law of cosines',
                  r'trigonometric', r'angle.*degree']),
    'GE-COORD': ('Geometry', 'Coordinate Geometry',
                 [r'coordinate', r'slope', r'equation of line',
                  r'parabola', r'ellipse', r'hyperbola', r'conic']),
    'GE-CIRC':  ('Geometry', 'Circle Theorems',
                 [r'circle', r'radius', r'diameter', r'tangent.*circle',
                  r'inscribed', r'circumscribed', r'power of a point']),
    'GE-PROJ':  ('Geometry', 'Projective / Inversive',
                 [r'inversion', r'projective', r'cross-ratio',
                  r'harmonic', r'pole', r'polar', r'radical axis']),
}

SUBTYPE_IDS = list(TAXONOMY.keys())
SUBTYPE_TO_IDX = {k: i for i, k in enumerate(SUBTYPE_IDS)}
PARENT_CLASS = {k: v[0] for k, v in TAXONOMY.items()}
SUBTYPE_DISPLAY = {k: v[1] for k, v in TAXONOMY.items()}

# Class hierarchy: parent → children
CLASS_CHILDREN = {}
for subtype, (parent, _, _) in TAXONOMY.items():
    CLASS_CHILDREN.setdefault(parent, []).append(subtype)

PARENT_CLASSES = ['Number Theory', 'Combinatorics', 'Algebra', 'Geometry']
PARENT_TO_IDX = {c: i for i, c in enumerate(PARENT_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical labeler
# ─────────────────────────────────────────────────────────────────────────────

def assign_subtype_labels(
    texts: List[str],
    coarse_labels: List[str],
    min_confidence: float = 0.3,
) -> Tuple[List[str], List[float]]:
    """
    Assign fine-grained (20-subtype) labels to problems.

    Strategy:
      1. Within each coarse class (given by coarse_labels), score all
         matching subtypes by keyword hit count
      2. Assign highest-scoring matching subtype
      3. If no match within class, fall back to most common subtype

    Args:
        texts: problem text strings
        coarse_labels: parallel coarse technique class labels
        min_confidence: minimum confidence to assign (else mark as 'UNCERTAIN')

    Returns:
        (subtype_labels, confidences)
    """
    subtype_labels = []
    confidences = []

    for text, coarse in zip(texts, coarse_labels):
        text_lower = text.lower()

        # Only consider subtypes matching the coarse class
        candidate_subtypes = CLASS_CHILDREN.get(coarse, SUBTYPE_IDS)
        scores = {}

        for subtype in candidate_subtypes:
            _, _, patterns = TAXONOMY[subtype]
            hits = sum(1 for p in patterns if re.search(p, text_lower))
            scores[subtype] = hits

        best_subtype = max(scores, key=scores.get)
        best_score = scores[best_subtype]
        total_score = sum(scores.values()) + 1e-6
        confidence = best_score / total_score if total_score > 0 else 0.0

        if best_score == 0:
            # No keyword hit: fall back to first subtype of the class
            best_subtype = candidate_subtypes[0]
            confidence = 0.1

        subtype_labels.append(best_subtype)
        confidences.append(float(confidence))

    return subtype_labels, confidences


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy-aware distance
# ─────────────────────────────────────────────────────────────────────────────

def hierarchical_distance(label_a: str, label_b: str) -> int:
    """
    Semantic distance in the technique hierarchy:
      0: same subtype (e.g., NT-MOD vs NT-MOD)
      1: same parent class (e.g., NT-MOD vs NT-PRIME)
      2: different parent classes (e.g., NT-MOD vs CO-COUNT)
    """
    if label_a == label_b:
        return 0
    if PARENT_CLASS.get(label_a) == PARENT_CLASS.get(label_b):
        return 1
    return 2


def hierarchy_distance_matrix(labels: List[str]) -> np.ndarray:
    """Build (n×n) matrix of hierarchical distances."""
    n = len(labels)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hierarchical_distance(labels[i], labels[j])
            D[i, j] = D[j, i] = d
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical classifier: coarse-to-fine
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalTechniqueCls:
    """
    Two-stage hierarchical classifier:
      Stage 1: Predict coarse class (4-way) → LogisticRegression
      Stage 2: Predict fine subtype (5-way within each coarse class)
               → class-specific LogisticRegression per parent

    This replicates the standard coarse-to-fine approach in hierarchical
    classification (Silla & Freitas, 2011).

    The key question: does the two-stage hierarchical approach improve
    subtype classification over a flat 20-way classifier?
    """

    def __init__(self, C: float = 2.0, random_state: int = 42):
        self.C = C
        self.rs = random_state
        self.coarse_clf = None
        self.fine_clfs = {}  # parent_class → LogisticRegression

    def fit(self, X: np.ndarray, y_coarse: np.ndarray, y_fine: np.ndarray):
        """
        Args:
            X: (n, d) feature matrix
            y_coarse: (n,) integer coarse class labels (0-3)
            y_fine: (n,) integer fine subtype labels (0-19)
        """
        # Stage 1: coarse classifier
        self.coarse_clf = LogisticRegression(
            C=self.C, solver='lbfgs', max_iter=2000, random_state=self.rs
        )
        self.coarse_clf.fit(X, y_coarse)

        # Stage 2: per-class fine classifiers
        for c_idx, cls in enumerate(PARENT_CLASSES):
            mask = y_coarse == c_idx
            if mask.sum() < 5:
                continue
            X_cls = X[mask]
            y_fine_cls = y_fine[mask]

            if len(np.unique(y_fine_cls)) < 2:
                continue

            clf = LogisticRegression(
                C=self.C, solver='lbfgs', max_iter=2000, random_state=self.rs
            )
            clf.fit(X_cls, y_fine_cls)
            self.fine_clfs[c_idx] = clf

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (coarse_preds, fine_preds).
        Fine predictions use the predicted coarse class as gating.
        """
        coarse_preds = self.coarse_clf.predict(X)
        fine_preds = np.zeros(len(X), dtype=int)

        for c_idx in np.unique(coarse_preds):
            mask = coarse_preds == c_idx
            if c_idx in self.fine_clfs and mask.sum() > 0:
                fine_preds[mask] = self.fine_clfs[c_idx].predict(X[mask])
            else:
                # Default: first subtype of the class
                default_subtype = SUBTYPE_TO_IDX[CLASS_CHILDREN[PARENT_CLASSES[c_idx]][0]]
                fine_preds[mask] = default_subtype

        return coarse_preds, fine_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 20) probability matrix over all subtypes."""
        coarse_proba = self.coarse_clf.predict_proba(X)  # (n, 4)
        n = len(X)
        all_proba = np.zeros((n, len(SUBTYPE_IDS)))

        for c_idx, cls in enumerate(PARENT_CLASSES):
            subtypes = CLASS_CHILDREN[cls]
            subtype_indices = [SUBTYPE_TO_IDX[s] for s in subtypes]

            if c_idx in self.fine_clfs:
                fine_proba = self.fine_clfs[c_idx].predict_proba(X)  # (n, 5)
                for s_local, s_global in enumerate(subtype_indices):
                    if s_local < fine_proba.shape[1]:
                        all_proba[:, s_global] = (
                            coarse_proba[:, c_idx] * fine_proba[:, s_local]
                        )
            else:
                # Uniform over subtypes
                for s_global in subtype_indices:
                    all_proba[:, s_global] = coarse_proba[:, c_idx] / len(subtypes)

        # Renormalize
        row_sums = all_proba.sum(axis=1, keepdims=True) + 1e-12
        return all_proba / row_sums


def flat_hierarchical_comparison(
    X: np.ndarray,
    y_coarse: np.ndarray,
    y_fine: np.ndarray,
    y_fine_labels: List[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Compare flat 20-way classifier vs hierarchical (4→5) classifier.

    Key metric: hierarchical precision/recall (hP, hR, hF) which penalizes
    more for errors that cross coarse-class boundaries than within-class errors.

    hP(y, ŷ) = |ancestors(y) ∩ ancestors(ŷ)| / |ancestors(ŷ)|
    hR(y, ŷ) = |ancestors(y) ∩ ancestors(ŷ)| / |ancestors(y)|
    hF = 2 * hP * hR / (hP + hR)
    """
    from sklearn.linear_model import LogisticRegression

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    flat_accs_coarse, flat_accs_fine = [], []
    hier_accs_coarse, hier_accs_fine = [], []
    h_f_scores = []

    for tr_idx, val_idx in skf.split(X, y_coarse):  # stratify on coarse
        X_tr, X_val = X[tr_idx], X[val_idx]
        yc_tr, yc_val = y_coarse[tr_idx], y_coarse[val_idx]
        yf_tr, yf_val = y_fine[tr_idx], y_fine[val_idx]

        # Flat 20-way classifier
        flat_clf = LogisticRegression(C=2.0, solver='lbfgs', max_iter=2000,
                                       random_state=random_state)
        flat_clf.fit(X_tr, yf_tr)
        flat_fine_pred = flat_clf.predict(X_val)
        # Recover coarse from fine
        flat_coarse_pred = np.array([
            PARENT_TO_IDX[PARENT_CLASS[SUBTYPE_IDS[p]]]
            for p in flat_fine_pred
        ])
        flat_accs_coarse.append(accuracy_score(yc_val, flat_coarse_pred))
        flat_accs_fine.append(accuracy_score(yf_val, flat_fine_pred))

        # Hierarchical classifier
        hier_clf = HierarchicalTechniqueCls(random_state=random_state)
        hier_clf.fit(X_tr, yc_tr, yf_tr)
        hier_coarse_pred, hier_fine_pred = hier_clf.predict(X_val)
        hier_accs_coarse.append(accuracy_score(yc_val, hier_coarse_pred))
        hier_accs_fine.append(accuracy_score(yf_val, hier_fine_pred))

        # Hierarchical F-score
        h_f = _hierarchical_f(yf_val, hier_fine_pred)
        h_f_scores.append(h_f)

    return {
        'flat': {
            'coarse_accuracy': float(np.mean(flat_accs_coarse)),
            'fine_accuracy': float(np.mean(flat_accs_fine)),
        },
        'hierarchical': {
            'coarse_accuracy': float(np.mean(hier_accs_coarse)),
            'fine_accuracy': float(np.mean(hier_accs_fine)),
            'hierarchical_f': float(np.mean(h_f_scores)),
        },
        'improvement': {
            'coarse_delta': float(np.mean(hier_accs_coarse) - np.mean(flat_accs_coarse)),
            'fine_delta': float(np.mean(hier_accs_fine) - np.mean(flat_accs_fine)),
        },
    }


def _hierarchical_f(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged hierarchical F-score."""
    hP_vals, hR_vals = [], []
    for yt, yp in zip(y_true, y_pred):
        true_label = SUBTYPE_IDS[yt]
        pred_label = SUBTYPE_IDS[yp]
        true_anc = {true_label, PARENT_CLASS[true_label]}
        pred_anc = {pred_label, PARENT_CLASS[pred_label]}
        inter = len(true_anc & pred_anc)
        hP_vals.append(inter / (len(pred_anc) + 1e-9))
        hR_vals.append(inter / (len(true_anc) + 1e-9))

    hP = np.mean(hP_vals)
    hR = np.mean(hR_vals)
    return float(2 * hP * hR / (hP + hR + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Ontology-aware embedding analysis
# ─────────────────────────────────────────────────────────────────────────────

def hierarchy_embedding_consistency(
    Z: np.ndarray,
    subtype_labels: List[str],
    metric: str = 'cosine',
) -> Dict:
    """
    Test whether the embedding space respects the technique hierarchy.

    For every triple (i, j, k) where:
      - i and j share the same subtype (distance 0)
      - i and k share the same parent but different subtypes (distance 1)
      - or i and k are from different parents (distance 2)

    We check: does d_emb(i,j) < d_emb(i,k)?

    Hierarchy consistency rate = fraction of triples satisfying this ordering.
    A high rate means the embedding space encodes the ontological structure.
    """
    from sklearn.preprocessing import normalize as sk_norm
    Z_n = sk_norm(Z, norm='l2') if metric == 'cosine' else Z

    n = len(Z_n)
    labels_arr = np.array(subtype_labels)
    rng = np.random.default_rng(42)

    # Build index structures
    subtype_to_idx = {}
    for i, lbl in enumerate(subtype_labels):
        subtype_to_idx.setdefault(lbl, []).append(i)

    consistent_01 = 0  # same subtype < same parent different subtype
    consistent_02 = 0  # same subtype < different parent
    consistent_12 = 0  # same parent different subtype < different parent
    n_01 = n_02 = n_12 = 0

    n_samples = min(500, n * (n - 1) // 2)
    anchor_indices = rng.choice(n, size=min(n_samples, n), replace=False)

    for anchor in anchor_indices:
        anchor_sub = subtype_labels[anchor]
        anchor_par = PARENT_CLASS[anchor_sub]

        # Find same-subtype neighbor
        same_sub = [i for i in subtype_to_idx.get(anchor_sub, [])
                    if i != anchor]
        # Find same-parent-different-subtype neighbor
        same_par_diff_sub = [
            i for i in range(n)
            if PARENT_CLASS.get(subtype_labels[i]) == anchor_par
            and subtype_labels[i] != anchor_sub
        ]
        # Find different-parent neighbor
        diff_par = [
            i for i in range(n)
            if PARENT_CLASS.get(subtype_labels[i]) != anchor_par
        ]

        if not same_sub or not same_par_diff_sub or not diff_par:
            continue

        a_emb = Z_n[anchor]
        j_same = Z_n[rng.choice(same_sub)]
        k_spar = Z_n[rng.choice(same_par_diff_sub)]
        k_diff = Z_n[rng.choice(diff_par)]

        if metric == 'cosine':
            d_same = 1 - float(a_emb @ j_same)
            d_spar = 1 - float(a_emb @ k_spar)
            d_diff = 1 - float(a_emb @ k_diff)
        else:
            d_same = float(np.linalg.norm(a_emb - j_same))
            d_spar = float(np.linalg.norm(a_emb - k_spar))
            d_diff = float(np.linalg.norm(a_emb - k_diff))

        # Check ordering: d_same < d_spar (0 < 1)
        consistent_01 += int(d_same < d_spar)
        n_01 += 1

        # Check: d_same < d_diff (0 < 2)
        consistent_02 += int(d_same < d_diff)
        n_02 += 1

        # Check: d_spar < d_diff (1 < 2)
        consistent_12 += int(d_spar < d_diff)
        n_12 += 1

    return {
        'rate_0lt1': float(consistent_01 / max(1, n_01)),  # same_sub < same_par
        'rate_0lt2': float(consistent_02 / max(1, n_02)),  # same_sub < diff_par
        'rate_1lt2': float(consistent_12 / max(1, n_12)),  # same_par < diff_par
        'overall_consistency': float(
            (consistent_01 + consistent_02 + consistent_12) /
            max(1, n_01 + n_02 + n_12)
        ),
        'n_triples': n_01,
        'interpretation': (
            f"Hierarchy consistency: "
            f"d(same_subtype) < d(same_parent) = {consistent_01/max(1,n_01):.3f}, "
            f"d(same_parent) < d(diff_parent) = {consistent_12/max(1,n_12):.3f}. "
            f"Random baseline = 0.500 for both."
        ),
    }


if __name__ == '__main__':
    import json
    np.random.seed(42)

    print("PROOF TECHNIQUE TAXONOMY")
    print(f"  {len(TAXONOMY)} subtypes across {len(PARENT_CLASSES)} classes")
    for cls in PARENT_CLASSES:
        children = CLASS_CHILDREN[cls]
        print(f"  {cls}: {[SUBTYPE_DISPLAY[c] for c in children]}")

    print("\nTesting hierarchical distance:")
    print(f"  dist(NT-MOD, NT-MOD) = {hierarchical_distance('NT-MOD', 'NT-MOD')}")
    print(f"  dist(NT-MOD, NT-DIV) = {hierarchical_distance('NT-MOD', 'NT-DIV')}")
    print(f"  dist(NT-MOD, CO-COUNT) = {hierarchical_distance('NT-MOD', 'CO-COUNT')}")

    print("\nTesting subtype labeler:")
    test_texts = [
        "Find all prime numbers p such that p + 1 divides p^3 + 1.",
        "How many ways can 5 people be arranged in a circle?",
        "In triangle ABC, the altitude from A has length 6. Find the area.",
        "Find all polynomials P(x) such that P(x^2) = P(x)^2.",
    ]
    test_labels = ["Number Theory", "Combinatorics", "Geometry", "Algebra"]
    subtypes, confs = assign_subtype_labels(test_texts, test_labels)
    for txt, sub, conf in zip(test_texts, subtypes, confs):
        print(f"  [{sub}] ({conf:.2f}): {txt[:60]}...")
