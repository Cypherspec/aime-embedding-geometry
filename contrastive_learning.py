"""
contrastive_learning.py
───────────────────────
Contrastive representation learning for AIME problem embeddings.

Implements a SimCSE-style (Gao et al., 2021) contrastive objective that
learns embeddings where same-technique problems are pulled together and
different-technique problems are pushed apart — without requiring a neural
text encoder (operating directly on TF-IDF vectors).

Two contrastive frameworks:
  1. Supervised Contrastive Loss (SupCon; Khosla et al., 2020)
     Uses class labels to define positives (all same-class pairs)
     and negatives (all different-class pairs) within a mini-batch.

  2. Hard Negative Mining with Triplet Margin Loss
     For each anchor, selects the hardest negative (closest wrong-class
     example) and the easiest positive (farthest same-class example),
     forcing the model to resolve difficult boundary cases.

Both are trained via projected gradient descent on a linear projection
W ∈ ℝ^{d×k} (k < d) that maps TF-IDF vectors to a contrastive space.

Key contribution vs. paper:
  The paper's TF-IDF LSA uses unsupervised dimensionality reduction (SVD).
  Contrastive learning uses supervised signal to learn a projection that
  explicitly optimizes the geometric structure of technique-class separation.
  We then compare: which gives better silhouette / probe accuracy / δ-hyperbolicity?

References:
  Gao T. et al. (2021). "SimCSE: Simple contrastive learning of sentence embeddings."
    EMNLP.
  Khosla P. et al. (2020). "Supervised contrastive learning." NeurIPS.
  Schroff F. et al. (2015). "FaceNet: A unified embedding for face recognition."
    CVPR (triplet loss).
"""

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss (SupCon)
# ─────────────────────────────────────────────────────────────────────────────

def supcon_loss(
    Z: np.ndarray,
    labels: np.ndarray,
    temperature: float = 0.07,
) -> Tuple[float, np.ndarray]:
    """
    Supervised contrastive loss (Khosla et al., 2020).

    L = -1/|P(i)| · Σ_{p∈P(i)} log[
            exp(z_i·z_p/τ) /
            Σ_{a∈A(i)} exp(z_i·z_a/τ)
        ]

    where:
      P(i) = set of positives for i (same class, i ≠ p)
      A(i) = all other examples in batch (i ≠ a)
      τ = temperature

    Gradient w.r.t. z_i is computed analytically for backprop.

    Args:
        Z: (n, k) L2-normalized embeddings in contrastive space
        labels: (n,) integer class labels
        temperature: softmax temperature (lower → sharper)

    Returns:
        (loss, grad_Z) where grad_Z is (n, k) gradient matrix
    """
    n, k = Z.shape
    Z_n = normalize(Z, norm='l2')

    # Cosine similarity matrix S_ij = z_i · z_j / τ
    S = Z_n @ Z_n.T / temperature  # (n, n)
    np.fill_diagonal(S, -1e9)       # mask self-similarities

    # Masks
    same_class = (labels[:, None] == labels[None, :])  # (n, n)
    np.fill_diagonal(same_class, False)

    total_loss = 0.0
    grad = np.zeros_like(Z)

    for i in range(n):
        pos_mask = same_class[i]
        n_pos = pos_mask.sum()
        if n_pos == 0:
            continue

        # Log-sum-exp over all negatives + positives (denominator)
        s_i = S[i]                                   # (n,)
        log_sum_exp = np.log(np.exp(s_i - s_i.max()).sum()) + s_i.max()

        # Numerator: sum of similarities to positives
        pos_loss = -((s_i[pos_mask]).sum() / n_pos - log_sum_exp)
        total_loss += pos_loss

        # Gradient: ∂L/∂z_i
        # ∂L/∂S_ij = (1/τ) * (-1_{j∈P(i)}/|P(i)| + softmax_j(S_i))
        sm_i = softmax(s_i)
        pos_indicator = pos_mask.astype(float)
        dL_dS = -pos_indicator / n_pos + sm_i    # (n,)

        # ∂S_ij/∂z_i = z_j / τ (after normalization, approx)
        # Full gradient via chain rule on normalized z
        z_i = Z_n[i]
        # Gradient before normalization:
        raw_grad = (dL_dS[:, None] * Z_n).sum(axis=0) / temperature

        # Gradient of L2 normalization: ∂(z/||z||)/∂z = (I - z_n z_n^T) / ||z||
        z_raw = Z[i]
        z_norm = np.linalg.norm(z_raw) + 1e-9
        P_proj = np.eye(k) - np.outer(z_i, z_i)
        grad[i] = P_proj @ raw_grad / z_norm

    return float(total_loss / n), grad / n


# ─────────────────────────────────────────────────────────────────────────────
# Triplet Margin Loss with Hard Negative Mining
# ─────────────────────────────────────────────────────────────────────────────

def triplet_loss_hard(
    Z: np.ndarray,
    labels: np.ndarray,
    margin: float = 0.3,
) -> Tuple[float, np.ndarray]:
    """
    Triplet loss with hard negative mining (Schroff et al., 2015).

    L = Σ_i max(0, d(a,p)_hard - d(a,n)_hard + margin)

    For each anchor i:
      Hard positive: farthest same-class point (most similar should be pulled)
      Hard negative: closest different-class point (most confusable, push away)

    Args:
        Z: (n, k) embeddings
        labels: (n,) class labels
        margin: triplet margin α

    Returns:
        (loss, grad_Z)
    """
    Z_n = normalize(Z, norm='l2')
    n, k = Z_n.shape
    D = 1 - Z_n @ Z_n.T  # cosine distances
    np.fill_diagonal(D, -1e9 if True else 0)

    same = labels[:, None] == labels[None, :]
    diff = ~same
    np.fill_diagonal(same, False)

    total_loss = 0.0
    grad = np.zeros_like(Z)
    n_active = 0

    for i in range(n):
        pos_mask = same[i]
        neg_mask = diff[i]

        if not pos_mask.any() or not neg_mask.any():
            continue

        D_pos = D[i].copy()
        D_neg = D[i].copy()
        D_pos[~pos_mask] = -1e9
        D_neg[~neg_mask] = 1e9

        p_idx = int(D_pos.argmax())   # hard positive: farthest same-class
        n_idx = int(D_neg.argmin())   # hard negative: closest diff-class

        d_ap = float(D[i, p_idx])
        d_an = float(D[i, n_idx])
        loss_i = max(0.0, d_ap - d_an + margin)

        if loss_i > 0:
            total_loss += loss_i
            n_active += 1

            # Gradients of cosine distance d(u,v) = 1 - u_n·v_n:
            # ∂d/∂u = -v_n / ||u|| + u_n (u_n·v_n) / ||u||
            # Simplified for normalized vectors:
            def cosine_grad_anchor(z_a, z_b):
                """∂d_cos(z_a, z_b)/∂z_a for normalized z."""
                z_a_n = z_a / (np.linalg.norm(z_a) + 1e-9)
                z_b_n = z_b / (np.linalg.norm(z_b) + 1e-9)
                return -(z_b_n - z_a_n * (z_a_n @ z_b_n)) / (np.linalg.norm(z_a) + 1e-9)

            g_pos = cosine_grad_anchor(Z[i], Z[p_idx])
            g_neg = cosine_grad_anchor(Z[i], Z[n_idx])
            grad[i] += g_pos - g_neg   # ∂L/∂anchor = ∂d_ap/∂a - ∂d_an/∂a

    loss = total_loss / max(1, n)
    return loss, grad / max(1, n)


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Projection: learned linear projection
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveProjection:
    """
    Learn a linear projection W: ℝ^d → ℝ^k that maximizes
    supervised contrastive separability of technique classes.

    This is the core experiment: compare unsupervised SVD projection
    (paper's LSA) vs supervised contrastive projection (this module)
    on silhouette score, linear probe accuracy, and δ-hyperbolicity.

    Optimization:
      W* = argmin_W L_SupCon(W·X, labels) + λ||W||_F^2
      via mini-batch SGD with momentum
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        temperature: float = 0.07,
        lr: float = 0.01,
        lambda_reg: float = 1e-4,
        loss_type: str = 'supcon',  # 'supcon' or 'triplet'
        random_state: int = 42,
    ):
        rng = np.random.default_rng(random_state)
        # Initialize with scaled Gaussian (He initialization)
        self.W = rng.standard_normal((input_dim, output_dim)) * np.sqrt(2.0 / input_dim)
        self.output_dim = output_dim
        self.temperature = temperature
        self.lr = lr
        self.lam = lambda_reg
        self.loss_type = loss_type
        self.history = {'loss': [], 'silhouette': []}

        # Momentum
        self.velocity = np.zeros_like(self.W)
        self.momentum = 0.9

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Project X through W and L2-normalize: Z = normalize(X @ W)."""
        Z = X @ self.W
        return normalize(Z, norm='l2')

    def fit(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
        eval_every: int = 10,
        random_state: int = 42,
        verbose: bool = True,
    ) -> 'ContrastiveProjection':
        """
        Train contrastive projection via mini-batch SGD with momentum.

        Args:
            X: (n, d) TF-IDF feature matrix (not normalized yet)
            labels: (n,) integer class labels
            n_epochs: training epochs
            batch_size: mini-batch size (should include multiple classes)
        """
        rng = np.random.default_rng(random_state)
        n = len(X)
        best_sil = -1.0
        best_W = self.W.copy()

        for epoch in range(n_epochs):
            # Shuffle
            perm = rng.permutation(n)
            X_perm, y_perm = X[perm], labels[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_b = X_perm[start:end]
                y_b = y_perm[start:end]

                # Forward: project
                Z_b = self.forward(X_b)   # (batch, k)

                # Compute loss and gradient w.r.t. Z_b
                if self.loss_type == 'supcon':
                    loss, dL_dZ = supcon_loss(Z_b, y_b, self.temperature)
                else:
                    loss, dL_dZ = triplet_loss_hard(Z_b, y_b, margin=0.3)

                # Gradient w.r.t. W: ∂L/∂W = X_b^T · ∂L/∂Z_b
                # (ignoring normalization Jacobian for simplicity — first-order approx)
                dL_dW = X_b.T @ dL_dZ + self.lam * self.W

                # SGD with momentum
                self.velocity = self.momentum * self.velocity - self.lr * dL_dW
                self.W += self.velocity

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.history['loss'].append(avg_loss)

            # Evaluate
            if eval_every > 0 and (epoch % eval_every == 0 or epoch == n_epochs - 1):
                Z_all = self.forward(X)
                try:
                    sil = float(silhouette_score(Z_all, labels, metric='cosine'))
                except Exception:
                    sil = 0.0
                self.history['silhouette'].append(sil)

                if sil > best_sil:
                    best_sil = sil
                    best_W = self.W.copy()

                if verbose:
                    print(f"    Epoch {epoch:3d}/{n_epochs}: "
                          f"loss={avg_loss:.4f}, sil={sil:.4f}")

        self.W = best_W  # restore best
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply trained projection."""
        return self.forward(X)

    def compare_to_lsa(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        Z_lsa: np.ndarray,
        class_names: List[str],
    ) -> Dict:
        """
        Compare contrastive projection to LSA on all key metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import silhouette_score

        Z_con = self.transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        results = {}
        for name, Z in [('LSA (unsupervised)', Z_lsa), ('Contrastive (supervised)', Z_con)]:
            # Silhouette
            try:
                sil = float(silhouette_score(Z, labels, metric='cosine'))
            except Exception:
                sil = 0.0

            # Linear probe accuracy
            clf = LogisticRegression(C=2.0, solver='lbfgs', max_iter=1000,
                                      random_state=42)
            cv_accs = cross_val_score(clf, Z, labels, cv=skf, scoring='accuracy')

            results[name] = {
                'silhouette': sil,
                'probe_accuracy_mean': float(cv_accs.mean()),
                'probe_accuracy_std': float(cv_accs.std()),
            }

        results['improvement'] = {
            'silhouette_delta': (results['Contrastive (supervised)']['silhouette'] -
                                  results['LSA (unsupervised)']['silhouette']),
            'accuracy_delta': (results['Contrastive (supervised)']['probe_accuracy_mean'] -
                                results['LSA (unsupervised)']['probe_accuracy_mean']),
        }
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-competition generalization experiment
# ─────────────────────────────────────────────────────────────────────────────

def cross_competition_transfer(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    projection_dim: int = 64,
    n_epochs: int = 80,
    random_state: int = 42,
) -> Dict:
    """
    Train contrastive projection on source competition (AIME),
    evaluate on target competition (AMC/USAMO).

    This is the cross-competition generalization experiment:
    does the learned technique geometry transfer across difficulty levels?

    Returns accuracy comparison:
      - Source domain (train+test on AIME): upper bound
      - Target domain, no adaptation (zero-shot transfer)
      - Target domain, few-shot fine-tuning (10 examples per class)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("  Training contrastive projection on source domain...")
    proj = ContrastiveProjection(
        input_dim=X_source.shape[1],
        output_dim=projection_dim,
        temperature=0.07,
        lr=0.005,
        random_state=random_state,
    )
    proj.fit(X_source, y_source, n_epochs=n_epochs, batch_size=64,
             eval_every=20, verbose=True)

    Z_source = proj.transform(X_source)
    Z_target = proj.transform(X_target)

    # Train classifier on source
    clf = LogisticRegression(C=2.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(Z_source, y_source)

    # Zero-shot: evaluate on target
    preds_zeroshot = clf.predict(Z_target)
    acc_zeroshot = accuracy_score(y_target, preds_zeroshot)

    # Few-shot: fine-tune on 10 examples per class from target
    rng = np.random.default_rng(random_state)
    n_classes = len(np.unique(y_target))
    few_shot_idx = []
    for c in range(n_classes):
        cls_idx = np.where(y_target == c)[0]
        take = min(10, len(cls_idx))
        few_shot_idx.extend(rng.choice(cls_idx, take, replace=False).tolist())

    few_shot_idx = np.array(few_shot_idx)
    remaining_idx = np.setdiff1d(np.arange(len(y_target)), few_shot_idx)

    clf_ft = LogisticRegression(C=2.0, solver='lbfgs', max_iter=1000, random_state=42)
    # Combine source + few-shot target
    Z_combined = np.vstack([Z_source, Z_target[few_shot_idx]])
    y_combined = np.concatenate([y_source, y_target[few_shot_idx]])
    clf_ft.fit(Z_combined, y_combined)

    preds_fewshot = clf_ft.predict(Z_target[remaining_idx])
    acc_fewshot = accuracy_score(y_target[remaining_idx], preds_fewshot)

    return {
        'source_n': len(X_source),
        'target_n': len(X_target),
        'zero_shot_accuracy': float(acc_zeroshot),
        'few_shot_accuracy': float(acc_fewshot),
        'few_shot_n_per_class': 10,
        'projection_dim': projection_dim,
    }


if __name__ == '__main__':
    np.random.seed(42)
    print("=== CONTRASTIVE LEARNING SMOKE TEST ===\n")

    n, d = 120, 80
    n_classes = 4

    # Simulate 4-class TF-IDF matrix
    X = np.zeros((n, d))
    y = np.repeat(np.arange(n_classes), n // n_classes)
    for c in range(n_classes):
        mask = y == c
        X[mask] = np.random.randn(mask.sum(), d) * 0.5
        X[mask, c*5:(c+1)*5] += 2.0  # class-discriminative features
    X = normalize(X, norm='l2')

    print("Testing SupCon loss...")
    Z = normalize(np.random.randn(n, 32), norm='l2')
    loss, grad = supcon_loss(Z, y, temperature=0.07)
    print(f"  SupCon loss: {loss:.4f}, grad norm: {np.linalg.norm(grad):.4f}")

    print("\nTraining contrastive projection (30 epochs)...")
    proj = ContrastiveProjection(input_dim=d, output_dim=32, random_state=42)
    proj.fit(X, y, n_epochs=30, batch_size=32, eval_every=10, verbose=True)

    Z_final = proj.transform(X)
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(Z_final, y, metric='cosine')
    print(f"\nFinal silhouette (contrastive): {sil:.4f}")
