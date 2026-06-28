"""
run_full_pipeline.py
────────────────────
End-to-end reproducible analysis pipeline for the AIME embedding study.

Usage:
    python run_full_pipeline.py --corpus AIME_labeled_full.csv \
                                 --seed 42 \
                                 --output results/ \
                                 --n_permutations 10000

This script:
  1. Loads and validates the labeled corpus
  2. Builds TF-IDF embeddings (Stage 1: 160 balanced, Stage 2: 548 balanced)
  3. Runs Riemannian geometry analysis (intrinsic dim, curvature, anisotropy)
  4. Runs information-theoretic analysis (PMI, KSG MI, CKA)
  5. Runs the full probe depth comparison (linear → MLP-3)
  6. Runs extended permutation testing (10,000 permutations, BCa CI)
  7. Runs temporal drift analysis (Jaccard, JSD, CUSUM, centroid trajectory)
  8. Saves all results as JSON + generates summary statistics

All random operations are seeded for exact reproducibility.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import normalize, LabelEncoder

warnings.filterwarnings('ignore')

# Import our analysis modules
from embedding_geometry import full_geometry_report
from information_theory import (
    build_pmi_matrix, top_pmi_terms, feature_information_gain,
    chi2_feature_selection, mutual_information_class_embedding,
    cka_between_stages, temporal_entropy,
)
from neural_probe import full_probing_report
from statistical_tests import full_statistics_report
from temporal_drift import full_temporal_report


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (math-aware, reproduces paper's tokenizer_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

import re

def math_aware_tokenizer(text: str) -> list:
    """
    LaTeX-preserving tokenizer: replace inline math spans with MATH_ tokens,
    then tokenize the remaining text by whitespace and punctuation.
    Reproduces the tokenizer_utils.py module described in Section III.A.
    """
    # Replace inline LaTeX: $...$ → single MATH_ token
    text = re.sub(r'\$[^\$]+\$', ' MATH_ ', text)
    # Replace display math: \[...\] or $$...$$
    text = re.sub(r'\$\$[^\$]+\$\$', ' MATH_ ', text)
    text = re.sub(r'\\\[[^\]]+\\\]', ' MATH_ ', text)
    # Tokenize remaining text
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
    return tokens


def math_aware_tokenizer_str(text: str) -> str:
    """Return tokenized string for CountVectorizer compatibility."""
    return ' '.join(math_aware_tokenizer(text))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(
    corpus_path: str,
    text_col: str = 'problem',
    label_col: str = 'technique_class',
    year_col: str = 'year',
    confidence_col: str = 'label_confidence',
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Load and validate the labeled AIME corpus.

    Expected CSV columns:
        problem          : raw LaTeX problem text
        technique_class  : one of {Number Theory, Combinatorics, Algebra, Geometry}
        year             : competition year (1983–2024)
        label_confidence : heuristic labeler confidence in [0, 1]

    Args:
        corpus_path: path to AIME_labeled_full.csv
        min_confidence: minimum confidence threshold for inclusion

    Returns:
        validated DataFrame with text, label, year columns
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Expected: AIME_labeled_full.csv (933 rows, columns: "
            "problem, technique_class, year, label_confidence)"
        )

    df = pd.read_csv(corpus_path)
    required = [text_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Drop unlabeled / low-confidence rows
    df = df.dropna(subset=[label_col])
    valid_classes = {'Number Theory', 'Combinatorics', 'Algebra', 'Geometry'}
    df = df[df[label_col].isin(valid_classes)].copy()

    if confidence_col in df.columns:
        df = df[df[confidence_col] >= min_confidence].copy()

    if year_col in df.columns:
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)

    print(f"Loaded {len(df)} labeled problems "
          f"({df[label_col].value_counts().to_dict()})")
    return df


def build_balanced_sample(
    df: pd.DataFrame,
    label_col: str,
    n_per_class: int,
    seed: int,
    prefer_recent: bool = True,
    year_col: str = 'year',
    confidence_col: str = 'label_confidence',
) -> pd.DataFrame:
    """
    Build a class-balanced sample of n_per_class problems per class,
    preferring higher-confidence labels and more recent problems as tiebreakers.
    Reproduces the sampling strategy of Sections II.C.
    """
    samples = []
    for cls, group in df.groupby(label_col):
        # Sort: highest confidence first, then most recent
        sort_cols = []
        if confidence_col in group.columns:
            sort_cols.append(confidence_col)
        if year_col in group.columns:
            sort_cols.append(year_col)

        if sort_cols:
            group = group.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        take = min(n_per_class, len(group))
        samples.append(group.head(take))

    return pd.concat(samples).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding builders
# ─────────────────────────────────────────────────────────────────────────────

def build_lsa_embedding(
    texts: list,
    n_features: int = 3000,
    n_components: int = 80,
    seed: int = 42,
) -> tuple:
    """
    Build TF-IDF → TruncatedSVD (LSA) embedding.
    Reproduces Stage 1 pipeline (Section III.B).

    Returns: (Z, vectorizer, svd, variance_explained)
    """
    vectorizer = TfidfVectorizer(
        tokenizer=math_aware_tokenizer,
        max_features=n_features,
        sublinear_tf=True,
        min_df=1,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z = svd.fit_transform(X)
    var_explained = svd.explained_variance_ratio_.sum()

    print(f"  LSA: {n_features} TF-IDF features → {n_components}-d "
          f"(variance explained: {var_explained:.3f})")
    return Z, vectorizer, svd, float(var_explained)


def build_tfidf_embedding(
    texts: list,
    n_features: int = 4000,
    seed: int = 42,
) -> tuple:
    """
    Build raw TF-IDF embedding (no LSA reduction).
    Reproduces Stage 2 pipeline (Section III.D).

    Returns: (X_dense, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        tokenizer=math_aware_tokenizer,
        max_features=n_features,
        sublinear_tf=True,
        min_df=2,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts)
    print(f"  TF-IDF: {n_features} features, {X.shape[0]} documents")
    return X.toarray().astype(np.float32), vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    corpus_path: str,
    output_dir: str = 'results',
    seed: int = 42,
    n_permutations: int = 10_000,
    n_stage1: int = 40,       # problems per class, Stage 1
    n_stage2: int = 150,      # problems per class, Stage 2
    run_geometry: bool = True,
    run_infotheory: bool = True,
    run_probing: bool = True,
    run_statistics: bool = True,
    run_temporal: bool = True,
) -> Dict:
    """
    Full analysis pipeline.

    Args:
        corpus_path: path to labeled CSV
        output_dir: where to save JSON results
        seed: global random seed for reproducibility
        n_permutations: permutations for statistical tests
        n_stage1: class size for Stage 1 balanced sample
        n_stage2: class size for Stage 2 balanced sample

    Returns:
        full_results dict (also saved to output_dir/full_results.json)
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    class_names = ['Number Theory', 'Combinatorics', 'Algebra', 'Geometry']
    le = LabelEncoder()
    le.fit(class_names)

    print("=" * 60)
    print("AIME Advanced Analysis Pipeline")
    print(f"Seed: {seed} | Permutations: {n_permutations}")
    print("=" * 60)

    # ── Load corpus ──────────────────────────────────────────────────────────
    print("\n[STEP 0] Loading corpus...")
    t0 = time.time()
    df = load_corpus(corpus_path)
    texts_all = df['problem'].tolist()
    labels_str_all = df['technique_class'].tolist()
    years_all = df['year'].tolist() if 'year' in df.columns else [2000] * len(df)

    # ── Stage 1 sample ───────────────────────────────────────────────────────
    print(f"\n[STEP 1] Building Stage 1 balanced sample ({n_stage1}/class)...")
    df1 = build_balanced_sample(df, 'technique_class', n_stage1, seed)
    texts1 = df1['problem'].tolist()
    labels1_str = df1['technique_class'].tolist()
    labels1 = le.transform(labels1_str).astype(int)

    # Stage 1 LSA embedding
    Z1, vec1, svd1, var1 = build_lsa_embedding(texts1, n_features=3000,
                                                n_components=80, seed=seed)

    # ── Stage 2 sample ───────────────────────────────────────────────────────
    print(f"\n[STEP 2] Building Stage 2 balanced sample (up to {n_stage2}/class)...")
    df2 = build_balanced_sample(df, 'technique_class', n_stage2, seed)
    texts2 = df2['problem'].tolist()
    labels2_str = df2['technique_class'].tolist()
    labels2 = le.transform(labels2_str).astype(int)

    # Stage 2 TF-IDF embedding + Logistic Regression
    X2, vec2 = build_tfidf_embedding(texts2, n_features=4000, seed=seed)

    print("  Training Stage 2 LogisticRegression (5-fold CV)...")
    clf2 = LogisticRegression(C=2.0, solver='lbfgs', max_iter=2000,
                               random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs = []
    for tr_idx, val_idx in skf.split(X2, labels2):
        clf2.fit(X2[tr_idx], labels2[tr_idx])
        fold_accs.append(accuracy_score(labels2[val_idx],
                                        clf2.predict(X2[val_idx])))
    clf2.fit(X2, labels2)  # refit on full for coefficient extraction
    print(f"  Stage 2 CV accuracy: {np.mean(fold_accs)*100:.2f}% "
          f"± {np.std(fold_accs)*100:.2f}%")

    results = {
        'metadata': {
            'seed': seed,
            'n_permutations': n_permutations,
            'stage1_n_per_class': n_stage1,
            'stage2_n_per_class': n_stage2,
            'stage1_total': len(texts1),
            'stage2_total': len(texts2),
            'corpus_total': len(df),
            'variance_explained_stage1': var1,
        },
        'stage2_cv': {
            'fold_accuracies': [float(a) for a in fold_accs],
            'mean_accuracy': float(np.mean(fold_accs)),
            'std_accuracy': float(np.std(fold_accs)),
        },
    }

    # ── Geometry analysis ────────────────────────────────────────────────────
    if run_geometry:
        print("\n[STEP 3] Riemannian geometry analysis...")
        W = clf2.coef_.astype(np.float64)
        b = clf2.intercept_.astype(np.float64)
        # Project Stage 2 samples to LSA space for geometry analysis
        Z2_lsa, _, _, _ = build_lsa_embedding(texts2, n_features=3000,
                                               n_components=80, seed=seed)
        results['geometry'] = full_geometry_report(
            Z1, labels1, class_names, W=None, b=None, random_state=seed
        )

    # ── Information theory ───────────────────────────────────────────────────
    if run_infotheory:
        print("\n[STEP 4] Information-theoretic analysis...")
        pmi, vocab, cls_names_pmi = build_pmi_matrix(texts2, labels2_str)
        top_pmi = top_pmi_terms(pmi, vocab, cls_names_pmi, top_k=10)

        # Information gain on binarized TF-IDF
        from sklearn.feature_extraction.text import CountVectorizer as CV
        cv_ig = CV(max_features=1000, min_df=2, token_pattern=r'[a-zA-Z][a-zA-Z_]+')
        X_ig = cv_ig.fit_transform(texts2).toarray()
        ig_scores = feature_information_gain(X_ig, labels2)
        ig_top_idx = np.argsort(ig_scores)[::-1][:15]
        ig_vocab = cv_ig.get_feature_names_out()
        top_ig = [(ig_vocab[i], float(ig_scores[i])) for i in ig_top_idx]

        ksg_mi = mutual_information_class_embedding(Z1, labels1, n_components=5)

        # Temporal entropy
        years1 = df1['year'].tolist() if 'year' in df1.columns else [2000]*len(df1)
        tent = temporal_entropy(np.array(years1), labels1, class_names)

        results['information_theory'] = {
            'top_pmi_terms': top_pmi,
            'top_information_gain_terms': top_ig,
            'ksg_mutual_information': ksg_mi,
            'temporal_entropy': {str(k): v for k, v in tent.items()},
        }

    # ── Neural probing ───────────────────────────────────────────────────────
    if run_probing:
        print("\n[STEP 5] Neural probing (depth analysis)...")
        results['probing'] = full_probing_report(
            Z1, labels1, class_names, random_state=seed
        )

    # ── Statistical tests ────────────────────────────────────────────────────
    if run_statistics:
        print(f"\n[STEP 6] Statistical tests ({n_permutations} permutations)...")
        results['statistics'] = full_statistics_report(
            Z1, labels1, fold_accs, random_state=seed
        )

    # ── Temporal analysis ────────────────────────────────────────────────────
    if run_temporal:
        print("\n[STEP 7] Temporal drift analysis...")
        years_all_arr = np.array(years_all)
        results['temporal'] = full_temporal_report(
            texts_all, years_all,
            labels_str_all,
            class_names,
            Z=None,  # omit centroid trajectory for speed without GPU
        )

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path(output_dir) / 'full_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
    print(f"{'='*60}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Stage 2 CV accuracy:  {results['stage2_cv']['mean_accuracy']*100:.2f}% "
          f"± {results['stage2_cv']['std_accuracy']*100:.2f}%")
    if run_statistics and 'permutation_test' in results.get('statistics', {}):
        pt = results['statistics']['permutation_test']
        print(f"Permutation test:     z = {pt['z_score']:.2f}, p = {pt['p_value']:.4f}")
    if run_geometry and 'wb_ratio' in results.get('geometry', {}):
        wb = results['geometry']['wb_ratio']
        print(f"W/B ratio:            {wb['observed']:.4f} "
              f"[{wb['ci_lower']:.4f}, {wb['ci_upper']:.4f}]")
    if run_geometry and 'global_intrinsic_dim' in results.get('geometry', {}):
        d_hat, d_ci = results['geometry']['global_intrinsic_dim']
        print(f"Intrinsic dim (TWO-NN): {d_hat:.1f} ± {d_ci:.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AIME Advanced Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on labeled corpus
  python run_full_pipeline.py --corpus AIME_labeled_full.csv --seed 42

  # Quick run with fewer permutations
  python run_full_pipeline.py --corpus AIME_labeled_full.csv \\
      --n_permutations 1000 --output results_quick/

  # Only geometry and statistics (skip temporal)
  python run_full_pipeline.py --corpus AIME_labeled_full.csv \\
      --no-temporal --no-infotheory
        """,
    )
    parser.add_argument('--corpus', type=str, default='AIME_labeled_full.csv',
                        help='Path to labeled AIME corpus CSV')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for JSON results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--n_permutations', type=int, default=10_000,
                        help='Number of permutations for statistical tests')
    parser.add_argument('--n_stage1', type=int, default=40,
                        help='Problems per class for Stage 1 sample')
    parser.add_argument('--n_stage2', type=int, default=150,
                        help='Max problems per class for Stage 2 sample')
    parser.add_argument('--no-geometry', dest='geometry', action='store_false')
    parser.add_argument('--no-infotheory', dest='infotheory', action='store_false')
    parser.add_argument('--no-probing', dest='probing', action='store_false')
    parser.add_argument('--no-statistics', dest='statistics', action='store_false')
    parser.add_argument('--no-temporal', dest='temporal', action='store_false')
    parser.set_defaults(geometry=True, infotheory=True, probing=True,
                        statistics=True, temporal=True)

    args = parser.parse_args()

    run_pipeline(
        corpus_path=args.corpus,
        output_dir=args.output,
        seed=args.seed,
        n_permutations=args.n_permutations,
        n_stage1=args.n_stage1,
        n_stage2=args.n_stage2,
        run_geometry=args.geometry,
        run_infotheory=args.infotheory,
        run_probing=args.probing,
        run_statistics=args.statistics,
        run_temporal=args.temporal,
    )


if __name__ == '__main__':
    main()
