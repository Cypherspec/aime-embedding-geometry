# AIME Advanced Analysis Suite

Advanced computational analysis of AIME problem technique classification.
Accompanies: "Probing Technique-Structured Geometry in AIME Embedding Space"

Paper: Nischay Kommisetty - Probing Technique-Structured Geometry in AIME Embedding Space_ A Computational and Statistical Analysis of Mathematical Technique Classification Across Four Decades of Competition Mathemati (5).pdf
## Modules

- `embedding_geometry.py`   — Riemannian geometry of TF-IDF manifold; sectional curvature estimation
- `manifold_analysis.py`    — Intrinsic dimensionality, persistent homology (Betti numbers), geodesic distance
- `neural_probe.py`         — Multi-layer nonlinear probe; representational similarity analysis (CKA)
- `information_theory.py`   — PMI/PPMI matrices, information gain, mutual information estimation (KSG)
- `statistical_tests.py`    — Permutation testing, bootstrap CIs, FDR correction, power analysis
- `feature_geometry.py`     — Gradient-based feature attribution; SHAP values for LR coefficients
- `temporal_drift.py`       — Vocabulary drift analysis; year-stratified embedding trajectories
- `run_full_pipeline.py`    — End-to-end reproducible pipeline with seeding and logging

## Usage

    python run_full_pipeline.py --corpus AIME_labeled_full.csv --seed 42 --output results/
