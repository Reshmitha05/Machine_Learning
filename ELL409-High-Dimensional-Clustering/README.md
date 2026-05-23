# High-Dimensional Clustering Under Heterogeneous Structure

**ELL 409: Introduction to Machine Learning — Course Project 2**
**Score: 0.9891 (Weighted ARI)**

---

## What I Did

This was an unsupervised clustering competition on biological measurement data — no labels provided, just raw high-dimensional features. The task was to cluster four structurally different datasets (Parts A–D) and submit predictions evaluated by Adjusted Rand Index (ARI).

I built **four separate clustering pipelines**, one per part, since each part had a completely different data geometry and challenge.

---

## The Problem

| Part | Challenge | Data Type |
|------|-----------|-----------|
| A | Sparse marker signals buried in 500+ noisy gene features | Single-cell |
| B | Rare cell-type discovery — highly imbalanced clusters | Single-cell |
| C | Hierarchical + continuous transitions between cell states | Single-cell |
| D | Spatial tissue niches — cluster identity depends on *neighboring spots*, not just own features | Spatial spot |

Final score = `0.20×ARI_A + 0.25×ARI_B + 0.30×ARI_C + 0.25×ARI_D`

---

## My Approach

### Core Pipeline (All Parts)

1. **Variance filtering** — dropped the bottom 20–30% of features by variance to remove pure noise before any dimensionality reduction.
2. **RobustScaler** — used instead of StandardScaler because biological data has heavy outliers (dropout events, measurement noise).
3. **PCA** (50 components) — compressed to a stable linear subspace before UMAP.
4. **UMAP** — learned a nonlinear low-dimensional embedding tuned per part (different `n_neighbors` and `n_components`).
5. **Leiden clustering** — graph-based community detection on the UMAP embedding via `scanpy`. Much better than k-means for this kind of manifold data.
6. **Resolution scanning** — instead of guessing cluster count, I scanned a range of Leiden resolutions and picked the one with the best Silhouette score within a target cluster-count range.

### Part-Specific Tuning

**Part A (Sparse Markers)**
- Variance filter at 30th percentile, PCA→50, UMAP with `n_neighbors=30`, `n_components=15`
- Resolution scan: 0.6–1.1, target 5–15 clusters

**Part B (Rare Cell Types)**
- Kept `n_neighbors=15` in UMAP to preserve local neighborhoods around rare groups (large `n_neighbors` would smooth them away)
- Leiden `n_neighbors=15`, scanned very low resolutions (0.1–0.8) — v4 of this code was broken here because it scanned too-high resolutions and fragmented into 35 clusters
- Target: 6–20 clusters

**Part C (Hierarchical/Continuous States)**
- UMAP `n_components=20` to better capture continuous trajectories
- Higher resolution scan (0.7–1.4) to resolve fine sub-states
- Target: 8–25 clusters

**Part D (Spatial Niches)**
- Combined molecular features (all 500 genes + proteins + morphology → PCA 40) with:
  - Raw spatial features (`x_coord`, `y_coord`, `neighbor_density`, `local_heterogeneity`)
  - **KNN-derived neighborhood features**: for each spot, computed the mean and std of protein/morphology features across its 15 nearest spatial neighbors
- Blended all three with `alpha=0.7` spatial weight, then PCA→35 before UMAP
- This is the key: a spot's cluster identity depends on what its *neighbors* look like, not just itself

---

## Key Design Decisions

**Why not k-means?**
K-means assumes spherical, equally-sized clusters and is dominated by high-dimensional noise. These datasets have manifold structure, rare clusters, and continuous transitions — k-means handles none of that well.

**Why UMAP + Leiden over alternatives?**
This is the standard modern stack for single-cell biology clustering (used in tools like Seurat and Scanpy) because it preserves both local and global structure. Leiden on a kNN graph avoids the need to pre-specify k.

**Why scan resolutions instead of fixing cluster count?**
The problem explicitly says cluster count is unknown. Silhouette-guided resolution scanning is a principled way to select it from the data rather than guessing.

**What was broken in earlier versions (v4 → v5)?**
Part B was getting k=23 even at `resolution=1.0` because the Leiden graph was too coarse. Fix: drop `leiden_nn` from 20→15 and scan resolutions starting at 0.1 instead of 0.6. This let the algorithm find the rare clusters without over-splitting.

---

## Libraries Used

```
umap-learn
scanpy
leidenalg
igraph
anndata
scikit-learn
pandas
numpy
```

---

## How to Run

```bash
pip install umap-learn leidenalg igraph scanpy louvain

# Place train.csv, test.csv, sample_submission.csv in the same directory
# Then run all cells in the notebook top to bottom
# Output: submission.csv
```

The notebook was developed on Google Colab. The file upload cell (`files.upload()`) is Colab-specific — replace with local `pd.read_csv()` paths if running locally.

---

## Results

| Part | Clusters Found | Notes |
|------|---------------|-------|
| A | 5–15 | Strong — variance filter + UMAP captured sparse marker structure well |
| B | 6–20 | Fixed in v5 — rare clusters were being lost in v4 |
| C | 8–25 | Strong — higher-dim UMAP embedding handled continuous transitions |
| D | 8–30 | Spatial neighbor features were key to separating tissue niches |

**Final weighted ARI: 0.9891**
