# AgriTopology: Differentiable Fuzzy RCC-8 Spatial Logic & Max-Tree Morphology in PyTorch

Pure PyTorch GPU/CPU operators for **Region Connection Calculus (RCC-8)** and **Hierarchical Mathematical Morphology (Max-Tree / Min-Tree)**, accompanied by reproducible benchmarks on **PhenoBench** (`n = 12,067` weed instances).

---

## 1. Motivation: Why Combine RCC-8 Spatial Logic and Max-Tree Morphology in PyTorch?

1. **Scaling Backbone Size Alone Hits a Ceiling on Boundary & Topological Errors**:
   - In the official **PhenoBench** benchmark paper by the University of Bonn (*Weyler et al., IEEE TPAMI 2024, Table III*), scaling a lightweight real-time network (**ERFNet**, Weed IoU **64.37%**, mIoU **85.98%**) up to a heavy **DeepLabV3+** backbone (39.8M parameters, Weed IoU **64.59%**, mIoU **85.97%**) improves Weed IoU by only **+0.22%p**, because standard pixel-wise cross-entropy fails to resolve crop-weed boundary confusion (`EC`) and isolated false-positive blobs (`DC`).
2. **Modern Foundation Models Struggle with Formal Topological Reasoning**:
   - As demonstrated by *Cohn & Blackwell (2024)*, even state-of-the-art LLMs achieve only **34.90%** exact accuracy (mean Jaccard $\le 0.60$) on RCC-8 compositional reasoning and fail systematically on inverse containment relations (`TPPi`, `NTPPi`).
3. **100% Differentiable Fuzzy Morphology Enables Lightweight Models to Learn Spatial Constraints**:
   - By implementing *Bloch (2005)*'s fuzzy morphological dilation, erosion, inclusion (Eq. 22), and adjacency (Eq. 35–36) directly in PyTorch (`core/diagnostics.py`), spatial logic rules can be backpropagated during training with **0.0 ms inference overhead**, empowering compact edge models without requiring massive GPU backbones.

---

## 2. Verified Empirical Findings on PhenoBench (Zero Data Leakage)

1. **Growth-Stage Spatial Contact Risk (`experiments/phenobench/experiment.py`, `n = 12,067` weeds)**:
   - Early (`05-15`): **2.7%** crop-contact rate (Train 2.8%, Val 2.6%).
   - Mid (`05-26`): **12.0%** crop-contact rate (Train 12.0%, Val 11.9%).
   - Late (`06-05`): **31.0%** crop-contact rate (Train 30.9%, Val 31.3%, $\chi^2 = 1337.12, p = 4.45 \times 10^{-291}$).
   - **94.6%** of late-stage contacts are boundary adjacency (`EC`), whereas only **2.9%** are interior containment (`NTPP`).
2. **Occlusion Disentanglement: Sobel Gradient vs. Higra Max-Tree LCA (`boundary_gradient.py`, `maxtree_higra.py`)**:
   - Classical Sobel boundary gradients fail to separate crop-crop boundaries (median **127.3**) from weed-crop boundaries (median **131.1**).
   - **Higra Max-Tree Lowest Common Ancestor (LCA)** merge altitude on Excess Green (ExG) separates true weed-crop contacts from intra-crop over-segmentation (**Cohen's $d = 0.643$, AUROC $= 0.700$**).
3. **False-Positive Blob Anatomy & Mitigation (`diagnose_errors.py`, `weed_boost_eval.py`, `train_topo_exg_loss.py`)**:
   - Out of **720** false-positive weed blobs in `SimpleUNet` (Val split), **89.0% (641)** are isolated `DC` (bare soil clutter) and **10.6% (76)** are crop-boundary `EC`.
   - Selective RCC-8 micro-blob filtering reduces false-positive blobs from **720 to 298 (-58.6%)** while boosting weed precision from **0.800 to 0.813**.
4. **Training-Free Leaf Hole Detection under Shadows (`pest_hole_final.py`, 80 mature leaves)**:
   - Under 10 shadow stains per leaf, classical area filtering suffers a mean count error of **4.14 holes**, whereas **Higra Min-Tree persistence (`attribute_dynamics >= 0.22`)** achieves a mean error of **0.54 holes (7.7x error reduction)**.

---

## 3. Academic References Implemented in This Repository

* **Randell, D. A., Cui, Z., & Cohn, A. G. (1992).** *A spatial logic based on regions and connection.* KR '92, pp. 165–176. (RCC-8 JEPD axioms in `core/rcc8.py` and `core/diagnostics.py`)
* **Bloch, I. (2005).** *Fuzzy spatial relationships for image processing and interpretation: a review.* Image and Vision Computing, 23(2), 89–110. DOI: `10.1016/j.imavis.2004.06.013` (Eq. 1–4, 15–16, 22, 35–36 implemented in `DifferentiableFuzzyRCC8Loss` and `compute_rcc8_matrix`)
* **Salembier, P., Oliveras, A., & Garrido, L. (1998).** *Antiextensive connected operators for image and sequence processing.* IEEE TIP, 7(4), 555–570. DOI: `10.1109/83.663500` (Max-Tree / Min-Tree connected component hierarchies)
* **Urbach, E. R., Roerdink, J. B. T. M., & Wilkinson, M. H. F. (2007).** *Connected shape-size pattern spectra for rotation and scale-invariant classification of gray-scale images.* IEEE TPAMI, 29(2), 272–285. DOI: `10.1109/TPAMI.2007.28` (Eq. 8 subtractive filtering rule; Eq. 42–48 first Hu moment invariant)
* **Teeninga, P., Moschini, U., Trager, S. C., & Wilkinson, M. H. F. (2016).** *Statistical attribute filtering to detect faint extended astronomical sources.* Mathematical Morphology - Theory and Applications, 1(1), 100–115. DOI: `10.1515/mathm-2016-0006`
* **Faezi, M. H., Peletier, R., & Wilkinson, M. H. F. (2024).** *Multi-Spectral Source-Segmentation Using Semantically-Informed Max-Trees.* IEEE Access, 12, 72288–72302. DOI: `10.1109/ACCESS.2024.3403309` (Eq. 3–21 multi-band semantic vectors and circularity in `agritopology/insect_topo.py`)
* **Cohn, A. G., & Blackwell, R. E. (2024).** *Can Large Language Models Reason about the Region Connection Calculus?* arXiv:2411.19589.
* **Weyler, J., Magistri, F., Marks, E., et al. (2024).** *PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain.* IEEE TPAMI. DOI: `10.1109/TPAMI.2024.3419548`
* **Teeninga, P. D., & Wilkinson, M. H. F. (2026).** *Shape Filtering and Max-Tree Attribute Computation on a GPU.* DGMM 2025 Proceedings, Springer LNCS 16296, pp. 277–290. DOI: `10.1007/978-3-032-09544-2_20` (Listing 1 parallel GPU leaffix/rootfix & Sec. 4.4 Pebay statistical moments in `core/diagnostics.py`)
