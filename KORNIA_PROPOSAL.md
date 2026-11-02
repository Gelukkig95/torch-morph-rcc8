# [Feature Request] Differentiable Fuzzy & Batched Crisp RCC-8 Spatial Logic Operators in `kornia.morphology` / `kornia.losses`

## 1. Motivation & Problem Statement
Enforcing **topological and spatial relationships** between predicted regions?such as requiring two structures to be externally adjacent (`EC`), disjoint (`DC`), tangentially contained (`TPP`), or prohibiting one region from floating inside another (`NTPP`)?is fundamental across medical imaging, remote sensing, and scene understanding.

However:
1. **Pixel-wise Losses & Foundation Models Lack Explicit Raster Topology**: Standard pixel-wise losses evaluate pixels independently, and recent benchmarks (*Cohn & Blackwell, 2024, arXiv:2411.19589; Blackwell & Cohn, 2026, COSIT, DOI: 10.4230/LIPIcs.COSIT.2026.4*) prove that even frontier VLMs drop from 87% (SVG) to 72% (PNG, GPT-5.2) and from 83% to 53% (Grok 4) on raster RCC-8 diagrams, struggle significantly on non-convex blobs ($t = 18.37, p = 0.0001$) and tangential relations (`TPP`, `TPPi`), collapse to 12.5% (`4/32`) on adjacent shades of red (`#FF0000` vs. `#FE0000`), and fail on Conceptual Neighbourhood video transitions (`3/22` for Sora 2).
2. **CPU Polygon Bottleneck**: Existing Python implementations of RCC-8 convert raster masks into `Shapely` polygons on the CPU via marching squares, breaking `torch.autograd` and introducing an $O(N^2)$ CPU bottleneck.

## 2. Proposed Solution & Mathematical Foundation
We propose a pure PyTorch, zero-dependency implementation of **Batched Crisp RCC-8 Matrix Classification** (`compute_rcc8_matrix`) and **Differentiable Fuzzy Morphological RCC-8 Loss** (`DifferentiableFuzzyRCC8Loss`), together with GPU parallel tree-reduction utilities:

1. **Vectorized Crisp RCC-8 Matrix (`compute_rcc8_matrix`)**:
   - Combines the 8 Jointly Exhaustive and Pairwise Disjoint (JEPD) axioms of **Randell, Cui, & Cohn (1992)** with morphological dilation $D_B(X)$ (Kuratowski closure) and dual erosion $E_B(X)$ (Kuratowski interior) from **Bloch (2005, Eq. 1, 2, 35)**.
   - Classifies all $(N, N)$ pairs of binary masks (`circle`, `square`, `triangle`, and non-convex `blob`) into (`DC`, `EC`, `PO`, `TPP`, `NTPP`, `EQ`, `TPPi`, `NTPPi`) via a single GPU matrix multiplication (`torch.matmul`) with `float64` precision protection.
2. **Differentiable Fuzzy RCC-8 Loss (`DifferentiableFuzzyRCC8Loss`)**:
   - Directly implements **Isabelle Bloch (2005)**, *Image and Vision Computing* 23(2), 89?110 (Eq. 15?16, 22, 36) across all 8 JEPD relations (`fuzzy_DC`, `fuzzy_EC`, `fuzzy_PO`, `fuzzy_TPP`, `fuzzy_NTPP`, `fuzzy_TPPi`, `fuzzy_NTPPi`, `fuzzy_EQ`).
   - Supports standard PyTorch `reduction` (`"mean"`, `"sum"`, `"none"`) and passes `torch.autograd.gradcheck` in `float64`.
3. **GPU Max-Tree Attribute Reductions (*Teeninga & Wilkinson, 2026*)**:
   - Provides race-free parallel `leaffix`/`rootfix` accumulation (`index_add_`) and numerically stable Pebay statistical moment merging (Sec. 4.3?4.4, p. 285).

## 3. Reference Implementation & Benchmarks
- Repository (`pytest` 12/12 passing): https://github.com/Gelukkig95/torch-morph-rcc8
- Requires **zero external C++ dependencies** for the core RCC-8 operators (100% `torch` and `torch.nn.functional`).
