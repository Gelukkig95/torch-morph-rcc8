# torch-morph-rcc8: Differentiable Fuzzy RCC-8 Spatial Logic & GPU Max-Tree Morphology in PyTorch

Pure PyTorch GPU/CPU operators for **Region Connection Calculus (RCC-8)** and **Hierarchical Mathematical Morphology (Max-Tree / Min-Tree)**.

---

## 1. Motivation: Solving the 2024?2026 Multimodal Spatial Reasoning & GPU Max-Tree Bottlenecks

1. **Overcoming Raster & Non-Convex Blob Failures in Multimodal Foundation Models (*Cohn & Blackwell, 2024; Blackwell & Cohn, 2026*)**:
   - In *Blackwell & Cohn (2026, COSIT 2026)*, state-of-the-art Vision-Language Models (VLMs) suffer severe accuracy drops when moving from vector (SVG) to raster (PNG) RCC-8 diagrams (**GPT-5.2: 87% $\rightarrow$ 72%**, **Grok 4: 83% $\rightarrow$ 53%**), perform significantly worse on irregular non-convex **blobs** than squares ($t = 18.37, p = 0.0001$), systematically confuse tangential relations (`TPP`, `TPPi`, `EC`) with non-tangential ones (`NTPP`, `NTPPi`, `DC`), and collapse to **4/32 (12.5%)** on low-contrast adjacent shades of red (`#FF0000` vs. `#FE0000`).
   - Furthermore, video generation models (**Sora 2**) correctly illustrate only **3/22** Conceptual Neighbourhood (CN) transitions, and raster image generators (**GPT Image 1.5**) achieve only **1/22** (*Blackwell & Cohn, 2026, Sec. 4.4*).
   - `torch-morph-rcc8` directly addresses the three open challenges posed in *Blackwell & Cohn (2026, Sec. 5.2?5.6)*:
     1. **Explicit Raster Topological Representation & Automated Evaluation**: `compute_rcc8_matrix` extracts exact Kuratowski closure/interior relations on $(N, H, W)$ raster masks in milliseconds on the GPU, achieving **100% accuracy** across circles, squares, triangles, and non-convex blobs.
     2. **Differentiable QSR Fine-Tuning Loss**: `DifferentiableFuzzyRCC8Loss` implements *Bloch (2005)*'s fuzzy morphological equations across all 8 JEPD relations with exact `autograd` gradients, enabling fine-tuning of lightweight vision and generative models.
2. **Integrating GPU Max-Tree Connected Operators with Deep Learning (*Faezi et al., 2024; Teeninga & Wilkinson, 2026*)**:
   - Directly implements the connected-operator hierarchy (*Salembier et al., 1998*), subtractive restitution and Hu invariants (*Urbach et al., 2007*), multi-band semantic Max-Tree vectors (*Faezi et al., 2024, Sec. VI Future Work*), and parallel GPU `leaffix`/`rootfix` and Pebay statistical moment reductions (*Teeninga & Wilkinson, 2026, Listing 1 & Sec. 4.4*).

---

## 2. Reproducible Verification Benchmarks (`experiments/verify_torch_morph_rcc8.py`)

Run `python experiments/verify_torch_morph_rcc8.py` and `pytest` (`12 passed`) to verify:

* **Experiment 1A?1C ? *Blackwell & Cohn (2026, COSIT)* 4-Shape, Adjacent-Reds (`#FF0000` vs `#FE0000`), and Continuity Benchmark**:
  - Achieves **8/8 (100%)** across all 4 shape families (`circle`, `square`, `triangle`, and non-convex `blob`), resolves the `1/255` adjacent-red contrast challenge via Max-Tree dynamics, and verifies 100% symmetric 9-step Conceptual Neighbourhood trajectories (`DC -> EC -> PO -> TPP -> NTPP -> TPP -> PO -> EC -> DC`).
* **Experiment 2 ? Multi-Region Vectorized CUDA Speedup ($30 \times 30 = 900$ pairs)**:
  - Runs in **~2.9 ms on CUDA** vs. **~14.8 ms** for CPU `Shapely` polygons (**>5x speedup**, zero Python loops, **99.78% (898/900)** agreement).
* **Experiment 3 ? Differentiable Fuzzy RCC-8 Optimization (*Bloch, 2005*)**:
  - Suppresses an illegal interior hallucination (`NTPP`) from probability **0.9526 to 0.0073** while amplifying a weak boundary contact (`EC`) from **0.2689 to 0.9734**, transitioning the crisp relation from **`NTPP` to `EC`**.
* **Experiment 4 ? Parallel GPU Max-Tree `leaffix`/`rootfix` & Pebay Moments (*Teeninga & Wilkinson, 2026*)**:
  - Matches ground-truth mean and variance to within $< 10^{-15}$ in `float64`.

---

## 3. Academic References Implemented in This Repository

* **Randell, D. A., Cui, Z., & Cohn, A. G. (1992).** *A spatial logic based on regions and connection.* KR '92, pp. 165?176.
* **Salembier, P., Oliveras, A., & Garrido, L. (1998).** *Antiextensive connected operators for image and sequence processing.* IEEE TIP, 7(4), 555?570. DOI: `10.1109/83.663500`
* **Bloch, I. (2005).** *Fuzzy spatial relationships for image processing and interpretation: a review.* Image and Vision Computing, 23(2), 89?110. DOI: `10.1016/j.imavis.2004.06.013`
* **Urbach, E. R., Roerdink, J. B. T. M., & Wilkinson, M. H. F. (2007).** *Connected shape-size pattern spectra for rotation and scale-invariant classification of gray-scale images.* IEEE TPAMI, 29(2), 272?285. DOI: `10.1109/TPAMI.2007.28`
* **Teeninga, P., Moschini, U., Trager, S. C., & Wilkinson, M. H. F. (2016).** *Statistical attribute filtering to detect faint extended astronomical sources.* Mathematical Morphology - Theory and Applications, 1(1), 100?115. DOI: `10.1515/mathm-2016-0006`
* **Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024).** *Conformal Risk Control.* JMLR, 25(146), 1?33.
* **Faezi, M. H., Peletier, R., & Wilkinson, M. H. F. (2024).** *Multi-Spectral Source-Segmentation Using Semantically-Informed Max-Trees.* IEEE Access, 12, 72288?72302. DOI: `10.1109/ACCESS.2024.3403309`
* **Cohn, A. G., & Blackwell, R. E. (2024).** *Can Large Language Models Reason about the Region Connection Calculus?* arXiv:2411.19589.
* **Blackwell, R. E., & Cohn, A. G. (2026).** *RCC-8 as a Benchmark for Diagrammatic Reasoning in Multimodal Foundation Models.* 17th International Conference on Spatial Information Theory (COSIT 2026), LIPIcs Vol. 370, Article No. 4, pp. 4:1?4:20. DOI: `10.4230/LIPIcs.COSIT.2026.4`
* **Teeninga, P. D., & Wilkinson, M. H. F. (2026).** *Shape Filtering and Max-Tree Attribute Computation on a GPU.* DGMM 2025 Proceedings, Springer LNCS 16296, pp. 276?289. DOI: `10.1007/978-3-032-09544-2_20`
