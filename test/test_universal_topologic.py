import numpy as np
import higra as hg
import torch

from torch_morph_rcc8 import (
    kuratowski_opening,
    classify_rcc8_grid_jepd,
    compute_rcc8_matrix,
    DifferentiableFuzzyRCC8Loss,
    TopoRCC8Transformer,
    extract_higra_rcc8_tokens,
    ConformalRiskCalibrator,
    gpu_pebay_moment_merge,
    gpu_tree_leaffix_accumulate,
    gpu_tree_rootfix_reconstruct,
)


def test_tier1_kuratowski_and_rcc8_jepd_axioms():
    """Verifies Randell et al. (1992) RCC-8 JEPD axioms and Kuratowski opening idempotence."""
    rng = np.random.default_rng(42)
    valid_relations = {"DC", "EC", "PO", "EQ", "TPP", "NTPP", "TPPi", "NTPPi"}
    for _ in range(30):
        A = rng.random((32, 32)) > 0.65
        B = rng.random((32, 32)) > 0.65
        op_A = kuratowski_opening(A)
        assert np.array_equal(kuratowski_opening(op_A), op_A)
        assert classify_rcc8_grid_jepd(A, B) in valid_relations

    masks = torch.zeros((3, 16, 16))
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 6:10] = 1.0
    masks[2, 10:14, 10:14] = 1.0
    mat = compute_rcc8_matrix(masks)
    assert int(mat[0, 1].item()) == 1  # EC
    assert int(mat[0, 2].item()) == 0  # DC


def test_tier1_monotonic_persistence_invariance():
    """Verifies invariance of Salembier et al. (1998) Max-Tree dynamics under strictly monotonic intensity scaling."""
    rng = np.random.default_rng(42)
    X = rng.random((48, 48)).astype(np.float64)
    graph = hg.get_4_adjacency_graph((48, 48))
    t1, a1 = hg.component_tree_max_tree(graph, X)
    t2, a2 = hg.component_tree_max_tree(graph, 3.7 * X + 1.25)
    d1, d2 = hg.attribute_dynamics(t1, a1), hg.attribute_dynamics(t2, a2)
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(d1)), np.argsort(np.argsort(d2)))[0, 1])
    assert rank_corr > 0.999


def test_tier1_bloch_fuzzy_rcc8_gradcheck():
    """Verifies analytical gradients of Bloch (2005) Eq. (16), (22), (36) via torch.autograd.gradcheck."""
    torch.manual_seed(42)
    loss_fn = DifferentiableFuzzyRCC8Loss(kernel_size=3, border_value=0.0)
    mu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))
    nu = torch.sigmoid(torch.randn(1, 1, 6, 6, dtype=torch.float64, requires_grad=True))

    def func(a, b):
        out = loss_fn(a, b)
        return out["fuzzy_EC"] + out["fuzzy_NTPP"]

    assert torch.autograd.gradcheck(func, (mu, nu), eps=1e-6, atol=1e-4)


def test_tier2_tw2026_pebay_moments_and_higra_tokens():
    """Verifies Teeninga & Wilkinson (2026) Sec. 4.4 Pebay parallel moments and Higra-RCC8 token extraction."""
    vals = torch.tensor([1.2, -0.5, 3.4, 0.8], dtype=torch.float64)
    area = torch.ones(4, dtype=torch.float64)
    mean = vals.clone()
    m2 = torch.zeros(4, dtype=torch.float64)

    gpu_pebay_moment_merge(area, mean, m2, torch.tensor([0, 2]), torch.tensor([1, 3]))
    gpu_pebay_moment_merge(area, mean, m2, torch.tensor([0]), torch.tensor([2]))

    assert torch.isclose(mean[0], vals.mean(), atol=1e-12)
    assert torch.isclose(m2[0] / 4.0, vals.var(unbiased=False), atol=1e-12)

    yy, xx = np.mgrid[0:64, 0:64]
    scene = np.full((64, 64), 0.10, dtype=np.float32)
    body = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 25
    wing = (((yy - 32) ** 2 + (xx - 40) ** 2) <= 16) & (~body)
    scene[body], scene[wing] = 0.80, 0.28
    feats, ec_mat, _, _ = extract_higra_rcc8_tokens(scene)
    model = TopoRCC8Transformer(d_model=8)
    probs, _ = model(feats, ec_mat)
    assert probs.shape[0] == feats.shape[0]


def test_tier3_crc_small_n_and_rootfix_reversal_and_large_mask():
    """Verifies fixes for: (1) CRC N <= 306, (2) rootfix reverse order & index_add_, (3) large mask area > 2048."""
    # 1. CRC calibration with N = 100 (<= 306)
    cal = ConformalRiskCalibrator(target_alpha=0.02)
    scores = np.concatenate([np.full(80, 0.85), np.full(20, 0.15)])
    is_dist = np.concatenate([np.zeros(80, dtype=bool), np.ones(20, dtype=bool)])
    lam = cal.calibrate(scores, is_dist)
    assert abs(lam - 0.16) < 1e-6

    # 2. Multi-step tree leaffix (with shared parent) & reversed rootfix
    edge_steps = [
        torch.tensor([[1, 2], [1, 3]]),  # Step 0: leaves 2, 3 merge into node 1
        torch.tensor([[0, 1]]),          # Step 1: node 1 merges into root 0
    ]
    attrs = torch.tensor([1.0, 1.0, 1.0, 1.0])
    gpu_tree_leaffix_accumulate(edge_steps, attrs)
    assert attrs.tolist() == [4.0, 3.0, 1.0, 1.0]

    deltas = torch.tensor([10.0, 5.0, 2.0, 3.0])
    gpu_tree_rootfix_reconstruct(edge_steps, deltas)
    assert deltas.tolist() == [10.0, 15.0, 17.0, 18.0]

    # 3. Large mask (> 2048 pixels) subset check & frame-border TPP consistency
    large_masks = torch.zeros((2, 128, 128), dtype=torch.float32)
    large_masks[0, 0:60, 0:60] = 1.0   # 3,600 pixels touching top-left frame border
    large_masks[1, 0:80, 0:80] = 1.0   # 6,400 pixels containing mask 0 and sharing top-left border
    mat = compute_rcc8_matrix(large_masks)
    assert int(mat[0, 1].item()) == 3  # TPP (Tangential Proper Part)
    assert int(mat[1, 0].item()) == 6  # TPPi
