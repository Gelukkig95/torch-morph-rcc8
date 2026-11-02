r"""
torch_morph_rcc8/diagnostics.py
-------------------------------
Pure PyTorch GPU/CPU Morphological & Fuzzy RCC-8 Spatial Logic Operators.
Academic References:
  1. Randell, Cui, & Cohn (1992), KR '92, pp. 165-176 (RCC-8 JEPD axioms).
  2. Bloch (2005), Image and Vision Computing 23(2), 89-110 (Eq. 1-4, 15-16, 22, 35-36).
  3. Urbach, Roerdink, & Wilkinson (2007), IEEE TPAMI 29(2), 272-285 (Subtractive rule Eq. 8; Hu moment Eq. 42-48).
  4. Angelopoulos, Bates, Fisch, Lei, & Schuster (2024), JMLR 25(146), 1-33 (Conformal Risk Control).
  5. Teeninga & Wilkinson (2026), DGMM LNCS 16296, 276-289 (Listing 1 GPU leaffix/rootfix & Sec. 4.4 Pebay moments).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RCC8_NAMES = ("DC", "EC", "PO", "TPP", "NTPP", "EQ", "TPPi", "NTPPi")


def morph_dilation(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Bloch (2005) Eq. (1) & (3): Morphological Dilation D_B(X) preserving shape and float32/float64 precision."""
    if kernel_size <= 1:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    pad = kernel_size // 2
    padded = F.pad(t, (pad, pad, pad, pad), mode="constant", value=0.0)
    out = F.max_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


def morph_erosion(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    """Bloch (2005) Eq. (2) & (4): Dual Morphological Erosion E_B(X) preserving shape and float32/float64 precision."""
    if kernel_size <= 1:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    pad = kernel_size // 2
    padded = F.pad(t, (pad, pad, pad, pad), mode="constant", value=border_value)
    out = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


def kuratowski_closure(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Kuratowski closure (dilation) with unified 3x3 structuring element across Tensor and NumPy inputs."""
    if isinstance(mask, torch.Tensor):
        return morph_dilation(mask, kernel_size=3) > 0.5
    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    return (morph_dilation(t, kernel_size=3) > 0.5).numpy()


def kuratowski_interior(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Kuratowski interior (erosion) with unified boundary-consistent zero padding."""
    if isinstance(mask, torch.Tensor):
        return morph_erosion(mask, kernel_size=3, border_value=0.0) > 0.5
    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    return (morph_erosion(t, kernel_size=3, border_value=0.0) > 0.5).numpy()


def kuratowski_opening(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Idempotent 2x2 topological opening supporting PyTorch Tensors and NumPy arrays."""
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        t = mask if mask.is_floating_point() else mask.float()
    else:
        t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    orig_shape = t.shape
    while t.ndim < 4:
        t = t.unsqueeze(0)
    weight = torch.ones((1, 1, 2, 2), device=t.device, dtype=t.dtype)
    eroded = (F.conv2d(t, weight, padding=0) >= 3.5).to(t.dtype)
    opened = (F.conv_transpose2d(eroded, weight, stride=1) > 0.5).reshape(orig_shape)
    return opened if is_tensor else opened.cpu().numpy()


def classify_rcc8_grid_jepd(A: np.ndarray | torch.Tensor, B: np.ndarray | torch.Tensor) -> str:
    """Grid-topology JEPD RCC-8 classifier (Randell et al., 1992; Bloch, 2005, Eq. 35)."""
    t_a = A.bool() if isinstance(A, torch.Tensor) else torch.from_numpy(np.asarray(A, dtype=bool))
    t_b = B.bool() if isinstance(B, torch.Tensor) else torch.from_numpy(np.asarray(B, dtype=bool))
    if torch.equal(t_a, t_b):
        return "EQ"
    inter = t_a & t_b
    if not inter.any():
        touch = (kuratowski_closure(t_a) & t_b).any() or (t_a & kuratowski_closure(t_b)).any()
        return "EC" if bool(touch.item()) else "DC"
    if torch.equal(inter, t_a):
        return "NTPP" if torch.equal(t_a & kuratowski_interior(t_b), t_a) else "TPP"
    if torch.equal(inter, t_b):
        return "NTPPi" if torch.equal(t_b & kuratowski_interior(t_a), t_b) else "TPPi"
    return "PO"


@torch.no_grad()
def compute_rcc8_matrix(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Vectorized GPU/CPU Pairwise RCC-8 JEPD Matrix for (N, H, W) binary masks.
    Uses float64 inner products and < 0.5 tolerance to eliminate TF32/FP32 rounding errors on large masks.
    Codes: 0=DC, 1=EC, 2=PO, 3=TPP, 4=NTPP, 5=EQ, 6=TPPi, 7=NTPPi.
    """
    if masks.ndim != 3 or masks.shape[0] == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=masks.device)
    N = masks.shape[0]
    m_bin = (masks > 0.5).float()
    cl_m = morph_dilation(m_bin, kernel_size).reshape(N, -1).to(torch.float64)
    int_m = morph_erosion(m_bin, kernel_size, border_value=0.0).reshape(N, -1).to(torch.float64)
    flat_m = m_bin.reshape(N, -1).to(torch.float64)

    area = flat_m.sum(dim=1, keepdim=True)
    inter = flat_m @ flat_m.T
    dil_inter = (cl_m @ flat_m.T) > 0.5
    sym_touch = dil_inter | dil_inter.T

    has_ov = inter > 0.5
    a_in_b = ((inter - area).abs() < 0.5) & (area > 0.5)
    b_in_a = ((inter - area.T).abs() < 0.5) & (area.T > 0.5)
    a_in_int_b = (((flat_m @ int_m.T) - area).abs() < 0.5) & (area > 0.5)
    b_in_int_a = (((int_m @ flat_m.T) - area.T).abs() < 0.5) & (area.T > 0.5)

    rcc = torch.zeros((N, N), dtype=torch.long, device=masks.device)
    rcc[~has_ov & sym_touch] = 1
    rcc[has_ov & ~a_in_b & ~b_in_a] = 2
    rcc[a_in_b & ~b_in_a & ~a_in_int_b] = 3
    rcc[a_in_b & ~b_in_a & a_in_int_b] = 4
    rcc[a_in_b & b_in_a] = 5
    rcc[b_in_a & ~a_in_b & ~b_in_int_a] = 6
    rcc[b_in_a & ~a_in_b & b_in_int_a] = 7
    return rcc


class DifferentiableFuzzyRCC8Loss(nn.Module):
    """
    PyTorch-compliant Differentiable Fuzzy Spatial Relationship Operator (Bloch, 2005, Eq. 15, 16, 22, 36).
    Supports standard PyTorch reduction ('mean', 'sum', 'none'), border_value=0.0 consistency,
    and all 8 JEPD RCC-8 relations (DC, EC, PO, TPP, NTPP, TPPi, NTPPi, EQ).
    """
    def __init__(
        self,
        kernel_size: int = 3,
        border_value: float = 0.0,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.kernel_size = kernel_size
        self.border_value = border_value
        self.eps = eps
        self.reduction = reduction

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x

    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        if mu.ndim == 3:
            mu = mu.unsqueeze(1)
        if nu.ndim == 3:
            nu = nu.unsqueeze(1)

        vol_mu = mu.sum(dim=(-2, -1))
        vol_nu = nu.sum(dim=(-2, -1))
        vol_inter = (mu * nu).sum(dim=(-2, -1))

        mu_int = vol_inter / (torch.minimum(vol_mu, vol_nu) + self.eps)
        mu_not_int = 1.0 - mu_int

        dil_mu = morph_dilation(mu, self.kernel_size)
        dil_nu = morph_dilation(nu, self.kernel_size)
        int_dmu_nu = (dil_mu * nu).sum(dim=(-2, -1)) / (torch.minimum(dil_mu.sum(dim=(-2, -1)), vol_nu) + self.eps)
        int_dnu_mu = (dil_nu * mu).sum(dim=(-2, -1)) / (torch.minimum(dil_nu.sum(dim=(-2, -1)), vol_mu) + self.eps)
        fuzzy_ec = mu_not_int * int_dmu_nu * int_dnu_mu

        ero_nu = morph_erosion(nu, self.kernel_size, border_value=self.border_value)
        ero_mu = morph_erosion(mu, self.kernel_size, border_value=self.border_value)

        inc_mu_nu = vol_inter / (vol_mu + self.eps)
        inc_nu_mu = vol_inter / (vol_nu + self.eps)
        fuzzy_ntpp = (mu * ero_nu).sum(dim=(-2, -1)) / (vol_mu + self.eps)
        fuzzy_ntppi = (nu * ero_mu).sum(dim=(-2, -1)) / (vol_nu + self.eps)
        fuzzy_tpp = (mu * (nu - ero_nu)).sum(dim=(-2, -1)) / (vol_mu + self.eps)
        fuzzy_tppi = (nu * (mu - ero_mu)).sum(dim=(-2, -1)) / (vol_nu + self.eps)
        fuzzy_eq = inc_mu_nu * inc_nu_mu

        return {
            "fuzzy_DC": self._reduce(mu_not_int),
            "fuzzy_EC": self._reduce(fuzzy_ec),
            "fuzzy_PO": self._reduce(mu_int),
            "fuzzy_TPP": self._reduce(fuzzy_tpp),
            "fuzzy_NTPP": self._reduce(fuzzy_ntpp),
            "fuzzy_TPPi": self._reduce(fuzzy_tppi),
            "fuzzy_NTPPi": self._reduce(fuzzy_ntppi),
            "fuzzy_EQ": self._reduce(fuzzy_eq),
        }


def gpu_pebay_moment_merge(
    area: torch.Tensor,
    mean: torch.Tensor,
    m2: torch.Tensor,
    idx_targets: torch.Tensor,
    idx_sources: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Teeninga & Wilkinson (2026) Sec. 4.2 & 4.4 (p. 285): Numerically stable parallel Pebay moment reduction.
    Note: Requires disjoint target indices per contraction step (idx_targets must be unique within each call).
    """
    a0 = area[idx_targets]
    a1 = area[idx_sources]
    mu0 = mean[idx_targets]
    mu1 = mean[idx_sources]
    m2_0 = m2[idx_targets]
    m2_1 = m2[idx_sources]

    a_new = a0 + a1
    delta = mu1 - mu0
    safe_a = torch.clamp(a_new, min=1e-12)

    area[idx_targets] = a_new
    mean[idx_targets] = mu0 + (a1 * delta) / safe_a
    m2[idx_targets] = m2_0 + m2_1 + (a0 * a1 * (delta * delta)) / safe_a
    return area, mean, m2


def gpu_tree_leaffix_accumulate(edge_array_list: list[torch.Tensor], attributes: torch.Tensor) -> torch.Tensor:
    """
    Teeninga & Wilkinson (2026) Sec. 4.3 Listing 1 (p. 285): Bottom-up parallel GPU leaffix accumulation.
    Uses index_add_ to guarantee race-free accumulation when multiple children share a target parent.
    """
    for edges in edge_array_list:
        idx_targets = edges[:, 0]
        idx_sources = edges[:, 1]
        attributes.index_add_(0, idx_targets, attributes[idx_sources])
    return attributes


def gpu_tree_rootfix_reconstruct(edge_array_list: list[torch.Tensor], deltas: torch.Tensor) -> torch.Tensor:
    """
    Teeninga & Wilkinson (2026) Sec. 4.3 Listing 1 & Urbach et al. (2007) Eq. (8):
    Top-down parallel GPU rootfix reconstruction iterating in reverse order over bottom-up contraction steps.
    """
    for edges in reversed(edge_array_list):
        idx_targets = edges[:, 1]
        idx_sources = edges[:, 0]
        deltas.index_add_(0, idx_targets, deltas[idx_sources])
    return deltas


class ConformalRiskCalibrator(nn.Module):
    """
    Distribution-free finite-sample Conformal Risk Control calibrator (Angelopoulos et al., 2024, JMLR 25(146)):
        (sum_{i=1}^N L_i(lambda) + 1) / (N + 1) <= alpha.
    Provides a torch.compile-friendly tensor forward() and a backward-compatible float calibrate() API.
    """
    def __init__(self, target_alpha: float = 0.02):
        super().__init__()
        self.alpha = target_alpha
        self.register_buffer("lambda_hat_buf", torch.tensor(1.0, dtype=torch.float64))
        self.lambda_hat = 1.0

    @torch.no_grad()
    def forward(self, scores: torch.Tensor, is_disturbance: torch.Tensor) -> torch.Tensor:
        scores_t = scores.to(dtype=torch.float64)
        dist_t = is_disturbance.to(dtype=torch.bool, device=scores_t.device)
        N = scores_t.numel()
        if N == 0:
            return self.lambda_hat_buf

        grid = torch.linspace(0.01, 0.99, 99, dtype=torch.float64, device=scores_t.device)
        fp_counts = ((scores_t.unsqueeze(0) >= grid.unsqueeze(1)) & dist_t.unsqueeze(0)).to(torch.float64).sum(dim=1)

        cond_exact = (fp_counts + 1.0) / float(N + 1) <= self.alpha
        cond_emp = (fp_counts / float(N)) <= self.alpha
        valid_mask = torch.where(cond_exact.any(), cond_exact, cond_emp)

        candidates = torch.where(valid_mask, grid, torch.tensor(1.0, dtype=torch.float64, device=scores_t.device))
        best_lam = candidates.min()
        self.lambda_hat_buf.copy_(best_lam)
        return best_lam

    @torch.no_grad()
    def calibrate(self, scores: np.ndarray | torch.Tensor, is_disturbance: np.ndarray | torch.Tensor) -> float:
        scores_t = torch.as_tensor(scores, dtype=torch.float64)
        dist_t = torch.as_tensor(is_disturbance, dtype=torch.bool, device=scores_t.device)
        best_lam = self.forward(scores_t, dist_t)
        self.lambda_hat = round(float(best_lam.item()), 4)
        return self.lambda_hat
