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
    if not t_a.any() or not t_b.any():
        return "DC"
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


RCC8_NAMES = ["DC", "EC", "PO", "TPP", "NTPP", "EQ", "TPPi", "NTPPi"]


def morph_dilation(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological dilation supporting 2D (H, W), 3D (N, H, W), 4D (B, C, H, W), and 5D (B, C, D, H, W) tensors."""
    if kernel_size <= 1:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    p1 = (kernel_size - 1) // 2
    p2 = (kernel_size - 1) - p1
    if t.ndim == 5:
        t_pad = F.pad(t, (p1, p2, p1, p2, p1, p2), mode="constant", value=float("-inf"))
        out = F.max_pool3d(t_pad, kernel_size=kernel_size, stride=1, padding=0)
    else:
        t_pad = F.pad(t, (p1, p2, p1, p2), mode="constant", value=float("-inf"))
        out = F.max_pool2d(t_pad, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


def morph_erosion(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    """Morphological erosion supporting 2D, 3D, 4D, and 5D tensors via duality."""
    if kernel_size <= 1:
        return x
    orig_shape = x.shape
    t = x if x.is_floating_point() else x.float()
    while t.ndim < 4:
        t = t.unsqueeze(0)
    p1 = (kernel_size - 1) // 2
    p2 = (kernel_size - 1) - p1
    if t.ndim == 5:
        x_pad = F.pad(t, (p1, p2, p1, p2, p1, p2), mode="constant", value=border_value)
        out = -F.max_pool3d(-x_pad, kernel_size=kernel_size, stride=1, padding=0)
    else:
        x_pad = F.pad(t, (p1, p2, p1, p2), mode="constant", value=border_value)
        out = -F.max_pool2d(-x_pad, kernel_size=kernel_size, stride=1, padding=0)
    return out.reshape(orig_shape)


@torch.no_grad()
def compute_rcc8_matrix(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Vectorized GPU/CPU Pairwise RCC-8 JEPD Matrix for 2D (N, H, W)/(B, N, H, W) or 3D (B, N, D, H, W) binary masks.
    Codes: 0=DC, 1=EC, 2=PO, 3=TPP, 4=NTPP, 5=EQ, 6=TPPi, 7=NTPPi.
    """
    if masks.ndim not in (3, 4, 5):
        raise ValueError(f"Expected 3D (N, H, W), 4D (B, N, H, W), or 5D (B, N, D, H, W) tensor, got {masks.shape}")

    is_unbatched = masks.ndim == 3
    if is_unbatched:
        masks = masks.unsqueeze(0)

    B, N = masks.shape[:2]
    if N == 0:
        out = torch.zeros((B, 0, 0), dtype=torch.long, device=masks.device)
        return out.squeeze(0) if is_unbatched else out

    m_bin = (masks > 0.5).to(dtype=torch.float32)
    m_flat_nd = m_bin.reshape(B * N, 1, *masks.shape[2:])

    cl_m = morph_dilation(m_flat_nd, kernel_size).to(dtype=torch.float64).reshape(B, N, -1)
    int_m = (morph_erosion(m_flat_nd, kernel_size, border_value=0.0) > 0.5).to(dtype=torch.float64).reshape(B, N, -1)
    flat_m = m_bin.to(dtype=torch.float64).reshape(B, N, -1)
    flat_m_t = flat_m.transpose(-2, -1)

    area = flat_m.sum(dim=-1, keepdim=True)
    area_t = area.transpose(-2, -1)

    inter = torch.matmul(flat_m, flat_m_t)
    dil_inter = torch.matmul(cl_m, flat_m_t) > 0.5
    sym_touch = dil_inter | dil_inter.transpose(-2, -1)

    has_ov = inter > 0.5
    a_in_b = ((inter - area).abs() < 0.5) & (area > 0.5)
    b_in_a = ((inter - area_t).abs() < 0.5) & (area_t > 0.5)

    inter_int_b = torch.matmul(flat_m, int_m.transpose(-2, -1))
    inter_int_a = torch.matmul(int_m, flat_m_t)
    a_in_int_b = ((inter_int_b - area).abs() < 0.5) & (area > 0.5)
    b_in_int_a = ((inter_int_a - area_t).abs() < 0.5) & (area_t > 0.5)

    rcc = torch.zeros((B, N, N), dtype=torch.long, device=masks.device)
    rcc[~has_ov & sym_touch] = 1
    rcc[has_ov & ~a_in_b & ~b_in_a] = 2
    rcc[a_in_b & ~b_in_a & ~a_in_int_b] = 3
    rcc[a_in_b & ~b_in_a & a_in_int_b] = 4
    rcc[a_in_b & b_in_a] = 5
    rcc[b_in_a & ~a_in_b & ~b_in_int_a] = 6
    rcc[b_in_a & ~a_in_b & b_in_int_a] = 7

    return rcc.squeeze(0) if is_unbatched else rcc


def fuzzy_rcc8_loss(
    mu: torch.Tensor,
    nu: torch.Tensor,
    kernel_size: int = 3,
    border_value: float = 0.0,
    eps: float = 1e-6,
    reduction: str = "mean",
    detach_denom: bool = False,
) -> dict[str, torch.Tensor]:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Invalid reduction: {reduction}. Choose from 'mean', 'sum', 'none'.")
    if mu.ndim == 3:
        mu = mu.unsqueeze(1)
    if nu.ndim == 3:
        nu = nu.unsqueeze(1)

    sum_dims = (-3, -2, -1) if mu.ndim == 5 else (-2, -1)
    vol_mu = mu.sum(dim=sum_dims)
    vol_nu = nu.sum(dim=sum_dims)
    if detach_denom:
        vol_mu, vol_nu = vol_mu.detach(), vol_nu.detach()
    vol_inter = (mu * nu).sum(dim=sum_dims)

    mu_int = vol_inter / (torch.minimum(vol_mu, vol_nu) + eps)
    mu_not_int = 1.0 - mu_int

    dil_mu = morph_dilation(mu, kernel_size)
    dil_nu = morph_dilation(nu, kernel_size)
    vol_dmu = dil_mu.sum(dim=sum_dims)
    vol_dnu = dil_nu.sum(dim=sum_dims)
    if detach_denom:
        vol_dmu, vol_dnu = vol_dmu.detach(), vol_dnu.detach()

    int_dmu_nu = (dil_mu * nu).sum(dim=sum_dims) / (torch.minimum(vol_dmu, vol_nu) + eps)
    int_dnu_mu = (dil_nu * mu).sum(dim=sum_dims) / (torch.minimum(vol_dnu, vol_mu) + eps)
    fuzzy_ec = mu_not_int * int_dmu_nu * int_dnu_mu

    ero_nu = morph_erosion(nu, kernel_size, border_value=border_value)
    ero_mu = morph_erosion(mu, kernel_size, border_value=border_value)

    inc_mu_nu = vol_inter / (vol_mu + eps)
    inc_nu_mu = vol_inter / (vol_nu + eps)
    fuzzy_ntpp = (mu * ero_nu).sum(dim=sum_dims) / (vol_mu + eps)
    fuzzy_ntppi = (nu * ero_mu).sum(dim=sum_dims) / (vol_nu + eps)
    fuzzy_tpp = F.relu(inc_mu_nu - fuzzy_ntpp)
    fuzzy_tppi = F.relu(inc_nu_mu - fuzzy_ntppi)
    fuzzy_eq = inc_mu_nu * inc_nu_mu
    fuzzy_po = mu_int * (1.0 - inc_mu_nu) * (1.0 - inc_nu_mu)

    raw_dict = {
        "fuzzy_DC": mu_not_int,
        "fuzzy_EC": fuzzy_ec,
        "fuzzy_PO": fuzzy_po,
        "fuzzy_TPP": fuzzy_tpp,
        "fuzzy_NTPP": fuzzy_ntpp,
        "fuzzy_TPPi": fuzzy_tppi,
        "fuzzy_NTPPi": fuzzy_ntppi,
        "fuzzy_EQ": fuzzy_eq,
    }
    if reduction == "mean":
        return {k: v.mean() for k, v in raw_dict.items()}
    if reduction == "sum":
        return {k: v.sum() for k, v in raw_dict.items()}
    return raw_dict


class DifferentiableFuzzyRCC8Loss(nn.Module):
    """PyTorch-compliant Differentiable Fuzzy RCC-8 Spatial Operator (Bloch, 2005)."""
    def __init__(
        self,
        kernel_size: int = 3,
        border_value: float = 0.0,
        eps: float = 1e-6,
        reduction: str = "mean",
        detach_denom: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_value = border_value
        self.eps = eps
        self.reduction = reduction
        self.detach_denom = detach_denom

    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        return fuzzy_rcc8_loss(
            mu, nu, self.kernel_size, self.border_value, self.eps, self.reduction, self.detach_denom
        )


class FuzzyRCC8Loss(nn.Module):
    """
    Fuzzy Morphological RCC-8 topological loss for 2D/3D segmentation networks.
    Unifies Bloch (2005) symmetric minimum-volume normalization (anchored=False)
    and Colliot, Camara, & Bloch (2006, Eq. 7) reference-object 1 - R(P) penalty
    normalization (anchored=True) across relations ('DC', 'PO', 'EC', 'NTPP', 'TPP').
    """
    def __init__(
        self,
        kernel_size: int = 3,
        border_value: float = 0.0,
        eps: float = 1e-6,
        reduction: str = "mean",
        target_relation: str = "DC",
        anchored: bool = True,
    ) -> None:
        super().__init__()
        if target_relation not in ("PO", "DC", "EC", "NTPP", "TPP"):
            raise ValueError(
                f"Invalid target_relation: {target_relation}. Choose from 'PO', 'DC', 'EC', 'NTPP', 'TPP'."
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}. Choose from 'mean', 'sum', 'none'.")
        self.kernel_size = kernel_size
        self.border_value = border_value
        self.eps = eps
        self.reduction = reduction
        self.target_relation = target_relation
        self.anchored = anchored

    def forward(self, anchor_zone: torch.Tensor, pred_prob: torch.Tensor) -> torch.Tensor:
        if anchor_zone.ndim == 3:
            anchor_zone = anchor_zone.unsqueeze(1)
        if pred_prob.ndim == 3:
            pred_prob = pred_prob.unsqueeze(1)

        a_clamped = torch.clamp(anchor_zone, 0.0, 1.0)
        if self.anchored:
            a_clamped = a_clamped.detach()
        p_clamped = torch.clamp(pred_prob, 0.0, 1.0)

        if self.target_relation == "DC":
            topo_zone = morph_dilation(a_clamped, self.kernel_size)
        elif self.target_relation == "PO":
            dil_a = morph_dilation(a_clamped, self.kernel_size)
            ero_a = morph_erosion(a_clamped, self.kernel_size, border_value=self.border_value)
            topo_zone = 1.0 - F.relu(dil_a - ero_a)
        elif self.target_relation == "EC":
            dil_a = morph_dilation(a_clamped, self.kernel_size)
            topo_zone = 1.0 - F.relu(dil_a - a_clamped)
        elif self.target_relation == "NTPP":
            ero_a = morph_erosion(a_clamped, self.kernel_size, border_value=self.border_value)
            topo_zone = 1.0 - ero_a
        else:  # "TPP"
            ero_a = morph_erosion(a_clamped, self.kernel_size, border_value=self.border_value)
            topo_zone = (1.0 - a_clamped) + 0.5 * ero_a

        sum_dims = (-3, -2, -1) if topo_zone.ndim == 5 else (-2, -1)
        vol_mu = topo_zone.sum(dim=sum_dims)
        vol_nu = p_clamped.sum(dim=sum_dims)
        vol_inter = (topo_zone * p_clamped).sum(dim=sum_dims)

        denom = (vol_mu if self.anchored else torch.minimum(vol_mu, vol_nu)) + self.eps
        loss_per_sample = vol_inter / denom

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


AnchoredFuzzyRCC8Loss = FuzzyRCC8Loss


def gpu_pebay_moment_merge(
    area: torch.Tensor,
    mean: torch.Tensor,
    m2: torch.Tensor,
    idx_targets: torch.Tensor,
    idx_sources: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teeninga & Wilkinson (2026) Sec. 4.4: Numerically stable parallel Pebay moment reduction."""
    a0, a1 = area[idx_targets], area[idx_sources]
    mu0, mu1 = mean[idx_targets], mean[idx_sources]
    m2_0, m2_1 = m2[idx_targets], m2[idx_sources]

    a_new = a0 + a1
    delta = mu1 - mu0
    safe_a = torch.clamp(a_new, min=1e-12)

    area[idx_targets] = a_new
    mean[idx_targets] = mu0 + (a1 * delta) / safe_a
    m2[idx_targets] = m2_0 + m2_1 + (a0 * a1 * (delta * delta)) / safe_a
    return area, mean, m2


def gpu_tree_leaffix_accumulate(edge_array_list: list[torch.Tensor], attributes: torch.Tensor) -> torch.Tensor:
    """Teeninga & Wilkinson (2026) Sec. 4.3: Race-condition-free GPU leaffix using index_add_."""
    for edges in edge_array_list:
        idx_targets = edges[:, 0]
        idx_sources = edges[:, 1]
        attributes.index_add_(0, idx_targets, attributes[idx_sources])
    return attributes


def gpu_tree_rootfix_reconstruct(edge_array_list: list[torch.Tensor], deltas: torch.Tensor) -> torch.Tensor:
    """Teeninga & Wilkinson (2026) Sec. 4.3 & Urbach et al. (2007): Top-down rootfix in reverse contraction order."""
    for edges in reversed(edge_array_list):
        idx_targets = edges[:, 1]
        idx_sources = edges[:, 0]
        deltas.index_add_(0, idx_targets, deltas[idx_sources])
    return deltas


class ConformalRiskCalibrator(nn.Module):
    """Finite-sample Conformal Risk Control calibrator (Angelopoulos et al., 2024)."""
    def __init__(self, target_alpha: float = 0.02):
        super().__init__()
        self.alpha = target_alpha
        self.register_buffer("lambda_hat", torch.tensor(0.50, dtype=torch.float32))

    @torch.no_grad()
    def calibrate(self, scores: torch.Tensor | np.ndarray, is_disturbance: torch.Tensor | np.ndarray) -> float:
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
        dist_t = torch.as_tensor(is_disturbance, dtype=torch.bool, device=scores_t.device)
        N = max(scores_t.numel(), 1)

        grid = torch.linspace(0.01, 0.99, 99, device=scores_t.device)
        fp_mask = (scores_t.unsqueeze(0) >= grid.unsqueeze(1)) & dist_t.unsqueeze(0)
        conformal_risk = (fp_mask.float().sum(dim=1) + 1.0) / (N + 1.0)

        valid = torch.nonzero(conformal_risk <= self.alpha, as_tuple=False)
        if valid.numel() > 0:
            self.lambda_hat.copy_(grid[valid[0, 0]])
        else:
            self.lambda_hat.copy_(grid[-1])
        return float(self.lambda_hat.item())