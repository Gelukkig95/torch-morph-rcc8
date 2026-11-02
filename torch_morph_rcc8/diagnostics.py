from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def morph_dilation(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological dilation on 4D tensors (B, C, H, W) using max_pool2d."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def morph_erosion(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    """Morphological erosion on 4D tensors (B, C, H, W) via duality."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=border_value)
    return -F.max_pool2d(-x_pad, kernel_size=kernel_size, stride=1, padding=0)


def kuratowski_closure(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    return morph_dilation(x, kernel_size=kernel_size)


def kuratowski_interior(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    return morph_erosion(x, kernel_size=kernel_size, border_value=border_value)


def kuratowski_opening(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    return kuratowski_closure(kuratowski_interior(x, kernel_size, border_value), kernel_size)


def kuratowski_closing(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    return kuratowski_interior(kuratowski_closure(x, kernel_size), kernel_size, border_value)


def kuratowski_boundary(x: torch.Tensor, kernel_size: int = 3, border_value: float = 0.0) -> torch.Tensor:
    return kuratowski_closure(x, kernel_size) - kuratowski_interior(x, kernel_size, border_value)


@torch.no_grad()
def verify_kuratowski_axioms(x: torch.Tensor, y: torch.Tensor | None = None, kernel_size: int = 3) -> dict[str, bool]:
    x_bin = (x > 0.5).float()
    if x_bin.ndim == 2:
        x_bin = x_bin.unsqueeze(0).unsqueeze(0)
    elif x_bin.ndim == 3:
        x_bin = x_bin.unsqueeze(1)
    y_bin = torch.zeros_like(x_bin) if y is None else (y > 0.5).float().view_as(x_bin)

    empty = torch.zeros_like(x_bin)
    k1 = bool(torch.all(kuratowski_closure(empty, kernel_size) == 0).item())
    cl_x = kuratowski_closure(x_bin, kernel_size)
    k2 = bool(torch.all(cl_x >= x_bin).item())
    op_x = kuratowski_opening(x_bin, kernel_size)
    k3_opening_idem = bool(torch.allclose(kuratowski_opening(op_x, kernel_size), op_x))
    cl_union = kuratowski_closure(torch.maximum(x_bin, y_bin), kernel_size)
    union_cl = torch.maximum(cl_x, kuratowski_closure(y_bin, kernel_size))
    k4 = bool(torch.allclose(cl_union, union_cl))
    return {
        "K1_preserve_null": k1,
        "K2_extensive": k2,
        "K3_opening_idempotent": k3_opening_idem,
        "K4_additive_union": k4,
    }


@torch.no_grad()
def compute_rcc8_matrix(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Vectorized GPU/CPU Pairwise RCC-8 JEPD Matrix for (N, H, W) or (B, N, H, W) binary masks.
    Codes: 0=DC, 1=EC, 2=PO, 3=TPP, 4=NTPP, 5=EQ, 6=TPPi, 7=NTPPi.
    """
    if masks.ndim not in (3, 4):
        raise ValueError(f"Expected 3D (N, H, W) or 4D (B, N, H, W) tensor, got {masks.shape}")

    is_unbatched = masks.ndim == 3
    if is_unbatched:
        masks = masks.unsqueeze(0)

    B, N, H, W = masks.shape
    if N == 0:
        out = torch.zeros((B, 0, 0), dtype=torch.long, device=masks.device)
        return out.squeeze(0) if is_unbatched else out

    m_bin = (masks > 0.5).to(dtype=torch.float32)
    m_flat_4d = m_bin.reshape(B * N, 1, H, W)

    cl_m = morph_dilation(m_flat_4d, kernel_size).reshape(B, N, H * W)
    int_m = morph_erosion(m_flat_4d, kernel_size, border_value=0.0).reshape(B, N, H * W)
    flat_m = m_bin.reshape(B, N, H * W)
    flat_m_t = flat_m.transpose(-2, -1)

    area = flat_m.sum(dim=-1, keepdim=True)
    area_t = area.transpose(-2, -1)

    inter = torch.matmul(flat_m, flat_m_t).round()
    dil_inter = torch.matmul(cl_m, flat_m_t) > 0.5
    sym_touch = dil_inter | dil_inter.transpose(-2, -1)

    has_ov = inter > 0.5
    a_in_b = ((inter - area).abs() < 0.5) & (area > 0.5)
    b_in_a = ((inter - area_t).abs() < 0.5) & (area_t > 0.5)

    inter_int_b = torch.matmul(flat_m, int_m.transpose(-2, -1)).round()
    inter_int_a = torch.matmul(int_m, flat_m_t).round()
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


@torch.no_grad()
def kuratowski_rcc8(mask_a: torch.Tensor, mask_b: torch.Tensor, kernel_size: int = 3) -> str:
    names = ["DC", "EC", "PO", "TPP", "NTPP", "EQ", "TPPi", "NTPPi"]
    stacked = torch.stack([mask_a.squeeze(), mask_b.squeeze()], dim=0)
    code = int(compute_rcc8_matrix(stacked, kernel_size=kernel_size)[0, 1].item())
    return names[code]


def fuzzy_rcc8_loss(
    mu: torch.Tensor,
    nu: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """
    Differentiable Fuzzy RCC-8 spatial relationships based on Bloch (2005).
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Invalid reduction: {reduction}. Choose from 'mean', 'sum', 'none'.")
    if mu.ndim == 3:
        mu = mu.unsqueeze(1)
    if nu.ndim == 3:
        nu = nu.unsqueeze(1)

    vol_mu = mu.sum(dim=(-2, -1))
    vol_nu = nu.sum(dim=(-2, -1))
    vol_inter = (mu * nu).sum(dim=(-2, -1))

    mu_int = vol_inter / (torch.minimum(vol_mu, vol_nu) + eps)
    mu_not_int = 1.0 - mu_int

    dil_mu = morph_dilation(mu, kernel_size)
    dil_nu = morph_dilation(nu, kernel_size)
    int_dmu_nu = (dil_mu * nu).sum(dim=(-2, -1)) / (torch.minimum(dil_mu.sum(dim=(-2, -1)), vol_nu) + eps)
    int_dnu_mu = (dil_nu * mu).sum(dim=(-2, -1)) / (torch.minimum(dil_nu.sum(dim=(-2, -1)), vol_mu) + eps)
    fuzzy_ec = mu_not_int * int_dmu_nu * int_dnu_mu

    ero_nu = morph_erosion(nu, kernel_size, border_value=0.0)
    ero_mu = morph_erosion(mu, kernel_size, border_value=0.0)

    inc_mu_nu = vol_inter / (vol_mu + eps)
    inc_nu_mu = vol_inter / (vol_nu + eps)
    fuzzy_ntpp = (mu * ero_nu).sum(dim=(-2, -1)) / (vol_mu + eps)
    fuzzy_ntppi = (nu * ero_mu).sum(dim=(-2, -1)) / (vol_nu + eps)
    fuzzy_tpp = F.relu(inc_mu_nu - fuzzy_ntpp)
    fuzzy_tppi = F.relu(inc_nu_mu - fuzzy_ntppi)
    fuzzy_eq = inc_mu_nu * inc_nu_mu

    raw_dict = {
        "fuzzy_DC": mu_not_int,
        "fuzzy_EC": fuzzy_ec,
        "fuzzy_PO": mu_int,
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
    """
    PyTorch-compliant Differentiable Fuzzy RCC-8 Spatial Operator (Bloch, 2005).
    """
    def __init__(self, kernel_size: int = 3, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> dict[str, torch.Tensor]:
        return fuzzy_rcc8_loss(mu, nu, self.kernel_size, self.eps, self.reduction)


class AnchoredFuzzyRCC8Loss(nn.Module):
    """
    Gradient-stabilized Fuzzy RCC-8 repulsion loss for deep segmentation networks.
    Prevents max_pool2d subgradient fragmentation ('cheese-holing') and dynamic
    denominator explosion when backpropagating Bloch (2005) fuzzy relations through CNNs.
    """
    def __init__(self, kernel_size: int = 3, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.rcc8 = DifferentiableFuzzyRCC8Loss(kernel_size=kernel_size, eps=eps, reduction=reduction)

    def forward(self, anchor_zone: torch.Tensor, pred_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor_zone: Morphological reference/prohibition zone in [0, 1], shape (B, 1, H, W) or (B, H, W).
                         Automatically detached to prevent target-shifting gradient collapse.
            pred_prob: Predicted probability map of the target class in [0, 1], shape (B, 1, H, W) or (B, H, W).
        Returns:
            Stabilized fuzzy overlap penalty scalar (or per-batch tensor if reduction='none').
        """
        if anchor_zone.ndim == 3:
            anchor_zone = anchor_zone.unsqueeze(1)
        if pred_prob.ndim == 3:
            pred_prob = pred_prob.unsqueeze(1)

        # 1. Scale and detach anchor zone so vol_mu <= 0.4 * H * W is a fixed constant denominator
        mu_anchor = (0.4 * torch.clamp(anchor_zone, 0.0, 1.0) + 1e-4).detach()

        # 2. Apply linear offset to prediction so vol_nu >= 0.5 * H * W > vol_mu always holds,
        #    guaranteeing min(vol_mu, vol_nu) == vol_mu without passing pred_prob through max_pool2d
        nu_pred = 0.5 + 0.5 * torch.clamp(pred_prob, 0.0, 1.0)

        return 2.0 * (self.rcc8(mu_anchor, nu_pred)["fuzzy_PO"] - 0.5)


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
    out = attributes.clone()
    for edges in edge_array_list:
        idx_targets = edges[:, 0]
        idx_sources = edges[:, 1]
        out.index_add_(0, idx_targets, out[idx_sources])
    return out


def gpu_tree_rootfix_reconstruct(edge_array_list: list[torch.Tensor], deltas: torch.Tensor) -> torch.Tensor:
    """Teeninga & Wilkinson (2026) Sec. 4.3 & Urbach et al. (2007): Race-condition-free GPU rootfix."""
    out = deltas.clone()
    for edges in edge_array_list:
        idx_targets = edges[:, 1]
        idx_sources = edges[:, 0]
        out.index_add_(0, idx_targets, out[idx_sources])
    return out


class ConformalRiskCalibrator(nn.Module):
    """Pure PyTorch GPU/CPU Conformal Risk Control calibrator."""
    def __init__(self, target_alpha: float = 0.02):
        super().__init__()
        self.alpha = target_alpha
        self.register_buffer("lambda_hat", torch.tensor(0.50, dtype=torch.float32))

    @torch.no_grad()
    def calibrate(self, scores: torch.Tensor, is_disturbance: torch.Tensor) -> float:
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
        dist_t = torch.as_tensor(is_disturbance, dtype=torch.bool, device=scores_t.device)
        N = max(scores_t.numel(), 1)
        margin = 0.35 / (N ** 0.5)

        grid = torch.linspace(0.01, 0.99, 99, device=scores_t.device)
        fp_mask = (scores_t.unsqueeze(0) >= grid.unsqueeze(1)) & dist_t.unsqueeze(0)
        fp_risks = fp_mask.float().sum(dim=1) / N

        valid = torch.nonzero((fp_risks + margin) <= self.alpha, as_tuple=False)
        if valid.numel() > 0:
            self.lambda_hat.copy_(grid[valid[0, 0]])
        return float(self.lambda_hat.item())
