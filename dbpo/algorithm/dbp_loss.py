"""
Drift-Based Policy (DBP) Loss.

This is the sole training objective for DBPO.
"""
import torch
import torch.nn.functional as F


def compute_pairwise_euclidean_distance(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Pairwise Euclidean (L2) distance between two sets of representations.

    Uses the dot-product expansion:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>

    Args:
        x : [Batch, N, Dim]
        y : [Batch, M, Dim]
        eps: numerical stability constant for sqrt at zero distance.

    Returns:
        [Batch, N, M] distance matrix.
    """
    xy_dot_product = torch.einsum("bnd,bmd->bnm", x, y)
    x_squared_norms = torch.einsum("bnd,bnd->bn", x, x)
    y_squared_norms = torch.einsum("bmd,bmd->bm", y, y)

    squared_distance = (
        x_squared_norms[:, :, None]
        + y_squared_norms[:, None, :]
        - 2 * xy_dot_product
    )
    return torch.sqrt(torch.clamp(squared_distance, min=eps))


def compute_dbp_loss(
    preds: torch.Tensor,
    pos_targets: torch.Tensor,
    neg_targets: torch.Tensor = None,
    w_pred: torch.Tensor = None,
    w_pos: torch.Tensor = None,
    w_neg: torch.Tensor = None,
    temp_schedule: tuple = (0.02, 0.05, 0.2),
) -> tuple:
    """
    Drift-Based Policy (DBP) Loss.

    Enforces generated trajectories to align with expert demonstration manifolds
    while repelling from sub-optimal modes, scaled across a multi-temperature
    schedule to smoothen the optimisation landscape.

    This function is the **only** loss used during DBPO pre-training.

    Args:
        preds       : [B, G, T*D]  — G policy samples per observation.
        pos_targets : [B, 1, T*D]  — expert demonstration(s).
        neg_targets : [B, K, T*D] or None — explicit negative modes.
        w_pred, w_pos, w_neg : optional importance weights.
        temp_schedule : tuple of temperatures R for the force simulation.

    Returns:
        loss        : [B] scalar DBP loss per batch element.
        diagnostics : dict with gradient-scale and force-magnitude metrics.
    """
    batch_size, num_preds, seq_len = preds.shape
    num_pos = pos_targets.shape[1]

    if neg_targets is None:
        neg_targets = preds.new_zeros(batch_size, 0, seq_len)
    num_neg = neg_targets.shape[1]

    if w_pred is None:
        w_pred = preds.new_ones(batch_size, num_preds)
    if w_pos is None:
        w_pos = preds.new_ones(batch_size, num_pos)
    if w_neg is None:
        w_neg = preds.new_ones(batch_size, num_neg)

    preds = preds.float()
    pos_targets = pos_targets.float()
    neg_targets = neg_targets.float()
    w_pred = w_pred.float()
    w_pos = w_pos.float()
    w_neg = w_neg.float()

    # Detach predictions to prevent self-referential gradient collapse
    anchored_preds = preds.detach()

    # Concatenate all spatial targets: [Generated, Negatives, Positives]
    all_targets = torch.cat([anchored_preds, neg_targets, pos_targets], dim=1)
    all_weights = torch.cat([w_pred, w_neg, w_pos], dim=1)

    # ------------------------------------------------------------------
    # Phase 1: Spatial Relationship & Manifold Scaling
    # ------------------------------------------------------------------
    with torch.no_grad():
        diagnostics = {}

        dists = compute_pairwise_euclidean_distance(anchored_preds, all_targets)
        weighted_dists = dists * all_weights[:, None, :]
        scale = weighted_dists.mean() / all_weights.mean()
        diagnostics["global_scale"] = scale

        dim_scale = torch.clamp(scale / (seq_len ** 0.5), min=1e-3)

        anchored_preds_norm = anchored_preds / dim_scale
        targets_norm = all_targets / dim_scale
        dists_norm = dists / torch.clamp(scale, min=1e-3)

        # ------------------------------------------------------------------
        # Phase 2: Topology Regularisation (prevent self-loops)
        # ------------------------------------------------------------------
        penalty_mask_value = 100.0
        identity_mask = torch.eye(num_preds, device=preds.device, dtype=preds.dtype)
        spatial_block_mask = F.pad(identity_mask, (0, num_neg + num_pos)).unsqueeze(0)
        dists_norm = dists_norm + spatial_block_mask * penalty_mask_value

        # ------------------------------------------------------------------
        # Phase 3: Thermodynamic Gradient Force Simulation
        # ------------------------------------------------------------------
        aggregated_forces = torch.zeros_like(anchored_preds_norm)

        for temperature in temp_schedule:
            logits = -dists_norm / temperature

            affinity_forward = torch.softmax(logits, dim=-1)
            affinity_backward = torch.softmax(logits, dim=-2)

            mutual_affinity = torch.sqrt(
                torch.clamp(affinity_forward * affinity_backward, min=1e-6)
            )
            mutual_affinity = mutual_affinity * all_weights[:, None, :]

            split_boundary = num_preds + num_neg
            affinity_neg_cluster = mutual_affinity[:, :, :split_boundary]
            affinity_pos_cluster = mutual_affinity[:, :, split_boundary:]

            sum_pos_attraction = affinity_pos_cluster.sum(dim=-1, keepdim=True)
            repulsive_coeff = -affinity_neg_cluster * sum_pos_attraction

            sum_neg_repulsion = affinity_neg_cluster.sum(dim=-1, keepdim=True)
            attractive_coeff = affinity_pos_cluster * sum_neg_repulsion

            force_coeffs = torch.cat([repulsive_coeff, attractive_coeff], dim=2)

            total_gradient_force = torch.einsum("biy,byx->bix", force_coeffs, targets_norm)
            accumulated_coeffs = force_coeffs.sum(dim=-1)
            total_gradient_force = (
                total_gradient_force - accumulated_coeffs.unsqueeze(-1) * anchored_preds_norm
            )

            force_magnitude = (total_gradient_force ** 2).mean()
            diagnostics[f"force_magnitude_T{temperature:.2f}"] = force_magnitude

            force_scale = torch.sqrt(torch.clamp(force_magnitude, min=1e-8))
            aggregated_forces = aggregated_forces + total_gradient_force / force_scale

        theoretical_target = anchored_preds_norm + aggregated_forces

    # ------------------------------------------------------------------
    # Phase 4: MSE Backpropagation Graph
    # ------------------------------------------------------------------
    trainable_preds_norm = preds / dim_scale.detach()
    spatial_diff = trainable_preds_norm - theoretical_target.detach()
    dbp_loss = (spatial_diff ** 2).mean(dim=(-1, -2))

    diagnostics = {k: v.mean() for k, v in diagnostics.items()}
    return dbp_loss, diagnostics
