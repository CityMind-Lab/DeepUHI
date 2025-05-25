import numpy as np
from pytorch_msssim import ms_ssim
import torch
import torch.nn as nn

PI = 3.141592653589793

def ssim_loss(pred, true, data_range=2.0, size_average=True):
    """Compute SSIM loss."""
    return 1 - ms_ssim(pred, true, data_range=data_range, size_average=size_average, win_size=3)

def RSE(pred, true):
    """Compute Relative Squared Error."""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    """Compute Correlation."""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)

def MAE(pred, true):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """Compute Mean Squared Error."""
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """Compute Root Mean Squared Error."""
    return np.sqrt(MSE(pred, true))

def sMAPE(pred, true):
    """Compute Symmetric Mean Absolute Percentage Error."""
    return np.mean(np.abs(pred - true) / ((np.abs(pred) + np.abs(true)) / 2))

def MAPE(pred, true):
    """Compute Mean Absolute Percentage Error."""
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """Compute Mean Squared Percentage Error."""
    return np.mean(np.square((pred - true) / true))

def WeightedMAPE(pred, true):
    """Compute Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(pred - true)) / np.sum(true)

def max_location_distance(pred, true, k=10, distance_threshold=3):
    """
    Calculate adjusted Top-K accuracy based on distance threshold.

    Args:
        pred (torch.Tensor): Predictions of shape [B, C, T] or [C, T].
        true (torch.Tensor): Ground truth of shape [B, C, T] or [C, T].
        k (int): Number of top predictions to consider.
        distance_threshold (float): Distance threshold for correct prediction.

    Returns:
        float: Adjusted Top-K accuracy.
    """
    distance_matrix_np = np.load('./dataset/adjacency_matrix.npy')
    distance_matrix = torch.tensor(distance_matrix_np, dtype=torch.float32).to(pred.device)

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true).float()

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if true.dim() == 2:
        true = true.unsqueeze(0)

    B, T, C = pred.shape
    assert pred.shape == true.shape, "Predictions and targets must have the same shape."
    assert distance_matrix.shape == (C, C), f"Distance matrix shape should be [C, C], got {distance_matrix.shape}."

    pred_top_k_idx = torch.topk(pred, k=k, dim=2).indices
    true_max_idx = torch.argmax(true, dim=2)

    true_max_idx_expanded = true_max_idx.unsqueeze(2).expand(-1, -1, k)
    distances = torch.zeros((B, T, k), device=pred.device)
    for i in range(k):
        distances[:, :, i] = distance_matrix[pred_top_k_idx[:, :, i], true_max_idx]

    correct = distances <= distance_threshold
    correct_count = correct.sum().item()
    total_count = B * T * k
    adjusted_top_k_accuracy = correct_count / total_count

    return adjusted_top_k_accuracy

def max_step_difference(pred, true, k=10, distance_threshold=5, temporal_threshold=5):
    """
    Calculate spatio-temporal Top-K accuracy based on distance and temporal thresholds.

    Args:
        pred (torch.Tensor): Predictions of shape [B, C, T] or [C, T].
        true (torch.Tensor): Ground truth of shape [B, C, T] or [C, T].
        k (int): Number of top predictions to consider.
        distance_threshold (float): Distance threshold.
        temporal_threshold (int): Temporal threshold.

    Returns:
        float: Spatio-temporal Top-K accuracy.
    """
    distance_matrix_np = np.load('./dataset/adjacency_matrix.npy')
    distance_matrix = torch.tensor(distance_matrix_np, dtype=torch.float32).to(pred.device)

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true).float()

    pred = pred.permute(0, 2, 1)
    true = true.permute(0, 2, 1)

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if true.dim() == 2:
        true = true.unsqueeze(0)

    B, C, T = pred.shape
    assert pred.shape == true.shape, "Predictions and targets must have the same shape."
    assert distance_matrix.shape == (C, C), f"Distance matrix shape should be [C, C], got {distance_matrix.shape}."

    pred_top_k_values, pred_top_k_indices = torch.topk(pred.reshape(B, -1), k=k, dim=1)
    true_top_k_values, true_top_k_indices = torch.topk(true.reshape(B, -1), k=1, dim=1)

    pred_top_k_indices = torch.stack((pred_top_k_indices // T, pred_top_k_indices % T), dim=2)
    true_top_k_indices = torch.stack((true_top_k_indices // T, true_top_k_indices % T), dim=2)

    distances = torch.zeros((B, k), device=pred.device)
    time_diffs = torch.zeros((B, k), device=pred.device)
    for i in range(k):
        distances[:, i] = distance_matrix[pred_top_k_indices[:, i, 0], true_top_k_indices[:, 0, 0]]
        time_diffs[:, i] = torch.abs(pred_top_k_indices[:, i, 1] - true_top_k_indices[:, 0, 1])

    correct = (distances <= distance_threshold) & (time_diffs <= temporal_threshold)
    sample_correct = correct.any(dim=1)
    correct_count = sample_correct.sum().item()
    total_count = B
    spatio_temporal_top_k_accuracy = correct_count / total_count

    return spatio_temporal_top_k_accuracy

def metric(pred, true):
    """Compute various evaluation metrics."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    smape = sMAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    wmape = WeightedMAPE(pred, true)
    #avg_distance = max_location_distance(pred, true)
    #avg_step_diff = max_step_difference(pred, true)
    return mae, mse, rmse, smape, mspe, rse, corr, wmape
#    return mae, mse, rmse, smape, mspe, rse, corr, wmape, avg_distance, avg_step_diff

def amp_loss(outputs, targets):
    """Compute amplitude loss."""
    B, _, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim=-1)

    out_norm = torch.norm(outputs, dim=-1, keepdim=True)
    tgt_norm = torch.norm(targets, dim=-1, keepdim=True)

    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(T-1):], auto_corr[..., :T]], dim=-1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)

    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim=-1).real
    cross_corr = torch.cat([cross_corr[..., -(T-1):], cross_corr[..., :T]], dim=-1)
    nac_out = cross_corr / (tgt_norm * out_norm)

    loss = torch.mean(torch.abs(nac_tgt - nac_out))
    return loss

def ashift_loss(outputs, targets):
    """Compute amplitude shift loss."""
    B, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim=-1)))

def phase_loss(outputs, targets):
    """Compute phase loss."""
    B, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim=-1)
    tgt_fourier = torch.fft.fft(targets, dim=-1)
    tgt_fourier_sq = tgt_fourier.real ** 2 + tgt_fourier.imag ** 2
    mask = (tgt_fourier_sq > T).float()
    topk_indices = tgt_fourier_sq.topk(k=int(T**0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[..., 0] = 1.
    mask = torch.where(mask > 0, 1., 0.).bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    out_fourier_sq = torch.abs(out_fourier.real) + torch.abs(out_fourier.imag)
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    phase_loss = (torch.mean(zero_error) + torch.mean(ae)) / (T ** 0.5)
    return phase_loss

def MAE_loss(outputs, targets):
    """Compute Mean Absolute Error loss."""
    return torch.mean(torch.abs(outputs - targets))

def MSE_loss(outputs, targets):
    """Compute Mean Squared Error loss."""
    return torch.mean((outputs - targets) ** 2)

smooth_l1 = nn.SmoothL1Loss(reduction='none')

def spatial_loss(pred, true, penalty_factor=1.0):
    """
    Compute spatial loss.

    Args:
        pred (torch.Tensor): Predictions of shape [B, C, T].
        true (torch.Tensor): Ground truth of shape [B, C, T].
        penalty_factor (float): Penalty factor.

    Returns:
        torch.Tensor: Spatial loss.
    """
    pred = pred.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
    true = true.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

    pred_mean = pred.mean(dim=2, keepdim=True)
    true_mean = true.mean(dim=2, keepdim=True)

    pred_diff = torch.abs(pred - pred_mean)
    true_diff = torch.abs(true - true_mean)

    diff_penalty = smooth_l1(pred_diff, true_diff).mean()
    pred_var = pred.var(dim=2)
    true_var = true.var(dim=2)
    var_penalty = penalty_factor * ((true_var - pred_var) ** 2).mean()

    spatial_loss = diff_penalty
    return spatial_loss

def st_loss(pred, true, data_range=2.0, size_average=True):
    """
    Compute SSIM-based spatio-temporal loss.

    Args:
        pred (torch.Tensor): Predictions of shape [B, T, C].
        true (torch.Tensor): Ground truth of shape [B, T, C].

    Returns:
        torch.Tensor: SSIM loss.
    """
    pred = pred.permute(0, 2, 1)
    true = true.permute(0, 2, 1)
    pred = pred - pred.mean(dim=1, keepdim=True)

    pred_ssim = pred.unsqueeze(1)
    true_ssim = true.unsqueeze(1)

    ssim_loss_value = ssim_loss(pred_ssim, true_ssim, data_range=data_range)
    loss = ssim_loss_value

    return loss

def va_loss(pred, true, alpha=0.5, gamma=0.5, beta=0.5):
    """
    Compute combined loss.

    Args:
        pred (torch.Tensor): Predictions of shape [B, C, T].
        true (torch.Tensor): Ground truth of shape [B, C, T].
        alpha (float): Weight for amplitude shift loss.
        gamma (float): Weight for phase loss.
        beta (float): Weight for MAE loss.

    Returns:
        torch.Tensor: Combined loss.
    """
    outputs = pred.permute(0, 2, 1)
    targets = true.permute(0, 2, 1)

    assert not torch.isnan(outputs).any(), "NaN value detected!"
    assert not torch.isinf(outputs).any(), "Inf value detected!"

    l_ashift = ashift_loss(outputs, targets)
    l_amp = amp_loss(outputs, targets)
    l_phase = phase_loss(outputs, targets)
    l_spatial = spatial_loss(outputs, targets)
    l_mae = MAE_loss(outputs, targets)

    loss = (
        l_ashift
        + l_amp / (l_amp / l_ashift).detach()
        + l_mae / (l_mae / l_ashift).detach()
    )

    assert loss == loss, "Loss is NaN!"
    return loss
