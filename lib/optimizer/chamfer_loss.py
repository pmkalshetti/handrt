# Ref: https://github.com/reyuwei/nr-reg/blob/27570ca7f12a1b4f62d952f20154f2b865028bde/loss/loss_collecter.py

import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input, _validate_chamfer_reduction_inputs
from pytorch3d.ops.knn import knn_gather, knn_points
import torch.nn.functional as F
def p3d_chamfer_distance_with_filter(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    angle_filter=None,
    distance_filter=None,
    weights=None,
    batch_reduction: str = "mean",
    point_reduction: str = "mean",
    wx = 1.0, 
    wy = 1.0,
):
    """
    Chamfer distance between two pointclouds x and y. 
    # squared

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - (
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - (
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if return_normals and angle_filter is not None:
        cham_norm_thres = 1 - torch.cos(torch.deg2rad(torch.tensor(angle_filter, dtype=torch.float).to(x.device)))
        norm_x_mask = cham_norm_x > cham_norm_thres
        norm_y_mask = cham_norm_y > cham_norm_thres

        cham_norm_x[norm_x_mask] = 0.0
        cham_norm_y[norm_y_mask] = 0.0
        cham_x[norm_x_mask] = 0.0
        cham_y[norm_y_mask] = 0.0

        x_lengths = torch.sum(~norm_x_mask)
        y_lengths = torch.sum(~norm_y_mask)
    
    if distance_filter is not None:
        dis_x_mask = cham_x > distance_filter
        dis_y_mask = cham_y > distance_filter
        cham_x[dis_x_mask] = 0.0
        cham_y[dis_y_mask] = 0.0
        
        if return_normals:
            cham_norm_x[dis_x_mask] = 0.0
            cham_norm_y[dis_y_mask] = 0.0

        x_lengths = torch.sum(~dis_x_mask)
        y_lengths = torch.sum(~dis_y_mask)

    if point_reduction is not None:
        # Apply point reduction sum
        cham_x = cham_x.sum(1)  # (N,)
        cham_y = cham_y.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
            cham_norm_y = cham_norm_y.sum(1)  # (N,)
        if point_reduction == "mean":
            cham_x /= x_lengths
            cham_y /= y_lengths
            if return_normals:
                cham_norm_x /= x_lengths
                cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x*wx + cham_y*wy
    cham_normals = cham_norm_x*wx + cham_norm_y*wy if return_normals else None

    return cham_dist, cham_normals