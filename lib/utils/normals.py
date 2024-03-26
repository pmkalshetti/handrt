import torch

def orient_normals_towards_camera(xyz_ren, nrm_ren, camera_pos):
    # Ref: https://github.com/isl-org/Open3D/blob/9d88662b1c2f2a4557acf79e1208117d0b2355ee/cpp/open3d/geometry/EstimateNormals.cpp#L338
    orientation_ref = camera_pos - xyz_ren   # (n_pts, 3)
    mask_inv_normal = torch.sum(orientation_ref * nrm_ren, -1) < 0    # (n_pts,)
    nrm_ren[mask_inv_normal] = -1 * nrm_ren[mask_inv_normal]
    return nrm_ren