import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ..render import renderutils as ru


def to_homo(pts_nx3):
    if isinstance(pts_nx3, np.ndarray):
        pts_nx4 = np.ones((pts_nx3.shape[0], 4))
        pts_nx4[:, :3] = pts_nx3
        return pts_nx4
    elif isinstance(pts_nx3, torch.Tensor):
        return torch.nn.functional.pad(pts_nx3, (0,1), 'constant', 1.0)

def create_4x4_trans_mat_from_R_t(R=np.eye(3), t=np.zeros(3)):
    mat_view = np.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [      0,       0,       0,    1],
    ])
    return mat_view

def K_to_4x4_proj_mat(K):
    return np.array([
        [K[0, 0], K[0, 1], K[0, 2], 0],
        [K[1, 0], K[1, 1], K[1, 2], 0],
        [K[2, 0], K[2, 1], K[2, 2], 0],
        [      0,       0,       1, 0]
    ])

def rt_to_4x4_mat(r, t):
    R = Rotation.from_rotvec(r).as_matrix()
    return create_4x4_trans_mat_from_R_t(R, t)

def Krt_to_proj_mat(K, r, t):
    mat_extr = rt_to_4x4_mat(r, t)
    mat_intr = K_to_4x4_proj_mat(K)
    return mat_intr @ mat_extr

def apply_proj_mat(pts_nx3, proj_mat):
    pts_nx4 = to_homo(pts_nx3)
    pts_transf_homo_nx4 = pts_nx4 @ proj_mat.T
    pts_transf_nx3 = pts_transf_homo_nx4[:, :3] / pts_transf_homo_nx4[:, 3:4]
    return pts_transf_nx3


def apply_Krt(pts_nx3, K, r, t):
    return apply_proj_mat(pts_nx3, Krt_to_proj_mat(K, r, t))



def opengl_persp_proj_mat_from_K(K, near, far, img_height, img_width):
    # Ref: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
    # Matrix([
    #     [-2*K00/width, 2*K01/width, (-2*K02 + width + 2*x0)/width, 0], 
    #     [0, 2*K11/height, (-2*K12 + height + 2*y0)/height, 0], 
    #     [0, 0, (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)], 
    #     [0, 0, -1, 0]
    # ])
    # Matrix([
    #     [-2*K00/width, 2*K01/width, (-2*K02 + width + 2*x0)/width, 0], 
    #     [0, -2*K11/height, (2*K12 - height + 2*y0)/height, 0], 
    #     [0, 0, (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)], 
    #     [0, 0, -1, 0]
    # ])
    
    proj_mat = np.array([
        [2*K[0,0]/img_width,         0            , (-2*K[0,2] + img_width)/img_width  ,           0           ],
        [        0          , -2*K[1,1]/img_height, (-2*K[1,2] + img_height)/img_height,           0           ],
        [        0          ,         0           ,        -(far+near)/(far-near)      , -2*far*near/(far-near)],
        [        0          ,         0           ,                  -1                ,            0          ]
    ])
    return proj_mat

def cam_to_clip_space(pts_nx3, proj_mat_4x4):
    if isinstance(pts_nx3, np.ndarray):
        pts_nx4 = np.ones((pts_nx3.shape[0], 4))
        pts_nx4[:, :3] = pts_nx3
        return pts_nx4 @ proj_mat_4x4.T
    elif isinstance(pts_nx3, torch.Tensor):
        pts_nx4 = torch.nn.functional.pad(pts_nx3, (0,1), 'constant', 1.0)
        return torch.matmul(pts_nx4, proj_mat_4x4.transpose(0, 1))
        # return ru.xfm_points(pts_nx3[None, :, :], proj_mat_4x4[None, :, :])[0]   # (n, 4) xfm_points requires batch input
    else:
        raise "Invalid array type"

def clip_to_ndc(pts_clip):
    # return pts_clip / pts_clip[:, 3:4]
    return pts_clip / pts_clip[:, 3:4]

def clip_to_img(pts_clip, img_height, img_width):
    pts_ndc = clip_to_ndc(pts_clip)
    
    pts2d_img = (pts_ndc[:, :2] + 1) / 2    # [-1, 1] to [0, 1]
    pts2d_img[:, 0] = pts2d_img[:, 0] * img_width
    pts2d_img[:, 1] = pts2d_img[:, 1] * img_height

    return pts2d_img

def depth_to_xyz_np(depth, mask, fx, fy, cx, cy):
    # Ref: https://github.com/USTC3DV/NDR-code/blob/3bdf1ddce8e135f797e212508dce146ce8d6388e/pose_initialization/registrate.py#L42
    height, width = depth.shape

    # matrix_u = np.arange(width).repeat(height, 0).reshape(width, height).transpose().astype(np.float32)
    # matrix_v = np.arange(height).repeat(width, 0).reshape(height, width).astype(np.float32)

    matrix_v, matrix_u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    matrix_v = matrix_v.astype(np.float32)
    matrix_u = matrix_u.astype(np.float32)

    x = depth * (matrix_u - cx) / fx
    y = depth * (matrix_v - cy) / fy
    xyz = np.concatenate([np.expand_dims(x, 2), np.expand_dims(y, 2), np.expand_dims(depth, 2)], 2)
    xyz_mask = xyz[np.nonzero(mask)]

    return xyz_mask

def depth_to_xyz(depth, mask, fx, fy, cx, cy):
    height, width = depth.shape
    matrix_v, matrix_u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    matrix_v = matrix_v.float().to(depth.device)
    matrix_u = matrix_u.float().to(depth.device)

    x = depth * (matrix_u - cx) / fx
    y = depth * (matrix_v - cy) / fy

    xyz = torch.stack([x, y, depth], axis=2)
    xyz_mask = xyz[torch.nonzero(mask, as_tuple=True)]
    return xyz_mask

