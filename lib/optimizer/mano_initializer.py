import torch
import numpy as np
import nvdiffrast.torch as dr
import cv2 as cv
import plotly.graph_objects as go
from lpips import LPIPS
from tqdm import tqdm
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_log_map,
)
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes, estimate_pointcloud_normals, sample_farthest_points, knn_gather, knn_points, corresponding_points_alignment
# from pytorch3d.ops.knn import knn_gather, knn_points
import torch.nn.functional as F
import subprocess
import open3d as o3d
from scipy.spatial.transform import Rotation
from pathlib import Path
import open3d as o3d

from .. import utils
from ..mano.mano import Mano
from ..render import optixutils as ou
from ..render import renderutils as ru
from ..render import bilateral_denoiser, light, texture, render, regularizer, util, mesh, material, obj
from .chamfer_loss import p3d_chamfer_distance_with_filter

logger = utils.get_logger(__name__)

def reproj_error(lmk2d_pred, lmk2d_data):
    return np.mean(np.linalg.norm(lmk2d_pred - lmk2d_data, axis=-1), axis=-1)

def normalize_image_points(u, v, resolution):
    """
    normalizes u, v coordinates from [0 ,image_size] to [-1, 1]
    :param u:
    :param v:
    :param resolution:
    :return:
    """
    u = 2 * (u - resolution[1] / 2.0) / resolution[1]
    v = 2 * (v - resolution[0] / 2.0) / resolution[0]
    return u, v

def cnt_area(cnt):
    area = cv.contourArea(cnt)
    return area

def obtain_mask_from_verts_img(img_raw, verts_img, faces):
    # create convex polygon from vertices in image space
    # Ref: https://github.com/SeanChenxy/HandAvatar/blob/3b1c70b9d8d829bfcea1255743daea6dd8ed0b1d/segment/seg_interhand2.6m_from_mano.py#L210
    mask = np.zeros_like(img_raw)
    for f in faces:
        triangle = np.array([
            [verts_img[f[0]][0], verts_img[f[0]][1]],
            [verts_img[f[1]][0], verts_img[f[1]][1]],
            [verts_img[f[2]][0], verts_img[f[2]][1]],
        ])
        cv.fillConvexPoly(mask, triangle, (255, 255, 255))
    
    # filter mask
    if mask.max()<20:
        print(f"mask is all black")
        return None
    mask_bool = mask[..., 0]==255
    sel_img = img_raw[mask_bool].mean(axis=-1)
    if sel_img.max()<20:
        print(f"sel_img is all black")
        return None
    sel_img = np.bitwise_and(sel_img>10, sel_img<200)
    mask_bool[mask_bool] = sel_img.astype('int32')
    mask = mask * mask_bool[..., None]
    contours, _ = cv.findContours(mask[..., 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=cnt_area, reverse=True)
    poly = contours[0].transpose(1, 0, 2).astype(np.int32)
    poly_mask = np.zeros_like(mask)
    poly_mask = cv.fillPoly(poly_mask, poly, (1,1,1))
    mask = mask * poly_mask

    return mask


class LmkSegDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, lmk_seg_dir):
        self.cfg = cfg

        if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
            num_frames = len(list(sorted(Path(f"{lmk_seg_dir}/rgba_seg").glob("*.png"))))
        elif cfg.MODE == "rgbd" or cfg.MODE == "depth":
            num_frames = len(list(sorted(Path(f"{lmk_seg_dir}/rgba_reg_seg").glob("*.png"))))
        # list_id_data = range(0, len(list_path_to_rgba_seg))

        self.list_id_frame = []
        if cfg.MODE == "rgb" or cfg.MODE == "rgbd":
            self.list_rgba = []
        if cfg.MODE == "rgbd" or cfg.MODE == "depth":
            self.list_depth = []
        if cfg.MODE == "rgb" or cfg.MODE == "rgbd":
            self.list_K_rgb = []        # intrinsic mat
        if cfg.MODE == "rgbd" or cfg.MODE == "depth":
            self.list_K_depth = []        # intrinsic mat
        if cfg.MODE == "rgb" or cfg.MODE == "rgbd":
            self.list_lmk2d_rgb = []    # 2D landmark in rgb image space
            if self.cfg.MODE == "rgb":
                self.list_lmk3d_cam = []    # 3D landmark in rgb camera space

        if cfg.MODE == "rgbd" or cfg.MODE == "depth":
            self.list_lmk2d_depth = []    # 2D landmark in depth image space
        
        for id_frame in tqdm(range(num_frames), desc="Appending to list"):
            self.list_id_frame.append(id_frame)
            
            if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                rgba = cv.cvtColor(cv.imread(f"{lmk_seg_dir}/rgba_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
                rgba = torch.from_numpy(rgba).float()/255
                self.list_rgba.append(rgba)
            elif cfg.MODE == "rgbd" and cfg.USE_REG_RGB:
                rgba = cv.cvtColor(cv.imread(f"{lmk_seg_dir}/rgba_reg_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
                rgba = torch.from_numpy(rgba).float()/255
                self.list_rgba.append(rgba)
                # rgba[..., 0:3] = util.srgb_to_rgb(rgba[..., 0:3])

            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                depth = np.load(f"{lmk_seg_dir}/depth_seg_npy/{id_frame:05d}.npy")   # (H, W) in mm
                self.list_depth.append(torch.from_numpy(depth).float())

            if self.cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                lmk2d_rgb = np.load(f"{lmk_seg_dir}/lmk2d_rgb/{id_frame:05d}.npy")   # (21, 2) in image space
                self.list_lmk2d_rgb.append(torch.from_numpy(lmk2d_rgb).float())

                if self.cfg.MODE == "rgb":
                    lmk3d_cam = np.load(f"{lmk_seg_dir}/lmk3d_cam/{id_frame:05d}.npy")   # (21, 3) in rgb camera space
                    self.list_lmk3d_cam.append(torch.from_numpy(lmk3d_cam).float())

            elif cfg.MODE == "rgbd" and cfg.USE_REG_RGB:
                lmk2d_rgb = np.load(f"{lmk_seg_dir}/lmk2d_rgb_reg/{id_frame:05d}.npy")   # (21, 2) in image space
                self.list_lmk2d_rgb.append(torch.from_numpy(lmk2d_rgb).float())

            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                lmk2d_depth = np.load(f"{lmk_seg_dir}/lmk2d_depth/{id_frame:05d}.npy")   # (21, 2) in image space
                self.list_lmk2d_depth.append(torch.from_numpy(lmk2d_depth).float())

            if self.cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                K_rgb = np.load(f"{lmk_seg_dir}/K_rgb/{id_frame:05d}.npy")  # (3, 3)
                self.list_K_rgb.append(torch.from_numpy(K_rgb).float())
            elif cfg.MODE == "rgbd" and cfg.USE_REG_RGB:
                K_rgb = np.load(f"{lmk_seg_dir}/K_depth/{id_frame:05d}.npy")  # (3, 3)
                self.list_K_rgb.append(torch.from_numpy(K_rgb).float())

            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                K_depth = np.load(f"{lmk_seg_dir}/K_depth/{id_frame:05d}.npy")  # (3, 3)
                self.list_K_depth.append(torch.from_numpy(K_depth).float())

        self.num_data = len(self.list_id_frame)

    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, id_frame):
        data_dict = {}
        data_dict["id_frame"] = self.list_id_frame[id_frame]

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            data_dict["rgba"] = self.list_rgba[id_frame].to(self.cfg.DEVICE)
        
        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            data_dict["depth"] = self.list_depth[id_frame].to(self.cfg.DEVICE)

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            data_dict["lmk2d_rgb"] = self.list_lmk2d_rgb[id_frame].to(self.cfg.DEVICE)
            if self.cfg.MODE == "rgb":
                data_dict["lmk3d_cam"] = self.list_lmk3d_cam[id_frame].to(self.cfg.DEVICE)

        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            data_dict["lmk2d_depth"] = self.list_lmk2d_depth[id_frame].to(self.cfg.DEVICE)

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            data_dict["K_rgb"] = self.list_K_rgb[id_frame].to(self.cfg.DEVICE)

        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            data_dict["K_depth"] = self.list_K_depth[id_frame].to(self.cfg.DEVICE)

        return data_dict
        
        # return {
        #     "id_frame": self.list_id_frame[id_frame],
        #     "rgba": self.list_rgba[id_frame].to(self.cfg.DEVICE),
        #     "depth": self.list_depth[id_frame].to(self.cfg.DEVICE),
        #     "lmk2d_rgb": self.list_lmk2d_rgb[id_frame].to(self.cfg.DEVICE),
        #     "lmk2d_depth": self.list_lmk2d_depth[id_frame].to(self.cfg.DEVICE),
        #     "K_rgb": self.list_K_rgb[id_frame].to(self.cfg.DEVICE),
        #     "K_depth": self.list_K_depth[id_frame].to(self.cfg.DEVICE)
        # }
    
class ManoInitializer:
    def __init__(self, cfg, lmk_seg_dir):
        self.cfg = cfg
        self.dataset = LmkSegDataset(cfg, lmk_seg_dir)
        self.num_data = len(self.dataset)

        # ==============================================================================================
        #  Create trainable mesh (with fixed topology)
        # ==============================================================================================
        self.mano = Mano(self.cfg.MANO).to(self.cfg.DEVICE)

        self.beta = torch.zeros(10, requires_grad=True, device=self.cfg.DEVICE)
        self.offset = torch.zeros(len(self.mano.v_template), 3, requires_grad=True, device=self.cfg.DEVICE)
        self.list_global_rot = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        self.list_global_transl = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        self.list_hand_pose = [torch.zeros(15*3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]

        # ==============================================================================================
        #  Setup nvdiffrast and optix
        # ==============================================================================================
        self.glctx = dr.RasterizeGLContext(device=self.cfg.DEVICE) # Context for training
        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()
            self.image_loss_fn = lambda img, ref: ru.image_loss(img, ref, loss="l1", tonemapper="log_srgb")
        
        # ==============================================================================================
        #  Setup additional required data
        # ==============================================================================================
        self.create_projection_matrices_from_K()

        self.base_mesh = mesh.Mesh(
            v_pos=self.mano.v_template, t_pos_idx=self.mano.faces,
            v_tex=self.mano.verts_uvs, t_tex_idx=self.mano.faces_uvs,
            # material=self.mat
        )
        self.base_mesh = mesh.auto_normals(self.base_mesh)

        tex_np = cv.cvtColor(cv.imread(self.cfg.MANO.HTML_KD), cv.COLOR_BGR2RGB)
        self.tex = torch.from_numpy(tex_np.astype(np.float32)/255).to(self.cfg.DEVICE)

    @torch.no_grad()
    def create_projection_matrices_from_K(self):
        
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            self.list_mv_rgb = []
            self.list_mvp_rgb = []
            self.list_campos_rgb = []

        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            self.list_mv_depth = []
            self.list_mvp_depth = []
            self.list_campos_depth = []

        # self.T_rgb_to_depth = get_color_to_depth_stereo_extrinsic_transform_4x4()
        if "guesswho" in self.cfg.EXPT_NAME:
            self.T_depth_to_rgb = np.eye(4, dtype=np.float32)
        else:
            if self.cfg.MODE == "rgbd":
                if not self.cfg.USE_REG_RGB:
                    self.T_depth_to_rgb = utils.get_calibrated_kinectv2_T_ir_to_rgb()
                else:
                    self.T_depth_to_rgb = np.eye(4, dtype=np.float32)


        for data in self.dataset:
            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                rgba = data["rgba"].cpu().numpy()
                rgb_height, rgb_width = rgba.shape[:2]
                K_rgb = data["K_rgb"].cpu().numpy()

                # ==============================================================================================
                #  Create projection matrix (rgb)
                # ==============================================================================================
                proj_mat_rgb = utils.opengl_persp_proj_mat_from_K(K_rgb, self.cfg.CAM_NEAR_FAR[0], self.cfg.CAM_NEAR_FAR[1], rgb_height, rgb_width)
                proj_mat_rgb = torch.from_numpy(proj_mat_rgb).float().to(self.cfg.DEVICE)
                # data["depth"] uses kinect camera convention (kinect (based on plot, after fliplr) x: right,   y: down,    z: forward)
                # rotate to use opengl camera convention (opengl/blender/nvdiffrast:                x: right,   y: up,      z: backward)
                R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                if self.cfg.MODE == "rgbd":
                    T_gl_to_rgb =  T_kin_to_gl @ self.T_depth_to_rgb    # kin (short for kinect) is same as rgb
                elif self.cfg.MODE == "rgb":
                    T_gl_to_rgb =  T_kin_to_gl
                mv_rgb = torch.from_numpy(T_gl_to_rgb).float().to(self.cfg.DEVICE)
                mvp_rgb = proj_mat_rgb @ mv_rgb
                campos_rgb = torch.linalg.inv(mv_rgb)[:3, 3]    # (3,)
                self.list_mv_rgb.append(mv_rgb)
                self.list_mvp_rgb.append(mvp_rgb)
                self.list_campos_rgb.append(campos_rgb)

            if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
                depth = data["depth"].cpu().numpy()
                depth_height, depth_width = depth.shape[:2]
                K_depth = data["K_depth"].cpu().numpy()

                # ==============================================================================================
                #  Create projection matrix (depth)
                # ==============================================================================================
                proj_mat_depth = utils.opengl_persp_proj_mat_from_K(K_depth, self.cfg.CAM_NEAR_FAR[0], self.cfg.CAM_NEAR_FAR[1], depth_height, depth_width)
                proj_mat_depth = torch.from_numpy(proj_mat_depth).float().to(self.cfg.DEVICE)
                # data["depth"] uses kinect camera convention (kinect (based on plot, after fliplr) x: right,   y: down,    z: forward)
                # rotate to use opengl camera convention (opengl/blender/nvdiffrast:                x: right,   y: up,      z: backward)
                R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                # T_gl_to_depth = self.T_rgb_to_depth @ T_kin_to_gl    # kin (short for kinect) is same as rgb
                T_gl_to_depth = T_kin_to_gl    # kin (short for kinect) is same as rgb
                mv_depth = torch.from_numpy(T_gl_to_depth).float().to(self.cfg.DEVICE)
                mvp_depth = proj_mat_depth @ mv_depth
                campos_depth = torch.linalg.inv(mv_depth)[:3, 3]    # (3,)
                self.list_mv_depth.append(mv_depth)
                self.list_mvp_depth.append(mvp_depth)
                self.list_campos_depth.append(campos_depth)

    def rasterize_mesh(self, v_clip, color=[177/255, 189/255, 180/255]):
        # nvdiffrast to rasterize
        rast, _ = dr.rasterize(self.glctx, v_clip[None, :, :], self.mano.faces.int(), self.cfg.IMG_RES)
        alpha = torch.clamp(rast[..., 3], 0, 1) # rast[:, :, 3] == 0  # Field triangle_id is the triangle index, offset by one. Pixels where no triangle was rasterized will receive a zero in all channels.
        
        texc, _ = dr.interpolate(self.mano.verts_uvs[None, :, :], rast, self.mano.faces_uvs.int())
        rgb = dr.texture(self.tex[None, :, :, :], texc, filter_mode="linear")
        # v_cols = torch.ones_like(v_ndc[:, :3]) * torch.tensor(color, device=self.cfg.DEVICE)
        # rgb, _ = dr.interpolate(v_cols[None, :, :], rast, self.mano.faces.int())
        
        # (H, W, 3), (H, W)
        return rgb[0], alpha[0]
    
    def render_depth(self, v_cam, mvp):
        v_clip = utils.cam_to_clip_space(v_cam, mvp)
        # nvdiffrast to rasterize
        rast, _ = dr.rasterize(self.glctx, v_clip[None, :, :], self.mano.faces.int(), self.cfg.IMG_RES)
        
        alpha = torch.clamp(rast[..., 3], 0, 1) # rast[:, :, 3] == 0  # Field triangle_id is the triangle index, offset by one. Pixels where no triangle was rasterized will receive a zero in all channels.
        alpha = alpha[0]    # (H, W)
        mask = alpha > 0.5

        depth, _ = dr.interpolate(v_cam[None, :, 2:3].contiguous(), rast, self.mano.faces.int()) # (1, H, W, 1)
        # with torch.no_grad():
        #     depth_wo_aa = depth.clone()
        depth = dr.antialias(depth, rast, v_clip, self.mano.faces.int())    # (1, H, W, 1)  # Note: this is necessary for gradients wrt silhouette
        depth = depth[0, :, :, 0]   # (H, W)

        # m to mm
        # depth = 1000 * depth

        # invert Z
        # depth = -1 * depth

        # # antialiasing introduces artifacts around silhouette
        # threshold_min_depth = depth_wo_aa[depth_wo_aa > 0].min()
        # mask_aa = depth_wo_aa > 0
        # mask_aa 


        return depth, mask
    
    def get_mesh(self, vert):
        opt_mesh = mesh.Mesh(v_pos=vert, base=self.base_mesh)
        # with torch.no_grad():
        #     ou.optix_build_bvh(self.optix_ctx, opt_mesh.v_pos.contiguous(), opt_mesh.t_pos_idx.int(), rebuild=1)
        opt_mesh = mesh.auto_normals(opt_mesh)
        opt_mesh = mesh.compute_tangents(opt_mesh)
        return opt_mesh
    
    # def render_normal(self, v_nrm, mvp):


    def forward_mano(self, data):
        id_frame = data["id_frame"]

        mano_out = self.mano(
            betas=self.beta[None, :],
            offsets=self.offset[None, :, :],
            # offsets=None,
            global_orient=torch.zeros((1, 3), device=self.cfg.DEVICE),
            transl=torch.zeros((1, 3), device=self.cfg.DEVICE),
            hand_pose=self.list_hand_pose[id_frame][None, :],
            flat_hand_mean=True
        )
        vert, lmk3d = mano_out.vertices[0], mano_out.joints[0]

        global_rot_mat = so3_exp_map(self.list_global_rot[id_frame][None, :])[0]
        global_transl = self.list_global_transl[id_frame]

        vert = vert @ global_rot_mat.t() + global_transl
        lmk3d = lmk3d @ global_rot_mat.t() + global_transl

        return vert, lmk3d

    @torch.no_grad()
    def initialize_global_pose_with_rgb(self, data):
        id_frame = data["id_frame"]

        self.list_global_rot[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        self.list_global_transl[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        vert, lmk3d = self.forward_mano(data)

        # ==============================================================================================
        #  Use landmark to get initial global pose
        # ==============================================================================================
        # we'll use 3D lmk in rgb camera space
        # use palm landmarks
        id_lmk_palm = [
            0,                  # root
            13, 1, 4, 10, 7,    # mcp: thumb, index, middle, ring, pinky
        ]
        lmk3d_mano_palm = lmk3d[id_lmk_palm].cpu().numpy()
        lmk3d_data_palm = data["lmk3d_cam"][id_lmk_palm].cpu().numpy()

        pc_lmk3d_data_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_data_palm))
        pc_lmk3d_mano_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_mano_palm))
        corres = np.repeat(np.arange(len(lmk3d_data_palm))[:, None], 2, axis=1)
        corres = o3d.utility.Vector2iVector(corres)
        trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T_glob_global = trans_est.compute_transformation(pc_lmk3d_mano_palm, pc_lmk3d_data_palm, corres)

        # update parameters
        R_glob = torch.from_numpy(T_glob_global[:3, :3]).float().to(self.cfg.DEVICE)
        t_glob = torch.from_numpy(T_glob_global[:3, 3]).float().to(self.cfg.DEVICE)
        r_glob = so3_log_map(R_glob[None, :, :])[0]
        self.list_global_rot[id_frame].data = r_glob
        self.list_global_transl[id_frame].data = t_glob

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            vert, lmk3d = self.forward_mano(data)

            rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_rgb[data["id_frame"]]))
            composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
            fig = go.Figure(go.Image(z=composite_img))
            fig.show()
            exit()
        

    @torch.no_grad()
    def initialize_global_pose_with_depth(self, data):        
        id_frame = data["id_frame"]

        self.list_global_rot[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        self.list_global_transl[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        vert, lmk3d = self.forward_mano(data)

        # ==============================================================================================
        #  Use landmark to get initial global pose
        # ==============================================================================================
        # we'll use 2D lmk and use the corresponding Z from depth image
        # Assumption: depth is available at these 2D locations and there is no occlusion among landmark points
        # We'll use this function only for first frame (or some keyframes) so this assumption will generally hold

        # use palm landmarks
        id_lmk_palm = [
            0,                  # root
            13, 1, 4, 10, 7,    # mcp: thumb, index, middle, ring, pinky
        ]
        lmk3d_mano_palm = lmk3d[id_lmk_palm].cpu().numpy()
        
        K_depth = data["K_depth"].cpu().numpy()
        fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
        lmk2d_data = data["lmk2d_depth"].cpu().numpy()    # (21, 2)
        lmk2d_data_palm = lmk2d_data[id_lmk_palm]   # (n_palm_lmk, 2)
        lmk_z_data_palm = data["depth"].cpu().numpy()[lmk2d_data_palm[:, 1].astype(int), lmk2d_data_palm[:, 0].astype(int)]  # (n_palm_lmk,)
        lmk_z_data_palm /= 1000 # mm to m
        assert np.all(lmk_z_data_palm > 0.001), f"Invalid depth at some landmark"
        lmk_x_data_palm = lmk_z_data_palm * (lmk2d_data_palm[:, 0] - cx_depth) / fx_depth # (n_palm_lmk,)
        lmk_y_data_palm = lmk_z_data_palm * (lmk2d_data_palm[:, 1] - cy_depth) / fy_depth # (n_palm_lmk,)
        lmk3d_data_palm = np.stack([lmk_x_data_palm, lmk_y_data_palm, lmk_z_data_palm], axis=1) # (n_palm_lmk, 3)
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts3d_depth = lmk3d_data_palm
            pts3d_rgb = utils.apply_proj_mat(pts3d_depth, self.T_depth_to_rgb)
            pts2d_rgb_proj = utils.apply_Krt(pts3d_rgb, data["K_rgb"].cpu().numpy(), r=np.zeros(3), t=np.zeros(3))[:, :2]
            color_plot_proj = utils.draw_pts_on_img((data["rgba"][:, :, :3].cpu().numpy()*255).astype(np.uint8), pts2d_rgb_proj, radius=10)
            fig = go.Figure(go.Image(z=color_plot_proj))
            fig.show()
            exit()

        pc_lmk3d_data_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_data_palm))
        pc_lmk3d_mano_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_mano_palm))
        corres = np.repeat(np.arange(len(lmk3d_data_palm))[:, None], 2, axis=1)
        corres = o3d.utility.Vector2iVector(corres)
        trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T_glob_global = trans_est.compute_transformation(pc_lmk3d_mano_palm, pc_lmk3d_data_palm, corres)

        # update parameters
        R_glob = torch.from_numpy(T_glob_global[:3, :3]).float().to(self.cfg.DEVICE)
        t_glob = torch.from_numpy(T_glob_global[:3, 3]).float().to(self.cfg.DEVICE)
        r_glob = so3_log_map(R_glob[None, :, :])[0]
        self.list_global_rot[id_frame].data = r_glob
        self.list_global_transl[id_frame].data = t_glob

        # forward
        vert, lmk3d = self.forward_mano(data)

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts3d_depth = lmk3d.cpu().numpy()
            pts3d_rgb = utils.apply_proj_mat(pts3d_depth, self.T_depth_to_rgb)
            pts2d_rgb_proj = utils.apply_Krt(pts3d_rgb, data["K_rgb"].cpu().numpy(), r=np.zeros(3), t=np.zeros(3))[:, :2]
            color_plot_proj = utils.draw_pts_on_img((data["rgba"][:, :, :3].cpu().numpy()*255).astype(np.uint8), pts2d_rgb_proj, radius=10)
            fig = go.Figure(go.Image(z=color_plot_proj))
            fig.show()
            exit()

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            # pts_pos = data["pc_pos"].cpu().numpy(); color_pos = "green"
            fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
            xyz_data = utils.depth_to_xyz_np(data["depth"].cpu().numpy(), data["depth"].cpu().numpy()>0, fx_depth, fy_depth, cx_depth, cy_depth)
            xyz_data /= 1000 # mm to m
            pts_pos = xyz_data; color_pos = "green"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            # vert_np = utils.apply_proj_mat(vert_np, T_glob_global)
            vert_np = vert.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
            
            pts_pos = lmk3d_data_palm
            scat_lmk3d_data_palm = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=10, color="brown"), showlegend=False)
            
            fig = go.Figure([scat_pc_data_pos, plotlymesh, scat_lmk3d_data_palm])
            # fig = go.Figure([scat_pc_data_pos, scat_lmk3d_data_palm])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()
            exit()
        
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            vert_np = vert.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
            
            vert_rgb_np = utils.apply_proj_mat(vert_np, self.T_depth_to_rgb)
            plotlymesh_rgb = go.Mesh3d(x=vert_rgb_np[:, 0], y=vert_rgb_np[:, 1], z=vert_rgb_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="magenta", flatshading=True, hoverinfo="none", opacity=0.5)

            fig = go.Figure([plotlymesh, plotlymesh_rgb])
            # fig = go.Figure([scat_pc_data_pos, scat_lmk3d_data_palm])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()

            exit()

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_depth[data["id_frame"]]))
            depth_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(data["depth"].cpu().numpy(), self.cfg.PLOT.DEPTH_NEAR, self.cfg.PLOT.DEPTH_FAR)
            composite_img = (utils.alpha_composite(depth_cm/255, torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
            fig = go.Figure(go.Image(z=composite_img))
            fig.show()
            exit()

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_rgb[data["id_frame"]]))
            composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
            fig = go.Figure(go.Image(z=composite_img))
            fig.show()
            exit()
            
        # ==============================================================================================
        #  Refine pose: ICP to register mesh to depth pointcloud
        # ==============================================================================================
        ## Matching: Projection-based
        depth_ren, mask_ren = self.render_depth(vert, self.list_mvp_depth[data["id_frame"]])    # (H, W)
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            fig = go.Figure(go.Heatmap(z=depth_ren.cpu().numpy()))
            fig.update_layout(width=depth_ren.shape[1], height=depth_ren.shape[0])
            fig.update_yaxes(autorange='reversed')
            fig.show()
            exit()

        K_depth = data["K_depth"]
        fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
        mask_depth_data = data["depth"] > 0
        mask_depth_common = mask_ren & mask_depth_data  # use correspondences present in both depth maps (this has the effect of boundary point rejection)
        # xyz_ren = utils.depth_to_xyz(depth_ren, mask_depth_common, fx_depth, fy_depth, cx_depth, cy_depth)
        # xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_common, fx_depth, fy_depth, cx_depth, cy_depth)
        xyz_ren = utils.depth_to_xyz(depth_ren, mask_ren, fx_depth, fy_depth, cx_depth, cy_depth)
        xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth)
        xyz_data = xyz_data / 1000
        
        ## Rejection: boundary, distance, normal
        if False:
            pc_ren = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz_ren))
            pc_ren.estimate_normals()
            pc_ren.orient_normals_towards_camera_location()
            pc_ren_bnd, pc_ren_mask_bnd = pc_ren.compute_boundary_points(radius=1)
            DEBUG_LOCAL = False
            if DEBUG_LOCAL:
                pts_pos = pc_ren.point.positions.numpy(); color_pos = "blue"
                scat_ren = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
                
                pts_pos = pc_ren_bnd.point.positions.numpy(); color_pos = "red"
                scat_ren_bnd = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
                
                fig = go.Figure([scat_ren, scat_ren_bnd])
                fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                fig.show()
                exit()
        
        nrm_ren = estimate_pointcloud_normals(xyz_ren[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_ren = utils.orient_normals_towards_camera(xyz_ren, nrm_ren, camera_pos=torch.zeros(3, device=xyz_ren.device))
        nrm_data = estimate_pointcloud_normals(xyz_data[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_data = utils.orient_normals_towards_camera(xyz_data, nrm_data, camera_pos=torch.zeros(3, device=xyz_data.device))
        
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = xyz_data.cpu().numpy(); color_pos = "green"; pts_nrm = nrm_data.cpu().numpy(); color_nrm = "red"
            scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            # list_scat_data_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            pts_pos = xyz_ren.cpu().numpy(); color_pos = "yellow"; pts_nrm = nrm_ren.cpu().numpy(); color_nrm = "magenta"
            scat_ren_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            # list_scat_ren_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            pts_pos = lmk3d_data_palm
            scat_lmk3d_data = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=5, color="brown"), showlegend=False)
            
            # fig = go.Figure([scat_data_pos, *list_scat_data_nrm, scat_ren_pos, *list_scat_ren_nrm, scat_lmk3d_data])
            fig = go.Figure([scat_data_pos, scat_ren_pos, scat_lmk3d_data])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()
            exit()

        
        # pc_data = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_data.cpu().numpy()))
        # pc_data.normals = o3d.utility.Vector3dVector(nrm_data.cpu().numpy())
        
        # pc_ren = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ren.cpu().numpy()))
        # pc_ren.normals = o3d.utility.Vector3dVector(nrm_ren.cpu().numpy())
        
        # corres = np.repeat(np.arange(len(xyz_ren))[:, None], 2, axis=1)
        # corres = o3d.utility.Vector2iVector(corres)
        # trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # T_glob_icp = trans_est.compute_transformation(pc_ren, pc_data, corres)

        # reduce points for computational performance
        xyz_data, arr_id_selected_pt = sample_farthest_points(xyz_data[None, :, :], K=self.cfg.INIT.N_SAMPLES_ON_PC)
        xyz_data = xyz_data[0]; arr_id_selected_pt = arr_id_selected_pt[0]
        nrm_data = nrm_data[arr_id_selected_pt]

        xyz_ren, arr_id_selected_pt = sample_farthest_points(xyz_ren[None, :, :], K=self.cfg.INIT.N_SAMPLES_ON_PC)
        xyz_ren = xyz_ren[0]; arr_id_selected_pt = arr_id_selected_pt[0]
        nrm_ren = nrm_ren[arr_id_selected_pt]

        # find corresponding points (both ways)
        xyz_ren_nn = knn_points(xyz_ren[None, :, :], xyz_data[None, :, :], K=1)
        xyz_data_corr = knn_gather(xyz_data[None, :, :], xyz_ren_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1  (resulting shape is (num_xyz_ren, 3))
        nrm_data_corr = knn_gather(nrm_data[None, :, :], xyz_ren_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1  (resulting shape is (num_xyz_ren, 3))
        
        xyz_data_nn = knn_points(xyz_data[None, :, :], xyz_ren[None, :, :], K=1)
        xyz_ren_corr = knn_gather(xyz_ren[None, :, :], xyz_data_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1   (resulting shape is (num_xyz_data, 3))
        nrm_ren_corr = knn_gather(nrm_ren[None, :, :], xyz_data_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1   (resulting shape is (num_xyz_data, 3))

        # plot correspondences
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            # Ref for color names: https://i.stack.imgur.com/xRwWi.png
            pts_pos = xyz_data.cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts_pos = xyz_ren.cpu().numpy(); color_pos = "darkorange"
            scat_pc_ren_icp_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts1 = xyz_ren.cpu().numpy(); pts2 = xyz_data_corr.cpu().numpy()
            skip = None; color_line = "darkorange"
            list_scat_xyz_ren_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            pts1 = xyz_data.cpu().numpy(); pts2 = xyz_ren_corr.cpu().numpy()
            skip = None; color_line = "dodgerblue"
            list_scat_xyz_data_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos, *list_scat_xyz_ren_nn, *list_scat_xyz_data_nn])
            # fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), yaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), zaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), ) )
            fig.show()

            exit()
        
        # calculate position and normal distances
        cham_pos_ren = xyz_ren_nn.dists[0, :, 0]    # (n_xyz_ren,)
        assert torch.allclose(cham_pos_ren, torch.sum((xyz_ren - xyz_data_corr)**2, axis=1)), "distance calculation (ren-to-data) is incorrect"
        cham_pos_data = xyz_data_nn.dists[0, :, 0]    # (n_xyz_data,)
        assert torch.allclose(cham_pos_data, torch.sum((xyz_data - xyz_ren_corr)**2, axis=1)), "distance calculation (data-to-ren) is incorrect"

        cham_nrm_ren = 1 - F.cosine_similarity(nrm_ren, nrm_data_corr, dim=1, eps=1e-6)    # (n_xyz_ren,)
        cham_nrm_data = 1 - F.cosine_similarity(nrm_data, nrm_ren_corr, dim=1, eps=1e-6)    # (n_xyz_data,)

        xyz_ren_orig = xyz_ren.clone()
        xyz_data_orig = xyz_data.clone()

        # filter correspondences based on angle between normal
        cham_nrm_threshold = 1 - torch.cos(torch.deg2rad(torch.tensor(self.cfg.INIT.CHAMF_NRM_THRESH, dtype=torch.float).to(self.cfg.DEVICE)))
        nrm_ren_mask = cham_nrm_ren < cham_nrm_threshold
        nrm_data_mask = cham_nrm_data < cham_nrm_threshold

        xyz_ren = xyz_ren[nrm_ren_mask]
        nrm_ren = nrm_ren[nrm_ren_mask]
        xyz_data_corr = xyz_data_corr[nrm_ren_mask]
        cham_pos_ren = cham_pos_ren[nrm_ren_mask]
        cham_nrm_ren = cham_nrm_ren[nrm_ren_mask]
        
        xyz_data = xyz_data[nrm_data_mask]
        nrm_data = nrm_data[nrm_data_mask]
        xyz_ren_corr = xyz_ren_corr[nrm_data_mask]
        cham_pos_data = cham_pos_data[nrm_data_mask]
        cham_nrm_data = cham_nrm_data[nrm_data_mask]

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            # Ref for color names: https://i.stack.imgur.com/xRwWi.png
            pts_pos = xyz_data_orig.cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts_pos = xyz_ren_orig.cpu().numpy(); color_pos = "darkorange"
            scat_pc_ren_icp_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts1 = xyz_ren.cpu().numpy(); pts2 = xyz_data_corr.cpu().numpy()
            skip = None; color_line = "darkorange"
            list_scat_xyz_ren_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            pts1 = xyz_data.cpu().numpy(); pts2 = xyz_ren_corr.cpu().numpy()
            skip = None; color_line = "dodgerblue"
            list_scat_xyz_data_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos, *list_scat_xyz_ren_nn, *list_scat_xyz_data_nn])
            # fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), yaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), zaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), ) )
            fig.show()

            exit()
        
        # filter correspondences based on euclidean distance
        pos_ren_mask = cham_pos_ren < self.cfg.INIT.CHAMF_POS_THRESH**2
        pos_data_mask = cham_pos_data < self.cfg.INIT.CHAMF_POS_THRESH**2

        xyz_ren = xyz_ren[pos_ren_mask]
        nrm_ren = nrm_ren[pos_ren_mask]
        xyz_data_corr = xyz_data_corr[pos_ren_mask]
        cham_pos_ren = cham_pos_ren[pos_ren_mask]
        cham_nrm_ren = cham_nrm_ren[pos_ren_mask]
        
        xyz_data = xyz_data[pos_data_mask]
        nrm_data = nrm_data[pos_data_mask]
        xyz_ren_corr = xyz_ren_corr[pos_data_mask]
        cham_pos_data = cham_pos_data[pos_data_mask]
        cham_nrm_data = cham_nrm_data[pos_data_mask]

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            # Ref for color names: https://i.stack.imgur.com/xRwWi.png
            pts_pos = xyz_data_orig.cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts_pos = xyz_ren_orig.cpu().numpy(); color_pos = "darkorange"
            scat_pc_ren_icp_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts1 = xyz_ren.cpu().numpy(); pts2 = xyz_data_corr.cpu().numpy()
            skip = None; color_line = "darkorange"
            list_scat_xyz_ren_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            pts1 = xyz_data.cpu().numpy(); pts2 = xyz_ren_corr.cpu().numpy()
            skip = None; color_line = "dodgerblue"
            list_scat_xyz_data_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos, *list_scat_xyz_ren_nn, *list_scat_xyz_data_nn])
            # fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), yaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), zaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), ) )
            fig.show()

            exit()
        
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = xyz_data_orig.cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            vert, lmk3d = self.forward_mano(data)
            vert_np = vert.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="darkorange", flatshading=True, hoverinfo="none", opacity=0.5)
           
            # fig = go.Figure([scat_pc_data_pos, *list_scat_pc_data_nrm, plotlymesh])
            fig = go.Figure([scat_pc_data_pos, plotlymesh])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()

            exit()

        # Umeyama
        simi_tranf = corresponding_points_alignment(xyz_ren[None, :, :], xyz_data_corr[None, :, :])
        # the estimated rotation matrix is multiplied from right side in pytorch3d (ref: look at _apply_similarity_transform at https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html)
        # as per our convention, we multiply transformation matrix from left side, so transpose the estimated rotation matrix
        R_glob_icp = simi_tranf.R.transpose(1, 2)[0].cpu().numpy()
        t_glob_icp = simi_tranf.T[0].cpu().numpy()
        T_glob_icp = utils.create_4x4_trans_mat_from_R_t(R_glob_icp, t_glob_icp)

        # update parameters
        T_glob = T_glob_icp @ T_glob_global
        R_glob = torch.from_numpy(T_glob[:3, :3]).float().to(self.cfg.DEVICE)
        t_glob = torch.from_numpy(T_glob[:3, 3]).float().to(self.cfg.DEVICE)
        r_glob = so3_log_map(R_glob[None, :, :])[0]
        self.list_global_rot[id_frame].data = r_glob
        self.list_global_transl[id_frame].data = t_glob

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = xyz_data_orig.cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            vert, lmk3d = self.forward_mano(data)
            vert_np = vert.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="darkorange", flatshading=True, hoverinfo="none", opacity=0.5)
           
            # fig = go.Figure([scat_pc_data_pos, *list_scat_pc_data_nrm, plotlymesh])
            fig = go.Figure([scat_pc_data_pos, plotlymesh])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()

            exit()
        

    @torch.no_grad()
    def initialize_from_previous_frame(self, id_frame):
        self.list_global_rot[id_frame].data = self.list_global_rot[id_frame-1].detach().clone().data
        self.list_global_transl[id_frame].data = self.list_global_transl[id_frame-1].detach().clone().data
        self.list_hand_pose[id_frame].data = self.list_hand_pose[id_frame-1].detach().clone().data
    
    @torch.no_grad()
    def initialize_global_pose_from_previous_frame(self, id_frame):
        self.list_global_rot[id_frame].data = self.list_global_rot[id_frame-1].detach().clone().data
        self.list_global_transl[id_frame].data = self.list_global_transl[id_frame-1].detach().clone().data
    
    @torch.no_grad()
    def initialize_hand_pose_from_previous_frame(self, id_frame):
        self.list_hand_pose[id_frame].data = self.list_hand_pose[id_frame-1].detach().clone().data

    def setup_optimizer(self):
        def lr_schedule(iter, fraction):
            warmup_iter = 0
            if iter < warmup_iter:
                return iter / warmup_iter 
            return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs
        
        if self.cfg.INIT.OPTIMIZE_SHAPE:
            params_shape = [
                {"params": [self.beta], "lr": self.cfg.INIT.LR_BETA},
                {"params": [self.offset], "lr": self.cfg.INIT.LR_OFFSET},
            ]
            self.optimizer_shape = torch.optim.SGD(params_shape, lr = self.cfg.INIT.LR_GEOM)
            self.scheduler_shape = torch.optim.lr_scheduler.LambdaLR(self.optimizer_shape, lr_lambda=lambda x: lr_schedule(x, 0.9))
    
        if self.cfg.INIT.OPTIMIZE_POSE:
            params_pose = [
                {"params": self.list_hand_pose, "lr": self.cfg.INIT.LR_HAND_POSE},
                {"params": self.list_global_rot, "lr": self.cfg.INIT.LR_GLOBAL_ROT},
                {"params": self.list_global_transl, "lr": self.cfg.INIT.LR_GLOBAL_TRANSL}
            ]
            self.optimizer_pose = torch.optim.SGD(params_pose, lr = self.cfg.INIT.LR_GEOM)
            self.scheduler_pose = torch.optim.lr_scheduler.LambdaLR(self.optimizer_pose, lr_lambda=lambda x: lr_schedule(x, 0.9))

    def compute_chamfer_loss(self, data, vert):
        # chamfer vs hausdorff vs earth mover: slide 34 onwards at https://3ddl.cs.princeton.edu/2016/slides/su.pdf
       
        ## Matching: Projection-based
        depth_ren, mask_ren = self.render_depth(vert, self.list_mvp_depth[data["id_frame"]])    # (H, W)

        # depth to pointcloud
        K_depth = data["K_depth"]
        fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
        mask_depth_data = data["depth"] > 0
        xyz_ren = utils.depth_to_xyz(depth_ren, mask_ren, fx_depth, fy_depth, cx_depth, cy_depth)
        
        # antialiasing introduces artifacts around silhouette
        # obtain mask before antialiasing for plotting
        with torch.no_grad():
            threshold_min_depth = vert[:, 2].min()
            # mask_xyz_ren_aa = xyz_ren[:, 2] > threshold_min_depth
        # xyz_ren = xyz_ren[mask_xyz_ren_aa]

        xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth)
        xyz_data = xyz_data / 1000

        # chamfer distance from ren to data (Ref: https://github.com/reyuwei/nr-reg/blob/27570ca7f12a1b4f62d952f20154f2b865028bde/loss/loss_collecter.py#L49)
        # chamf_pos_dists = torch.sum(torch.square(xyz_data - xyz_ren), axis=1)   # (n_pts,)
        # # filter based on distance
        # dist_fil = chamf_pos_dists > self.cfg.INIT.CHAMF_DIST_THRESH
        # chamf_pos_dists[dist_fil] = 0.0
        # # sum across points
        # loss_chamf_pos = torch.sum(chamf_pos_dists)
        # return loss_chamf_pos, torch.zeros(1, device=self.cfg.DEVICE)
    
        # estimate normal
        nrm_ren = estimate_pointcloud_normals(xyz_ren[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_ren = utils.orient_normals_towards_camera(xyz_ren, nrm_ren, camera_pos=torch.zeros(3, device=xyz_ren.device))
        nrm_data = estimate_pointcloud_normals(xyz_data[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_data = utils.orient_normals_towards_camera(xyz_data, nrm_data, camera_pos=torch.zeros(3, device=xyz_data.device))

        # loss_pos, loss_nrm = p3d_chamfer_distance_with_filter(
        #     xyz_ren[None, :, :], xyz_data[None, :, :],
        #     x_normals=nrm_ren[None, :, :], y_normals=nrm_data[None, :, :],
        #     angle_filter=90, distance_filter=0.01,
        # )
        # return loss_pos, loss_nrm

        # reduce points for computational performance
        xyz_data_selected, arr_id_selected_pt = sample_farthest_points(xyz_data[None, :, :], K=self.cfg.INIT.N_SAMPLES_ON_PC)
        # xyz_data = xyz_data[0]; arr_id_selected_pt = arr_id_selected_pt[0]
        arr_id_selected_pt = arr_id_selected_pt[0]
        xyz_data = xyz_data[arr_id_selected_pt]
        nrm_data = nrm_data[arr_id_selected_pt]

        xyz_ren_selected, arr_id_selected_pt = sample_farthest_points(xyz_ren[None, :, :], K=self.cfg.INIT.N_SAMPLES_ON_PC)
        # xyz_ren = xyz_ren[0]; arr_id_selected_pt = arr_id_selected_pt[0]
        arr_id_selected_pt = arr_id_selected_pt[0]
        xyz_ren = xyz_ren[arr_id_selected_pt]
        nrm_ren = nrm_ren[arr_id_selected_pt]

        # find corresponding points (both ways)
        xyz_ren_nn = knn_points(xyz_ren[None, :, :], xyz_data[None, :, :], K=1)
        xyz_data_corr = knn_gather(xyz_data[None, :, :], xyz_ren_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1  (resulting shape is (num_xyz_ren, 3))
        nrm_data_corr = knn_gather(nrm_data[None, :, :], xyz_ren_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1  (resulting shape is (num_xyz_ren, 3))
        
        xyz_data_nn = knn_points(xyz_data[None, :, :], xyz_ren[None, :, :], K=1)
        xyz_ren_corr = knn_gather(xyz_ren[None, :, :], xyz_data_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1   (resulting shape is (num_xyz_data, 3))
        nrm_ren_corr = knn_gather(nrm_ren[None, :, :], xyz_data_nn.idx)[0, :, 0, :]    # first 0: batch; third 0: K=1   (resulting shape is (num_xyz_data, 3))
           
        # calculate position and normal distances
        # cham_pos_ren = xyz_ren_nn.dists[0, :, 0]    # (n_xyz_ren,)
        # cham_pos_data = xyz_data_nn.dists[0, :, 0]    # (n_xyz_data,)
        cham_pos_ren = torch.sum((xyz_ren - xyz_data_corr)**2, axis=1)
        cham_pos_data = torch.sum((xyz_data - xyz_ren_corr)**2, axis=1)

        cham_nrm_ren = 1 - F.cosine_similarity(nrm_ren, nrm_data_corr, dim=1, eps=1e-6)    # (n_xyz_ren,)
        cham_nrm_data = 1 - F.cosine_similarity(nrm_data, nrm_ren_corr, dim=1, eps=1e-6)    # (n_xyz_data,)

        with torch.no_grad():
            xyz_ren_orig = xyz_ren.clone()
            # xyz_ren_orig = xyz_ren_orig[xyz_ren_orig[:, 2] > threshold_min_depth]
            xyz_data_orig = xyz_data.clone()

        # filter correspondences based on angle between normal
        if self.cfg.INIT.CHAMF_FILTER_NRM_DIST:
            cham_nrm_threshold = 1 - torch.cos(torch.deg2rad(torch.tensor(self.cfg.INIT.CHAMF_NRM_THRESH, dtype=torch.float).to(self.cfg.DEVICE)))
            nrm_ren_mask = cham_nrm_ren < cham_nrm_threshold
            nrm_data_mask = cham_nrm_data < cham_nrm_threshold

            xyz_ren = xyz_ren[nrm_ren_mask]
            nrm_ren = nrm_ren[nrm_ren_mask]
            xyz_data_corr = xyz_data_corr[nrm_ren_mask]
            cham_pos_ren = cham_pos_ren[nrm_ren_mask]
            cham_nrm_ren = cham_nrm_ren[nrm_ren_mask]
            
            xyz_data = xyz_data[nrm_data_mask]
            nrm_data = nrm_data[nrm_data_mask]
            xyz_ren_corr = xyz_ren_corr[nrm_data_mask]
            cham_pos_data = cham_pos_data[nrm_data_mask]
            cham_nrm_data = cham_nrm_data[nrm_data_mask]

        # filter correspondences based on euclidean distance
        if self.cfg.INIT.CHAMF_FILTER_POS_DIST:
            pos_ren_mask = cham_pos_ren < self.cfg.INIT.CHAMF_POS_THRESH**2
            pos_data_mask = cham_pos_data < self.cfg.INIT.CHAMF_POS_THRESH**2

            xyz_ren = xyz_ren[pos_ren_mask]
            nrm_ren = nrm_ren[pos_ren_mask]
            xyz_data_corr = xyz_data_corr[pos_ren_mask]
            cham_pos_ren = cham_pos_ren[pos_ren_mask]
            cham_nrm_ren = cham_nrm_ren[pos_ren_mask]
            
            xyz_data = xyz_data[pos_data_mask]
            nrm_data = nrm_data[pos_data_mask]
            xyz_ren_corr = xyz_ren_corr[pos_data_mask]
            cham_pos_data = cham_pos_data[pos_data_mask]
            cham_nrm_data = cham_nrm_data[pos_data_mask]

        # DEBUG_LOCAL = False
        # if DEBUG_LOCAL:
        if self.DEBUG_CHAMF_LOSS:
            # Ref for color names: https://i.stack.imgur.com/xRwWi.png
            pts_pos = xyz_data_orig.detach().cpu().numpy(); color_pos = "dodgerblue"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts_pos = xyz_ren_orig.detach().cpu().numpy(); color_pos = "darkorange"
            scat_pc_ren_icp_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            pts1 = xyz_ren.detach().cpu().numpy(); pts2 = xyz_data_corr.detach().cpu().numpy()
            skip = None; color_line = "darkorange"
            list_scat_xyz_ren_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            pts1 = xyz_data.detach().cpu().numpy(); pts2 = xyz_ren_corr.detach().cpu().numpy()
            skip = None; color_line = "dodgerblue"
            list_scat_xyz_data_nn = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_line, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts1[::skip], pts2[::skip])]
           
            fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos, *list_scat_xyz_ren_nn, *list_scat_xyz_data_nn])
            # fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), yaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), zaxis=dict(showbackground=False, showticklabels=False, title="", visible=False), ) )
            fig.update_layout(scene_camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=-2.0)))
            fig.write_html(f"{self.log_chamf_dir}/{self.id_frame:05d}.html")

            # fig.show()
            # exit()
        

        # reduce along points
        loss_pos_ren = cham_pos_ren.mean()
        loss_nrm_ren = cham_nrm_ren.mean()

        loss_pos_data = cham_pos_data.mean()
        loss_nrm_data = cham_nrm_data.mean()
        
        # add loss in both directions (ren<-->data)
        loss_pos = self.cfg.INIT.CHAMF_W_REN*loss_pos_ren + self.cfg.INIT.CHAMF_W_DATA*loss_pos_data
        loss_nrm = self.cfg.INIT.CHAMF_W_REN*loss_nrm_ren + self.cfg.INIT.CHAMF_W_DATA*loss_nrm_data

        return loss_pos, loss_nrm

        
    def compute_lmk2d_loss(self, data, lmk3d):

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            lmk2d_data = data["lmk2d_rgb"].clone()
            lmk2d = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_rgb[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1])
        elif self.cfg.MODE == "depth":
            lmk2d_data = data["lmk2d_depth"].clone()
            lmk2d = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_depth[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1])
        lmk2d_data[:, 0], lmk2d_data[:, 1] = normalize_image_points(lmk2d_data[:, 0], lmk2d_data[:, 1], self.cfg.IMG_RES)
        lmk2d[:, 0], lmk2d[:, 1] = normalize_image_points(lmk2d[:, 0], lmk2d[:, 1], self.cfg.IMG_RES)

        # # only consider palm and fingertips
        # id_lmk_fingertips = [
        #     0,                    # root
        #     13, 1, 4, 10, 7,      # mcp: thumb, index, middle, ring, pinky
        #     20, 16, 17, 19, 18,   # fingertips [thumb to pinky]
        # ]
        # lmk2d_data = lmk2d_data[id_lmk_fingertips]
        # lmk2d = lmk2d[id_lmk_fingertips]

        diff = lmk2d_data - lmk2d

        # compute general landmark term
        lmk2d_loss = torch.norm(diff, dim=1, p=1)

        return lmk2d_loss.mean()

    def render_sil(self, vert, mvp):
        v_clip = utils.cam_to_clip_space(vert, mvp)
        # v_clip = ru.xfm_points(vert[None, :, :], mvp[None, :, :])[0]   # (n, 4) xfm_points requires batch input
        rast, _ = dr.rasterize(self.glctx, v_clip[None, :, :], self.mano.faces.int(), self.cfg.IMG_RES)
        v_color_white = torch.ones([v_clip.shape[0], 1], device=self.cfg.DEVICE)
        sil, _ = dr.interpolate(v_color_white[None, :, :], rast, self.mano.faces.int())   # (1, H, W, 1)
        sil = dr.antialias(sil, rast, v_clip, self.mano.faces.int())    # (1, H, W, 1)  # Note: this is necessary for gradients wrt silhouette
        sil = sil[0, :, :, 0]

        return sil

    def compute_sil_loss(self, data, vert):
        sil = self.render_sil(vert, self.list_mvp_rgb[data["id_frame"]])
        data_sil = data["rgba"][:, :, 3]

        DEBUG = False
        if DEBUG:
            fig = go.Figure(go.Heatmap(z=sil.detach().cpu().numpy()*255))
            fig.show()
        
            fig = go.Figure(go.Heatmap(z=data_sil.detach().cpu().numpy()*255))
            fig.show()
            exit()
        
        # sil_loss = (sil - data_sil).abs().mean()

        # Ref: https://github.com/SeanChenxy/HandAvatar/blob/d7da7dfc5c2d341fce50624413b65fa6177d62d8/handavatar/core/train/trainers/handavatar/trainer.py#L174
        sil_loss = (1 - (sil*data_sil).sum() / (sil+data_sil - sil*data_sil).abs().sum())
        # this works well

        # Ref: https://github.com/SeanChenxy/HandMesh/blob/dbdca0d4fba75d7b010762edd4499b8102e06071/cmr/models/cmr_sg.py#L241
        # sil_loss = F.binary_cross_entropy(sil, data_sil, reduction='mean')
        # this diverges sometimes

        return sil_loss
    
    def compute_temp_loss(self, id_frame):
        if id_frame == 0:
            return torch.zeros(()), torch.zeros(()), torch.zeros(())
        
        hand_pose_temp_loss = (self.list_hand_pose[id_frame] - self.list_hand_pose[id_frame-1]).abs().mean()
        global_rot_temp_loss = (self.list_global_rot[id_frame] - self.list_global_rot[id_frame-1]).abs().mean()
        global_transl_temp_loss = (self.list_global_transl[id_frame] - self.list_global_transl[id_frame-1]).abs().mean()
        
        return hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss

    def compute_beta_reg_loss(self):
        reg_beta = self.beta ** 2
        return reg_beta.sum()
    
    def compute_laplacian_loss(self, vert):
        return regularizer.laplace_regularizer_const(vert, self.mano.faces)


    def compute_loss(self, data, vert, lmk3d):
        loss_dict = {}
        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            loss_dict["chamf_pos"], loss_dict["chamf_nrm"] = self.compute_chamfer_loss(data, vert)
            loss_dict["chamf_pos"] = loss_dict["chamf_pos"] * self.cfg.INIT.W_CHAMFER_POS
            # loss_dict["chamf_nrm"] = loss_dict["chamf_nrm"] * self.cfg.INIT.W_CHAMFER_NRM # TODO: normal gives nan
            loss_dict["chamf_nrm"] = torch.tensor(0, dtype=torch.float32, device=self.cfg.DEVICE)
            
        # lmk3d_loss = self.compute_lmk3d_loss(data, lmk3d)
        loss_dict["lmk2d"] = self.compute_lmk2d_loss(data, lmk3d) * self.cfg.INIT.W_LMK2D
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            loss_dict["sil"] = self.compute_sil_loss(data, vert) * self.cfg.INIT.W_SIL
        loss_dict["beta_reg"] = self.compute_beta_reg_loss() * self.cfg.INIT.W_BETA_REG
        loss_dict["laplace"] = self.compute_laplacian_loss(vert) * self.cfg.INIT.W_LAPLACE_REG
        
        loss_dict["hand_pose_temp"], loss_dict["global_rot_temp"], loss_dict["global_transl_temp"] = self.compute_temp_loss(data["id_frame"])
        loss_dict["hand_pose_temp"] = loss_dict["hand_pose_temp"] * self.cfg.INIT.W_TEMP_HAND_POSE
        loss_dict["global_rot_temp"] = loss_dict["global_rot_temp"] * self.cfg.INIT.W_TEMP_GLOBAL_ROT
        loss_dict["global_transl_temp"] = loss_dict["global_transl_temp"] * self.cfg.INIT.W_TEMP_GLOBAL_TRANSL

        loss_dict["total"] = utils.sum_dict(loss_dict)

        # + lmk3d_loss * self.cfg.INIT.W_LMK3D \
        # + loss_chamf_nrm * self.cfg.INIT.W_CHAMFER_NRM \
        # total_loss = loss_chamf_pos * self.cfg.INIT.W_CHAMFER_POS \
        #             + lmk2d_loss * self.cfg.INIT.W_LMK2D \
        #             + sil_loss * self.cfg.INIT.W_SIL \
        #             + beta_reg_loss * self.cfg.INIT.W_BETA_REG \
        #             + laplace_loss * self.cfg.INIT.W_LAPLACE_REG \
        #             + hand_pose_temp_loss * self.cfg.INIT.W_TEMP_HAND_POSE \
        #             + global_rot_temp_loss * self.cfg.INIT.W_TEMP_GLOBAL_ROT \
        #             + global_transl_temp_loss * self.cfg.INIT.W_TEMP_GLOBAL_TRANSL \

        # return total_loss, loss_chamf_pos, loss_chamf_nrm, lmk3d_loss, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss
        # return total_loss, loss_chamf_pos, loss_chamf_nrm, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss

        # total_loss = torch.tensor([0], dtype=torch.float32, device=self.cfg.DEVICE)
                
        return loss_dict
        
            
    def optimize(self, log_dir):
        utils.create_dir(log_dir, True)
        logger.info(f"Initialization log will be saved at {log_dir}")

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            log_mesh_on_rgb_dir = f"{log_dir}/mesh_on_rgb"; utils.create_dir(log_mesh_on_rgb_dir, True)
        log_lmk2d_data_vs_mano_dir = f"{log_dir}/lmk2d_data_vs_mano"; utils.create_dir(log_lmk2d_data_vs_mano_dir, True)
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            log_sil_data_vs_mano_dir = f"{log_dir}/sil_data_vs_mano"; utils.create_dir(log_sil_data_vs_mano_dir, True)
        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            log_pc_vs_mesh_dir = f"{log_dir}/pc_vs_mesh"; utils.create_dir(log_pc_vs_mesh_dir, True)
            log_depth_diff_dir = f"{log_dir}/depth_vs_mesh"; utils.create_dir(log_depth_diff_dir, True)
        path_log_loss_file = f"{log_dir}/loss.txt"

        self.DEBUG_CHAMF_LOSS = False
        if self.DEBUG_CHAMF_LOSS:
            self.log_chamf_dir = f"{log_dir}/chamf"; utils.create_dir(self.log_chamf_dir, True)

        self.setup_optimizer()

        _W_LMK3D = self.cfg.INIT.W_LMK3D
        _W_LMK2D = self.cfg.INIT.W_LMK2D

        with open(path_log_loss_file, "w") as log_loss_file:
            for data in self.dataset:
                id_frame = data["id_frame"]
                if self.DEBUG_CHAMF_LOSS:
                    self.id_frame = id_frame

                DEBUG_FRAME = False
                if DEBUG_FRAME:
                    self.cfg.INIT.W_TEMP = 0
                    if id_frame != 0:
                        continue
                    if self.cfg.MODE == "rgb":
                        self.initialize_global_pose_with_rgb(data)
                    else:
                        self.initialize_global_pose_with_depth(data)
                    # self.cfg.INIT.OPTIMIZE_SHAPE = False
                else:
                    if id_frame == 0:
                        if self.cfg.MODE == "rgb":
                            self.initialize_global_pose_with_rgb(data)
                        else:
                            self.initialize_global_pose_with_depth(data)
                    else:
                        self.initialize_from_previous_frame(id_frame)
                        self.cfg.INIT.OPTIMIZE_SHAPE = False
                        # with torch.no_grad():
                        #     lmk2d_data_proj = utils.clip_to_img(utils.cam_to_clip_space(data["lmk3d_cam"], self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1])
                        #     error = reproj_error(lmk2d_data_proj.cpu().numpy(), data["lmk2d"].cpu().numpy())
                        #     if error > self.cfg.INIT.REPROJ_ERROR_THRESH:
                        #         # cannot rely on landmark, so use previous frame's pose
                        #         logger.warning(f"At frame {id_frame}, data reproj error ({error:.3f}) exceeded! Initializing from previous frame.")
                        #         self.initialize_from_previous_frame(id_frame)
                        #         self.cfg.INIT.W_LMK3D = _W_LMK3D/10
                        #         self.cfg.INIT.W_LMK2D = _W_LMK2D/10
                        #     else:
                        #         self.initialize_global_pose(data)
                        #         self.cfg.INIT.W_LMK3D = _W_LMK3D
                        #         self.cfg.INIT.W_LMK2D = _W_LMK2D
                    

                        # # initialize current frame's pose from previous frame
                        # # if loss is high, reinitialize global pose from palm points
                        # self.initialize_hand_pose_from_previous_frame(id_frame)
                        # self.initialize_global_pose_from_previous_frame(id_frame)
                        # with torch.no_grad():
                        #     vert, lmk3d = self.forward_mano(data)
                        #     lmk3d_loss = self.compute_lmk3d_loss(data, lmk3d)
                        #     # err_lmk3d = self.compute_lmk3d_error(data, lmk3d)
                        #     if lmk3d_loss.item() > 0.025:
                        #         logger.warning(f"Re-initializing global pose from palm points for frame {id_frame} (previous frame's global pose leads to lmk3d_loss={lmk3d_loss:>.5f})")
                        #         # self.initialize_global_pose(data)
                        #         num_steps = self.cfg.INIT.STEPS_WITH_GLOBAL_POSE
                        #     else:
                        #         num_steps = self.cfg.INIT.STEPS_WITH_PREV_POSE


                if DEBUG_FRAME:
                    with torch.no_grad():
                        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                            # plot mesh
                            vert, lmk3d = self.forward_mano(data)
                            rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_rgb[data["id_frame"]]))
                            composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                            cv.imwrite(f"{log_mesh_on_rgb_dir}/init.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                        # # plot 3D landmark
                        # lmk3d_mano = lmk3d.cpu().numpy()
                        # lmk3d_data = data["lmk3d_cam"].cpu().numpy()
                        # scat_mano = go.Scatter3d(x=lmk3d_mano[:, 0], y=lmk3d_mano[:, 1], z=lmk3d_mano[:, 2], mode="markers", name="lmk3d_mano", customdata=np.arange(len(lmk3d_mano)), hovertemplate="%{customdata}<extra></extra>")
                        # scat_data = go.Scatter3d(x=lmk3d_data[:, 0], y=lmk3d_data[:, 1], z=lmk3d_data[:, 2], mode="markers", name="lmk3d_data", customdata=np.arange(len(lmk3d_data)), hovertemplate="%{customdata}<extra></extra>")
                        # fig = go.Figure([scat_data, scat_mano])
                        # fig.update_layout(scene=dict(
                        #     # aspectmode="cube",
                        #     aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
                        #     # xaxis=dict(range=[-0.1, 0.1]),
                        #     # yaxis=dict(range=[-0.1, 0.1]),
                        #     # zaxis=dict(range=[-0.1, 0.1]),
                        # ))
                        # fig.write_html(f"{log_lmk3d_data_vs_mano_dir}/init.html")

                        # plot 2D landmark
                        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                            lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_rgb[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                            lmk2d_data = data["lmk2d_rgb"].cpu().numpy()
                            rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                            rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))    
                            cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/init.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
                        elif self.cfg.MODE == "depth":
                            lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_depth[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()  
                            lmk2d_data = data["lmk2d_depth"].cpu().numpy()
                            depth_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(data["depth"].cpu().numpy(), self.cfg.KINECT.DEPTH_NEAR, self.cfg.KINECT.DEPTH_FAR)
                            depth_lmk2d_plot = utils.draw_pts_on_img(depth_cm[:, :, :3], lmk2d_mano, radius=5, color=(0, 0, 255))
                            depth_lmk2d_plot = utils.draw_pts_on_img(depth_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0)) 
                            cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/init.png", cv.cvtColor(depth_lmk2d_plot, cv.COLOR_RGB2BGR))

                        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                            # plot sil
                            sil_mano = self.render_sil(vert, self.list_mvp_rgb[data["id_frame"]]).cpu().numpy()
                            sil_data = data["rgba"][:, :, 3].cpu().numpy()
                            composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                            composite_sil[:, :, 1] = sil_data*255
                            composite_sil[:, :, 2] = sil_mano*255
                            cv.imwrite(f"{log_sil_data_vs_mano_dir}/init.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

                        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
                            # plot mesh + depth pc
                            K_depth = data["K_depth"]
                            fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                            mask_depth_data = data["depth"] > 0
                            xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth) / 1000
                            pc_pos = xyz_data.cpu().numpy()
                            scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                            vert_np = vert.detach().cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
                            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
                            # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                            fig = go.Figure([scat_pc_pos, plotlymesh])
                            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                            fig.write_html(f"{log_pc_vs_mesh_dir}/init.html")

                            # plot depth_data - depth_ren
                            depth_ren, mask_ren = self.render_depth(vert, self.list_mvp_depth[data["id_frame"]])    # (H, W)
                            depth_ren = depth_ren.cpu().numpy() * 1000    # m to mm
                            depth_data = data["depth"].cpu().numpy()    # in mm
                            depth_diff_cm_with_alpha = utils.depth_diff_and_cm_with_alpha(depth_data, depth_ren, self.cfg.PLOT.DEPTH_DIFF_MAX_THRESH)
                            cv.imwrite(f"{log_depth_diff_dir}/init.png", cv.cvtColor(depth_diff_cm_with_alpha, cv.COLOR_RGBA2BGRA))
                
                # multiple iterations per frame
                for id_step in range(self.cfg.INIT.N_STEPS):
                    if DEBUG_FRAME:
                        self.id_frame = id_step

                    # ==============================================================================================
                    #  Forward pass
                    # ==============================================================================================
                    vert, lmk3d = self.forward_mano(data)
                    # total_loss, loss_chamf_pos, loss_chamf_nrm, lmk3d_loss, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss = self.compute_loss(data, vert, lmk3d)
                    # total_loss, loss_chamf_pos, loss_chamf_nrm, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss = self.compute_loss(data, vert, lmk3d)
                    loss_dict = self.compute_loss(data, vert, lmk3d)

                    # ==============================================================================================
                    #  Backpropagate
                    # ==============================================================================================
                    if self.cfg.OPT.OPTIMIZE_SHAPE:
                        self.optimizer_shape.zero_grad()
                    if self.cfg.OPT.OPTIMIZE_POSE:
                        self.optimizer_pose.zero_grad()

                    loss_dict["total"].backward()

                    if self.cfg.INIT.OPTIMIZE_SHAPE:
                        self.optimizer_shape.step()
                        self.scheduler_shape.step()
                    if self.cfg.INIT.OPTIMIZE_POSE:
                        self.optimizer_pose.step()
                        # self.scheduler_pose.step()
                    
                    # ==============================================================================================
                    #  Log output
                    # ==============================================================================================
                    # err_lmk3d = self.compute_lmk3d_error(data, lmk3d)
                    # log_loss_str = f"[Frame:{id_frame:>3d}/{len(self.dataset)-1:>3d}] [Step:{id_step:>3d}/{self.cfg.INIT.N_STEPS-1:>3d}]  total_loss:{total_loss.item():>7f} | loss_chamf_pos: {loss_chamf_pos.item():>7f} | loss_chamf_nrm: {loss_chamf_nrm.item():>7f} | lmk3d_loss:{lmk3d_loss.item():>7f} | lmk2d_loss:{lmk2d_loss.item():>7f} | sil_loss:{sil_loss.item():>7f} | beta_reg_loss:{beta_reg_loss.item():>7f} | laplace_loss:{laplace_loss.item():>7f} | hand_pose_temp_loss: {hand_pose_temp_loss.item():>7f} | global_rot_temp_loss: {global_rot_temp_loss.item():>7f} | global_transl_temp_loss: {global_transl_temp_loss.item():>7f}"
                    # log_loss_str = f"[Frame:{id_frame:>3d}/{len(self.dataset)-1:>3d}] [Step:{id_step:>3d}/{self.cfg.INIT.N_STEPS-1:>3d}]  total_loss:{total_loss.item():>7f} | loss_chamf_pos: {loss_chamf_pos.item():>7f} | loss_chamf_nrm: {loss_chamf_nrm.item():>7f} | lmk2d_loss:{lmk2d_loss.item():>7f} | sil_loss:{sil_loss.item():>7f} | beta_reg_loss:{beta_reg_loss.item():>7f} | laplace_loss:{laplace_loss.item():>7f} | hand_pose_temp_loss: {hand_pose_temp_loss.item():>7f} | global_rot_temp_loss: {global_rot_temp_loss.item():>7f} | global_transl_temp_loss: {global_transl_temp_loss.item():>7f}"
                    log_loss_str = (
                        f"[Frame:{id_frame:>3d}/{len(self.dataset)-1:>3d}] [Step:{id_step:>3d}/{self.cfg.INIT.N_STEPS-1:>3d}]"
                        f" total:{loss_dict['total'].item():>7f}"
                    )
                    if "chamf_pos" in loss_dict:
                        log_loss_str += (
                            f" | chamf_pos: {loss_dict['chamf_pos'].item():>7f}"
                            f" | chamf_nrm: {loss_dict['chamf_nrm'].item():>7f}"
                        )
                    log_loss_str += (
                        f" | lmk2d: {loss_dict['lmk2d'].item():>7f}"
                    )
                    if "sil" in loss_dict:
                        log_loss_str += (
                            f" | sil: {loss_dict['sil'].item():>7f}"
                        )
                    log_loss_str += (
                        f" | beta_reg: {loss_dict['beta_reg'].item():>7f}"
                        f" | laplace: {loss_dict['laplace'].item():>7f}"
                        f" | hand_pose_temp: {loss_dict['hand_pose_temp'].item():>7f}"
                        f" | global_rot_temp: {loss_dict['global_rot_temp'].item():>7f}"
                        f" | global_transl_temp: {loss_dict['global_transl_temp'].item():>7f}"
                    )
                    logger.info(log_loss_str)
                    log_loss_file.write(f"{log_loss_str}\n")

                    if DEBUG_FRAME:
                        with torch.no_grad():
                            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                                # plot mesh
                                vert, lmk3d = self.forward_mano(data)
                                rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_rgb[data["id_frame"]]))
                                composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                                cv.imwrite(f"{log_mesh_on_rgb_dir}/{id_step:05d}.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                            # plot 2D landmark
                            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                                lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_rgb[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                                lmk2d_data = data["lmk2d_rgb"].cpu().numpy()
                                rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                                rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))    
                                cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_step:05d}.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
                            elif self.cfg.MODE == "depth":
                                lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_depth[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()  
                                lmk2d_data = data["lmk2d_depth"].cpu().numpy()
                                depth_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(data["depth"].cpu().numpy(), self.cfg.KINECT.DEPTH_NEAR, self.cfg.KINECT.DEPTH_FAR)
                                depth_lmk2d_plot = utils.draw_pts_on_img(depth_cm[:, :, :3], lmk2d_mano, radius=5, color=(0, 0, 255))
                                depth_lmk2d_plot = utils.draw_pts_on_img(depth_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0)) 
                                cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_step:05d}.png", cv.cvtColor(depth_lmk2d_plot, cv.COLOR_RGB2BGR))

                            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                                # plot sil
                                sil_mano = self.render_sil(vert, self.list_mvp_rgb[data["id_frame"]]).cpu().numpy()
                                sil_data = data["rgba"][:, :, 3].cpu().numpy()
                                composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                                composite_sil[:, :, 1] = sil_data*255
                                composite_sil[:, :, 2] = sil_mano*255
                                cv.imwrite(f"{log_sil_data_vs_mano_dir}/{id_step:05d}.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

                            if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
                                # plot mesh + depth
                                K_depth = data["K_depth"]
                                fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                                mask_depth_data = data["depth"] > 0
                                xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth) / 1000
                                pc_pos = xyz_data.cpu().numpy()
                                scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                                vert_np = vert.detach().cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
                                plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
                                # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                                fig = go.Figure([scat_pc_pos, plotlymesh])
                                fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                                fig.update_layout(scene_camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=-2.0)))
                                fig.write_html(f"{log_pc_vs_mesh_dir}/{id_step:05d}.html")

                                # plot depth_data - depth_ren
                                depth_ren, mask_ren = self.render_depth(vert, self.list_mvp_depth[data["id_frame"]])    # (H, W)
                                depth_ren = depth_ren.cpu().numpy() * 1000    # m to mm
                                depth_data = data["depth"].cpu().numpy()    # in mm
                                depth_diff_cm_with_alpha = utils.depth_diff_and_cm_with_alpha(depth_data, depth_ren, self.cfg.PLOT.DEPTH_DIFF_MAX_THRESH)
                                cv.imwrite(f"{log_depth_diff_dir}/{id_step:05d}.png", cv.cvtColor(depth_diff_cm_with_alpha, cv.COLOR_RGBA2BGRA))


                if DEBUG_FRAME:
                    break

                with torch.no_grad():
                    if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                        # plot mesh
                        vert, lmk3d = self.forward_mano(data)
                        rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp_rgb[data["id_frame"]]))
                        composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                        cv.imwrite(f"{log_mesh_on_rgb_dir}/{id_frame:05d}.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                    # plot 2D landmark
                    if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                        lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_rgb[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                        lmk2d_data = data["lmk2d_rgb"].cpu().numpy()
                        rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                        rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))    
                        cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_frame:05d}.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
                    elif self.cfg.MODE == "depth":
                        lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp_depth[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()  
                        lmk2d_data = data["lmk2d_depth"].cpu().numpy()
                        depth_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(data["depth"].cpu().numpy(), self.cfg.KINECT.DEPTH_NEAR, self.cfg.KINECT.DEPTH_FAR)
                        depth_lmk2d_plot = utils.draw_pts_on_img(depth_cm[:, :, :3], lmk2d_mano, radius=5, color=(0, 0, 255))
                        depth_lmk2d_plot = utils.draw_pts_on_img(depth_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0)) 
                        cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_frame:05d}.png", cv.cvtColor(depth_lmk2d_plot, cv.COLOR_RGB2BGR))

                    if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                        # plot sil
                        sil_mano = self.render_sil(vert, self.list_mvp_rgb[data["id_frame"]]).cpu().numpy()
                        sil_data = data["rgba"][:, :, 3].cpu().numpy()
                        composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                        composite_sil[:, :, 1] = sil_data*255
                        composite_sil[:, :, 2] = sil_mano*255
                        cv.imwrite(f"{log_sil_data_vs_mano_dir}/{id_frame:05d}.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

                    if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
                        # plot mesh + depth
                        K_depth = data["K_depth"]
                        fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                        mask_depth_data = data["depth"] > 0
                        xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth) / 1000
                        pc_pos = xyz_data.cpu().numpy()
                        scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                        vert_np = vert.detach().cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
                        plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
                        # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                        fig = go.Figure([scat_pc_pos, plotlymesh])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.update_layout(scene_camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=-2.0)))
                        fig.write_html(f"{log_pc_vs_mesh_dir}/{id_frame:05d}.html")

                        # plot depth_data - depth_ren
                        depth_ren, mask_ren = self.render_depth(vert, self.list_mvp_depth[data["id_frame"]])    # (H, W)
                        depth_ren = depth_ren.cpu().numpy() * 1000    # m to mm
                        depth_data = data["depth"].cpu().numpy()    # in mm
                        depth_diff_cm_with_alpha = utils.depth_diff_and_cm_with_alpha(depth_data, depth_ren, self.cfg.PLOT.DEPTH_DIFF_MAX_THRESH)
                        cv.imwrite(f"{log_depth_diff_dir}/{id_frame:05d}.png", cv.cvtColor(depth_diff_cm_with_alpha, cv.COLOR_RGBA2BGRA))

                    
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            height_rgba_seg, width_rgba_seg = self.dataset[0]["rgba"].cpu().numpy().shape[:2]
            subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_rgba_seg}", f"{height_rgba_seg}", f"{log_mesh_on_rgb_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{log_lmk2d_data_vs_mano_dir}"])
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{log_sil_data_vs_mano_dir}"])
        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            height_depth_seg, width_depth_seg = self.dataset[0]["depth"].cpu().numpy().shape[:2]
            subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_depth_seg}", f"{height_depth_seg}", f"{log_depth_diff_dir}"])

        if DEBUG_FRAME:
            exit()
        
    @torch.no_grad()
    def save_results(self, out_dir):
        utils.create_dir(out_dir, True)
        logger.info(f"Initialization results will be saved at {out_dir}")

        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            out_mv_rgb_dir = f"{out_dir}/mv_rgb"; utils.create_dir(out_mv_rgb_dir, True)
            out_mvp_rgb_dir = f"{out_dir}/mvp_rgb"; utils.create_dir(out_mvp_rgb_dir, True)
            out_campos_rgb_dir = f"{out_dir}/campos_rgb"; utils.create_dir(out_campos_rgb_dir, True)
            out_rgba_seg_dir = f"{out_dir}/rgba_seg"; utils.create_dir(out_rgba_seg_dir, True)
        
        if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
            out_mv_depth_dir = f"{out_dir}/mv_depth"; utils.create_dir(out_mv_depth_dir, True)
            out_mvp_depth_dir = f"{out_dir}/mvp_depth"; utils.create_dir(out_mvp_depth_dir, True)
            out_campos_depth_dir = f"{out_dir}/campos_depth"; utils.create_dir(out_campos_depth_dir, True)

        out_beta_dir = f"{out_dir}/beta"; utils.create_dir(out_beta_dir, True)
        out_offset_dir = f"{out_dir}/offset"; utils.create_dir(out_offset_dir, True)
        out_global_rot_dir = f"{out_dir}/global_rot"; utils.create_dir(out_global_rot_dir, True)
        out_global_transl_dir = f"{out_dir}/global_transl"; utils.create_dir(out_global_transl_dir, True)
        out_hand_pose_dir = f"{out_dir}/hand_pose"; utils.create_dir(out_hand_pose_dir, True)
        out_vert_dir = f"{out_dir}/vert"; utils.create_dir(out_vert_dir, True)
        
        np.save(f"{out_beta_dir}/beta.npy", self.beta.cpu().numpy())
        np.save(f"{out_offset_dir}/offset.npy", self.offset.cpu().numpy())
        
        for id_frame, data in enumerate(tqdm(self.dataset, desc="Writing initialization")):
            np.save(f"{out_global_rot_dir}/{id_frame:05d}.npy", self.list_global_rot[id_frame].cpu().numpy())
            np.save(f"{out_global_transl_dir}/{id_frame:05d}.npy", self.list_global_transl[id_frame].cpu().numpy())
            np.save(f"{out_hand_pose_dir}/{id_frame:05d}.npy", self.list_hand_pose[id_frame].cpu().numpy())
            
            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                np.save(f"{out_mv_rgb_dir}/{id_frame:05d}.npy", self.list_mv_rgb[id_frame].cpu().numpy())
                np.save(f"{out_mvp_rgb_dir}/{id_frame:05d}.npy", self.list_mvp_rgb[id_frame].cpu().numpy())
                np.save(f"{out_campos_rgb_dir}/{id_frame:05d}.npy", self.list_campos_rgb[id_frame].cpu().numpy())
            
            if self.cfg.MODE == "rgbd" or self.cfg.MODE == "depth":
                np.save(f"{out_mv_depth_dir}/{id_frame:05d}.npy", self.list_mv_depth[id_frame].cpu().numpy())
                np.save(f"{out_mvp_depth_dir}/{id_frame:05d}.npy", self.list_mvp_depth[id_frame].cpu().numpy())
                np.save(f"{out_campos_depth_dir}/{id_frame:05d}.npy", self.list_campos_depth[id_frame].cpu().numpy())
            
            vert, lmk3d = self.forward_mano(self.dataset[id_frame])
            np.save(f"{out_vert_dir}/{id_frame:05d}.npy", vert.cpu().numpy())

            # segment image using mesh
            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                if self.cfg.INIT.SEG_USING_MESH:
                    rgb = (data["rgba"].cpu().numpy()*255).astype(np.uint8)[:, :, :3]
                    img_height, img_width = rgb.shape[:2]
                    vert_img = utils.clip_to_img(utils.cam_to_clip_space(vert.cpu().numpy(), self.list_mvp_rgb[id_frame].cpu().numpy()), img_height, img_width)
                    mask = obtain_mask_from_verts_img(rgb, vert_img.astype(np.int32), self.mano.faces.cpu().numpy())
                    mask = mask[:, :, 0] == 255
                    rgb_seg = rgb*mask[:, :, None] + np.zeros_like(rgb)*(~mask[:, :, None])   # (H, W, 3)
                    rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)
                    cv.imwrite(f"{out_rgba_seg_dir}/{id_frame:05d}.png",  cv.cvtColor(rgba_seg, cv.COLOR_RGBA2BGRA))
                else:
                    cv.imwrite(f"{out_rgba_seg_dir}/{id_frame:05d}.png",  cv.cvtColor((data["rgba"].cpu().numpy()*255).astype(np.uint8), cv.COLOR_RGBA2BGRA))




