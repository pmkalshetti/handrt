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
import subprocess
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
from pathlib import Path
from pytorch3d.ops import sample_points_from_meshes, estimate_pointcloud_normals, sample_farthest_points, knn_gather, knn_points, corresponding_points_alignment

from .. import utils
from ..mano.mano import Mano
from ..render import optixutils as ou
from ..render import renderutils as ru
from ..render import bilateral_denoiser, light, texture, render, regularizer, util, mesh, material, obj
from ..metrics.ssim import calculate_ssim,  calculate_msssim
# from .mano_initializer import get_depth_to_color_stereo_extrinsic_transform_4x4

logger = utils.get_logger(__name__)

def create_trainable_mat(cfg):
    kd_min, kd_max = torch.tensor(cfg.MAT.KD_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.KD_MAX, dtype=torch.float32, device=cfg.DEVICE)
    # num_channels = 3
    # kd_init = torch.ones(size=cfg.MAT.TEXTURE_RES + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
    # kd_init = cv.imread(cfg.HTML_KD).astype(np.float32) / 255
    # mask_kd = cv.cvtColor(cv.imread(cfg.HTML_KD), cv.COLOR_BGR2GRAY) == 128
    # kd_init[mask_kd] = 1
    # kd_init = texture.srgb_to_rgb(texture.Texture2D(kd_init))
    
    kd_init = texture.srgb_to_rgb(texture.load_texture2D(cfg.MANO.HTML_KD, channels=3))
    kd_map_opt = texture.create_trainable(kd_init , cfg.MAT.TEXTURE_RES, True, [kd_min, kd_max])

    ks_min, ks_max = torch.tensor(cfg.MAT.KS_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.KS_MAX, dtype=torch.float32, device=cfg.DEVICE)
    ksR = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=0.0, high=0.01)
    ksG = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
    ksB = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

    ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), cfg.MAT.TEXTURE_RES, True, [ks_min, ks_max])
    
    nrm_min, nrm_max = torch.tensor(cfg.MAT.NRM_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.NRM_MAX, dtype=torch.float32, device=cfg.DEVICE)
    normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), cfg.MAT.TEXTURE_RES, True, [nrm_min, nrm_max])

    mat = {
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt,
        'bsdf'   : cfg.MAT.BSDF,
        'no_perturbed_nrm': cfg.MAT.NO_PERTURBED_NRM
    }

    return mat

@torch.no_grad()
def mix_background(data):
    # Mix background into a dataset image
    background = torch.zeros(data['rgba'].shape[0:2] + (3,), dtype=torch.float32, device=data['rgba'].device)
    data['background'] = background
    data['rgba'] = torch.cat((torch.lerp(background, data['rgba'][..., 0:3], data['rgba'][..., 3:4]), data['rgba'][..., 3:4]), dim=-1)

    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, preprocess_dir, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
            num_frames = len(list(sorted(Path(f"{preprocess_dir}/lmk_seg/rgba_seg").glob("*.png"))))
        elif cfg.MODE == "rgbd" or cfg.MODE == "depth":
            num_frames = len(list(sorted(Path(f"{preprocess_dir}/lmk_seg/rgba_reg_seg").glob("*.png"))))

        beta = np.load(f"{preprocess_dir}/initialization/out/beta/beta.npy")
        self.beta = torch.from_numpy(beta).float()

        offset = np.load(f"{preprocess_dir}/initialization/out/offset/offset.npy")
        self.offset = torch.from_numpy(offset).float()

        self.list_global_rot = []
        self.list_global_transl = []
        self.list_hand_pose = []
        self.list_vert = []

        self.list_id_frame = []
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            self.list_rgba = []
            self.list_mv_rgb = []
            self.list_mvp_rgb = []
            self.list_campos_rgb = []

        
        if self.cfg.MODE == "depth" or self.cfg.MODE == "rgbd":
            self.list_depth = []
            self.list_mv_depth = []
            self.list_mvp_depth = []
            self.list_campos_depth = []
            self.list_K_depth = []

        if "guesswho" in self.cfg.EXPT_NAME:
            self.T_depth_to_rgb = np.eye(4, dtype=np.float32)
        else:
            if self.cfg.MODE == "rgbd":
                if not self.cfg.USE_REG_RGB:
                    self.T_depth_to_rgb = utils.get_calibrated_kinectv2_T_ir_to_rgb()
                else:
                    self.T_depth_to_rgb = np.eye(4, dtype=np.float32)


        for id_frame in tqdm(range(num_frames), desc="Appending to list"):
            self.list_id_frame.append(id_frame)

            global_rot = np.load(f"{preprocess_dir}/initialization/out/global_rot/{id_frame:05d}.npy")   # (3,)
            self.list_global_rot.append(torch.from_numpy(global_rot).float())

            global_transl = np.load(f"{preprocess_dir}/initialization/out/global_transl/{id_frame:05d}.npy")   # (3,)
            self.list_global_transl.append(torch.from_numpy(global_transl).float())
            
            hand_pose = np.load(f"{preprocess_dir}/initialization/out/hand_pose/{id_frame:05d}.npy")   # (45,)
            self.list_hand_pose.append(torch.from_numpy(hand_pose).float())

            vert = np.load(f"{preprocess_dir}/initialization/out/vert/{id_frame:05d}.npy")   # (778, 3)
            self.list_vert.append(torch.from_numpy(vert).float())

            # rgba = cv.cvtColor(cv.imread(f"{preprocess_dir}/preprocess/rgba_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
            if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
                rgba = cv.cvtColor(cv.imread(f"{preprocess_dir}/initialization/out/rgba_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
                rgba = torch.from_numpy(rgba).float()/255
                rgba[..., 0:3] = util.srgb_to_rgb(rgba[..., 0:3])
                self.list_rgba.append(rgba)

                R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                # T_gl_to_rgb =  T_kin_to_gl @ self.T_depth_to_rgb    # kin (short for kinect) is same as rgb
                if self.cfg.MODE == "rgbd":
                    T_gl_to_rgb =  T_kin_to_gl @ self.T_depth_to_rgb    # kin (short for kinect) is same as rgb
                elif self.cfg.MODE == "rgb":
                    T_gl_to_rgb =  T_kin_to_gl
                mv_rgb = T_gl_to_rgb
                # mv_rgb = np.load(f"{preprocess_dir}/initialization/out/mv_rgb/{id_frame:05d}.npy")      # (4, 4)
                self.list_mv_rgb.append(torch.from_numpy(mv_rgb).float())

                mvp_rgb = np.load(f"{preprocess_dir}/initialization/out/mvp_rgb/{id_frame:05d}.npy")   # (4, 4)
                self.list_mvp_rgb.append(torch.from_numpy(mvp_rgb).float())

                campos_rgb = torch.linalg.inv(torch.from_numpy(mv_rgb).float())[:3, 3]    # (3,)
                # campos_rgb = np.load(f"{preprocess_dir}/initialization/out/campos_rgb/{id_frame:05d}.npy")      # (4, 4)
                self.list_campos_rgb.append(campos_rgb)

            if self.cfg.MODE == "depth" or self.cfg.MODE == "rgbd":
                depth = np.load(f"{preprocess_dir}/lmk_seg/depth_seg_npy/{id_frame:05d}.npy")   # (H, W) in mm
                self.list_depth.append(torch.from_numpy(depth).float())

                R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                # T_gl_to_depth = self.T_rgb_to_depth @ T_kin_to_gl    # kin (short for kinect) is same as rgb
                T_gl_to_depth = T_kin_to_gl    # kin (short for kinect) is same as rgb
                mv_depth = T_gl_to_depth
                # mv_depth = np.load(f"{preprocess_dir}/initialization/out/mv_depth/{id_frame:05d}.npy")      # (4, 4)
                self.list_mv_depth.append(torch.from_numpy(mv_depth).float())

                mvp_depth = np.load(f"{preprocess_dir}/initialization/out/mvp_depth/{id_frame:05d}.npy")   # (4, 4)
                self.list_mvp_depth.append(torch.from_numpy(mvp_depth).float())

                campos_depth = torch.linalg.inv(torch.from_numpy(mv_depth).float())[:3, 3]    # (3,)
                # campos_depth = np.load(f"{preprocess_dir}/initialization/out/campos_depth/{id_frame:05d}.npy")      # (4, 4)
                self.list_campos_depth.append(campos_depth)

                K_depth = np.load(f"{preprocess_dir}/lmk_seg/K_depth/{id_frame:05d}.npy")
                self.list_K_depth.append(torch.from_numpy(K_depth).float())

        self.num_data = len(self.list_id_frame)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, id_frame):
        data_dict = {}
        data_dict["id_frame"] = self.list_id_frame[id_frame]
        
        data_dict["beta"] = self.beta.to(self.cfg.DEVICE)
        data_dict["offset"] = self.offset.to(self.cfg.DEVICE)
        data_dict["global_rot"] = self.list_global_rot[id_frame].to(self.cfg.DEVICE)
        data_dict["global_transl"] = self.list_global_transl[id_frame].to(self.cfg.DEVICE)
        data_dict["hand_pose"] = self.list_hand_pose[id_frame].to(self.cfg.DEVICE)
        data_dict["vert"] = self.list_vert[id_frame].to(self.cfg.DEVICE)
        
        if self.cfg.MODE == "rgb" or self.cfg.MODE == "rgbd":
            data_dict["rgba"] = self.list_rgba[id_frame].to(self.cfg.DEVICE)
            data_dict["mv_rgb"] = self.list_mv_rgb[id_frame].to(self.cfg.DEVICE)
            data_dict["mvp_rgb"] = self.list_mvp_rgb[id_frame].to(self.cfg.DEVICE)
            data_dict["campos_rgb"] = self.list_campos_rgb[id_frame].to(self.cfg.DEVICE)
        
        if self.cfg.MODE == "depth" or self.cfg.MODE == "rgbd":
            data_dict["depth"] = self.list_depth[id_frame].to(self.cfg.DEVICE)
            data_dict["mv_depth"] = self.list_mv_depth[id_frame].to(self.cfg.DEVICE)
            data_dict["mvp_depth"] = self.list_mvp_depth[id_frame].to(self.cfg.DEVICE)
            data_dict["campos_depth"] = self.list_campos_depth[id_frame].to(self.cfg.DEVICE)
            data_dict["K_depth"] = self.list_K_depth[id_frame].to(self.cfg.DEVICE)

        return data_dict
        # return {
        #     "id_frame": self.list_id_frame[id_frame],
        #     "rgba": self.list_rgba[id_frame].to(self.device),
        #     "beta": self.beta.to(self.device),
        #     "offset": self.offset.to(self.device),
        #     "global_rot": self.list_global_rot[id_frame].to(self.device),
        #     "global_transl": self.list_global_transl[id_frame].to(self.device),
        #     "hand_pose": self.list_hand_pose[id_frame].to(self.device),
        #     "vert": self.list_vert[id_frame].to(self.device),
        #     "mv_rgb": self.list_mv_rgb[id_frame].to(self.device),
        #     "mvp_rgb": self.list_mvp_rgb[id_frame].to(self.device),
        #     "campos_rgb": self.list_campos_rgb[id_frame].to(self.device),

        #     "depth": self.list_depth[id_frame].to(self.device),
        #     "mv_depth": self.list_mv_depth[id_frame].to(self.device),
        #     "mvp_depth": self.list_mvp_depth[id_frame].to(self.device),
        #     "campos_depth": self.list_campos_depth[id_frame].to(self.device),
        # }

class Optimizer:
    def __init__(self, cfg, preprocess_dir):
        self.cfg = cfg
        self.dataset = Dataset(self.cfg, preprocess_dir, self.cfg.DEVICE)
        self.num_data = len(self.dataset)

        # ==============================================================================================
        #  Create trainable mesh (with fixed topology)
        # ==============================================================================================
        self.mano = Mano(self.cfg.MANO).to(self.cfg.DEVICE)
        self.initialize_mano_params()

        # ==============================================================================================
        #  Create trainable material
        # ==============================================================================================
        self.mat = create_trainable_mat(self.cfg)

        # ==============================================================================================
        #  Create trainable light
        # ==============================================================================================
        if self.cfg.OPT.OPTIMIZE_LIGHT:
            self.lgt = light.create_trainable_env_rnd(self.cfg.RENDER.PROBE_RES, scale=0.0, bias=0.5)
        else:
            self.lgt = light.create_trainable_env_rnd(self.cfg.RENDER.PROBE_RES, scale=0.0, bias=0.0)
        
        # ==============================================================================================
        #  Setup denoiser
        # ==============================================================================================
        self.denoiser = bilateral_denoiser.BilateralDenoiser().to(self.cfg.DEVICE)
        
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
        self.base_mesh = mesh.Mesh(
            v_pos=self.mano.v_template, t_pos_idx=self.mano.faces,
            v_tex=self.mano.verts_uvs, t_tex_idx=self.mano.faces_uvs,
            material=self.mat
        )
        self.base_mesh = mesh.auto_normals(self.base_mesh)

        self.lpips = LPIPS(net='vgg').to(self.cfg.DEVICE)

        self.setup_optimizer()

        if self.cfg.OPT.USE_CALIBRATED:
            self.lgt = light.load_env(f"{self.cfg.OPT.CALIB_OUT_DIR}/light/probe.hdr", scale=1, res=[self.cfg.RENDER.PROBE_RES, self.cfg.RENDER.PROBE_RES])
            self.base_mesh = mesh.load_mesh(f"{self.cfg.OPT.CALIB_OUT_DIR}/mesh/mesh.obj")
            self.base_mesh = mesh.auto_normals(self.base_mesh)
            self.mat = self.base_mesh.material.copy()
            self.beta = torch.from_numpy(np.load(f"{self.cfg.OPT.CALIB_OUT_DIR}/beta/beta.npy")).float().to(self.cfg.DEVICE)
            self.offset = torch.from_numpy(np.load(f"{self.cfg.OPT.CALIB_OUT_DIR}/offset/offset.npy")).float().to(self.cfg.DEVICE)

    @torch.no_grad()
    def initialize_mano_params(self):
        self.beta = torch.zeros(10, requires_grad=True, device=self.cfg.DEVICE)
        self.offset = torch.zeros(len(self.mano.v_template), 3, requires_grad=True, device=self.cfg.DEVICE)
        self.list_global_rot = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        self.list_global_transl = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        self.list_hand_pose = [torch.zeros(15*3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]

        if self.cfg.OPT.USE_INIT:
            for id_data, data in enumerate(self.dataset):
                if id_data == 0:
                    self.beta.data = data["beta"].clone()
                    self.offset.data = data["offset"].clone()
                self.list_global_rot[id_data] = data["global_rot"].clone()
                self.list_global_transl[id_data] = data["global_transl"].clone()
                self.list_hand_pose[id_data] = data["hand_pose"].clone()

            logger.debug(f"Mano params initialized")

    def setup_optimizer(self):
        def lr_schedule(iter, fraction):
            warmup_iter = 0
            if iter < warmup_iter:
                return iter / warmup_iter 
            return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs
        
        if self.cfg.OPT.OPTIMIZE_MATERIAL:
            self.optimizer_mat = torch.optim.Adam((material.get_parameters(self.mat)), lr=self.cfg.OPT.LEARNING_RATE_LGT)
            self.scheduler_mat = torch.optim.lr_scheduler.LambdaLR(self.optimizer_mat, lr_lambda=lambda x: lr_schedule(x, 0.9))

        if self.cfg.OPT.OPTIMIZE_LIGHT:
            self.optimizer_light = torch.optim.Adam((self.lgt.parameters()), lr=self.cfg.OPT.LEARNING_RATE_LGT)
            self.scheduler_light = torch.optim.lr_scheduler.LambdaLR(self.optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9))

        if self.cfg.OPT.OPTIMIZE_SHAPE:
            params_shape = [
                {"params": [self.beta], "lr": self.cfg.OPT.LR_BETA},
                {"params": [self.offset], "lr": self.cfg.OPT.LR_OFFSET},
            ]
            self.optimizer_shape = torch.optim.SGD(params_shape, lr = self.cfg.OPT.LR_GEOM)
            self.scheduler_shape = torch.optim.lr_scheduler.LambdaLR(self.optimizer_shape, lr_lambda=lambda x: lr_schedule(x, 0.9))
    
        if self.cfg.OPT.OPTIMIZE_POSE:
            params_pose = [
                {"params": self.list_hand_pose, "lr": self.cfg.OPT.LR_HAND_POSE},
                {"params": self.list_global_rot, "lr": self.cfg.OPT.LR_GLOBAL_ROT},
                {"params": self.list_global_transl, "lr": self.cfg.OPT.LR_GLOBAL_TRANSL}
            ]
            self.optimizer_pose = torch.optim.SGD(params_pose, lr = self.cfg.OPT.LR_GEOM)
            self.scheduler_pose = torch.optim.lr_scheduler.LambdaLR(self.optimizer_pose, lr_lambda=lambda x: lr_schedule(x, 0.9))

    def forward_mano(self, data):
        id_frame = data["id_frame"]

        mano_out = self.mano(
            betas=self.beta[None, :],
            offsets=self.offset[None, :, :],
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

    def get_mesh(self, vert):
        opt_mesh = mesh.Mesh(v_pos=vert, base=self.base_mesh)
        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, opt_mesh.v_pos.contiguous(), opt_mesh.t_pos_idx.int(), rebuild=1)
        opt_mesh = mesh.auto_normals(opt_mesh)
        opt_mesh = mesh.compute_tangents(opt_mesh)
        return opt_mesh
    
    def forward(self, data, train=False):
        if train:
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.lgt.update_pdf()
            vert, lmk3d = self.forward_mano(data)
            opt_mesh = self.get_mesh(vert)
            buffers = render.render_mesh(self.cfg, self.glctx, opt_mesh, data["mvp_rgb"][None, :, :], data["campos_rgb"][None, :], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=True, background=data['background'][None, :, :, :], optix_ctx=self.optix_ctx, denoiser=self.denoiser)
        else:
            with torch.no_grad(): 
                vert, lmk3d = self.forward_mano(data)
                opt_mesh = self.get_mesh(vert)
                buffers = render.render_mesh(self.cfg, self.glctx, opt_mesh, data["mvp_rgb"][None, :, :], data["campos_rgb"][None, :], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, background=data['background'][None, :, :, :], optix_ctx=self.optix_ctx, denoiser=self.denoiser)

        # Note: each buffer has a batch dimension
        
        return opt_mesh, buffers
    
    def render_depth(self, v_cam, mvp):
        v_clip = utils.cam_to_clip_space(v_cam, mvp)
        # nvdiffrast to rasterize
        rast, _ = dr.rasterize(self.glctx, v_clip[None, :, :], self.mano.faces.int(), self.cfg.IMG_RES)
        
        alpha = torch.clamp(rast[..., 3], 0, 1) # rast[:, :, 3] == 0  # Field triangle_id is the triangle index, offset by one. Pixels where no triangle was rasterized will receive a zero in all channels.
        alpha = alpha[0]    # (H, W)
        mask = alpha > 0.5

        depth, _ = dr.interpolate(v_cam[None, :, 2:3].contiguous(), rast, self.mano.faces.int()) # (1, H, W, 1)
        depth = dr.antialias(depth, rast, v_clip, self.mano.faces.int())    # (1, H, W, 1)  # Note: this is necessary for gradients wrt silhouette
        depth = depth[0, :, :, 0]   # (H, W)

        return depth, mask
    
    def compute_chamfer_loss(self, data, vert):
        # chamfer vs hausdorff vs earth mover: slide 34 onwards at https://3ddl.cs.princeton.edu/2016/slides/su.pdf
       
        ## Matching: Projection-based
        depth_ren, mask_ren = self.render_depth(vert, data["mvp_depth"])    # (H, W)

        # depth to pointcloud
        K_depth = data["K_depth"]
        fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
        mask_depth_data = data["depth"] > 0
        xyz_ren = utils.depth_to_xyz(depth_ren, mask_ren, fx_depth, fy_depth, cx_depth, cy_depth)
        
        xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx_depth, fy_depth, cx_depth, cy_depth)
        xyz_data = xyz_data / 1000

        # estimate normal
        nrm_ren = estimate_pointcloud_normals(xyz_ren[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_ren = utils.orient_normals_towards_camera(xyz_ren, nrm_ren, camera_pos=torch.zeros(3, device=xyz_ren.device))
        nrm_data = estimate_pointcloud_normals(xyz_data[None, :, :], neighborhood_size=5, use_symeig_workaround=False)[0]
        nrm_data = utils.orient_normals_towards_camera(xyz_data, nrm_data, camera_pos=torch.zeros(3, device=xyz_data.device))

        # reduce points for computational performance
        xyz_data_selected, arr_id_selected_pt = sample_farthest_points(xyz_data[None, :, :], K=self.cfg.OPT.N_SAMPLES_ON_PC)
        # xyz_data = xyz_data[0]; arr_id_selected_pt = arr_id_selected_pt[0]
        arr_id_selected_pt = arr_id_selected_pt[0]
        xyz_data = xyz_data[arr_id_selected_pt]
        nrm_data = nrm_data[arr_id_selected_pt]

        xyz_ren_selected, arr_id_selected_pt = sample_farthest_points(xyz_ren[None, :, :], K=self.cfg.OPT.N_SAMPLES_ON_PC)
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
        if self.cfg.OPT.CHAMF_FILTER_NRM_DIST:
            cham_nrm_threshold = 1 - torch.cos(torch.deg2rad(torch.tensor(self.cfg.OPT.CHAMF_NRM_THRESH, dtype=torch.float).to(self.cfg.DEVICE)))
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
        if self.cfg.OPT.CHAMF_FILTER_POS_DIST:
            pos_ren_mask = cham_pos_ren < self.cfg.OPT.CHAMF_POS_THRESH**2
            pos_data_mask = cham_pos_data < self.cfg.OPT.CHAMF_POS_THRESH**2

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
        # if DEBUG_CHAMF_LOSS:
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
        loss_pos = self.cfg.OPT.CHAMF_W_REN*loss_pos_ren + self.cfg.OPT.CHAMF_W_DATA*loss_pos_data
        loss_nrm = self.cfg.OPT.CHAMF_W_REN*loss_nrm_ren + self.cfg.OPT.CHAMF_W_DATA*loss_nrm_data

        return loss_pos, loss_nrm

    def compute_loss(self, buffers, data, id_step, num_steps, opt_mesh):
        t_iter = (id_step+1) / num_steps
        color_ref = data['rgba'][None, :, :, :]

        # Image-space loss, split into a coverage component and a color component
        img_loss  = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += self.image_loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device=self.cfg.DEVICE)

        if self.cfg.MAT.BSDF == "bsdf":
            # Monochrome shading regularizer
            reg_loss = reg_loss + regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, self.cfg.OPT.LAMBDA_DIFFUSE, self.cfg.OPT.LAMBDA_SPECULAR)

        # Material smoothness regularizer
        reg_loss = reg_loss + regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.cfg.OPT.LAMBDA_KD, lambda_ks=self.cfg.OPT.LAMBDA_KS, lambda_nrm=self.cfg.OPT.LAMBDA_NRM)

        # Chroma regularizer
        reg_loss = reg_loss + regularizer.chroma_loss(buffers['kd'], color_ref, self.cfg.OPT.LAMBDA_CHROMA)

        # Perturbed normal regularizer
        if 'perturbed_nrm_grad' in buffers:
            reg_loss = reg_loss + torch.mean(buffers['perturbed_nrm_grad']) * self.cfg.OPT.LAMBDA_NRM2

        # Laplacian regularizer. 
        if self.cfg.OPT.LAPLACE == "absolute":
            reg_loss = reg_loss + regularizer.laplace_regularizer_const(opt_mesh.v_pos, opt_mesh.t_pos_idx) * self.cfg.OPT.LAPLACE_SCALE * (1 - t_iter)
        elif self.cfg.OPT.LAPLACE == "relative":
            init_mesh = mesh.Mesh(v_pos=data["vert"][0], base=self.base_mesh)
            reg_loss = reg_loss + regularizer.laplace_regularizer_const(opt_mesh.v_pos - init_mesh.v_pos, init_mesh.t_pos_idx) * self.cfg.OPT.LAPLACE_SCALE * (1 - t_iter)                

        ref = torch.clamp(util.rgb_to_srgb(color_ref[...,0:3]), 0.0, 1.0)
        opt = torch.clamp(util.rgb_to_srgb(buffers['shaded'][...,0:3]), 0.0, 1.0)
        lpips_loss = self.lpips(opt.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2), normalize=True)
        reg_loss = reg_loss + self.cfg.OPT.LAMBDA_LPIPS * torch.mean(lpips_loss)

        if self.cfg.OPT.OPTIMIZE_LIGHT:
            # Light white balance regularizer
            reg_loss = reg_loss + self.lgt.regularizer() * self.cfg.OPT.W_LGT_REG

        # if self.cfg.MODE == "depth" or self.cfg.MODE == "rgbd":
        #     chamf_pos, chamf_nrm = self.compute_chamfer_loss(data, opt_mesh.v_pos)
        #     reg_loss = reg_loss + chamf_pos * self.cfg.OPT.W_CHAMFER_POS

        return img_loss, reg_loss
     
    @torch.no_grad()
    def render_buffers(self, data, opt_mesh, buffers):
        result_dict = {}
        result_dict['ref'] = util.rgb_to_srgb(data['rgba'][...,0:3])
        result_dict['ref'] = torch.cat([result_dict['ref'], data['rgba'][...,3:4]], dim=2)
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][0, ...,0:3])
        result_dict['opt'] = torch.cat([result_dict['opt'], buffers['shaded'][0, ...,3:4]], dim=2)
        
        result_dict['light_image'] = self.lgt.generate_image(self.cfg.IMG_RES)
        result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
        result_dict['light_image'] = torch.cat([result_dict['light_image'], torch.ones([self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 1], device=self.cfg.DEVICE)], dim=2)
        
        # white_bg = torch.ones_like(batch['background'])
        result_dict["kd"] = util.rgb_to_srgb(render.render_mesh(self.cfg, self.glctx, opt_mesh, data["mvp_rgb"][None, :, :], data["campos_rgb"][None, :], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="kd", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][..., 0:3])[0]
        result_dict['kd'] = torch.cat([result_dict['kd'], buffers['shaded'][0, ...,3:4]], dim=2)

        result_dict["ks"] = render.render_mesh(self.cfg, self.glctx, opt_mesh, data["mvp_rgb"][None, :, :], data["campos_rgb"][None, :], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="ks", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][0, ..., 0:3]
        result_dict['ks'] = torch.cat([result_dict['ks'], buffers['shaded'][0, ...,3:4]], dim=2)
        
        result_dict["normal"] = render.render_mesh(self.cfg, self.glctx, opt_mesh, data["mvp_rgb"][None, :, :], data["campos_rgb"][None, :], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="normal", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][0, ..., 0:3]
        result_dict['normal'] = torch.cat([result_dict['normal'], buffers['shaded'][0, ...,3:4]], dim=2)

        result_image = torch.cat([result_dict['ref'], result_dict['opt'], result_dict['light_image'], result_dict["kd"], result_dict["ks"], result_dict["normal"]], axis=1)
        if not self.cfg.MAT.NO_PERTURBED_NRM:
            result_dict["perturbed_nrm"] = (buffers['perturbed_nrm'][0, ...,0:3] + 1.0) * 0.5
            result_dict['perturbed_nrm'] = torch.cat([result_dict['perturbed_nrm'], buffers['shaded'][0, ...,3:4]], dim=2)
            result_image = torch.cat([result_image, result_dict["perturbed_nrm"]], axis=1)
        
        if self.cfg.MAT.BSDF == "pbr":
            result_dict["diffuse_light"] = util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]
            result_dict['diffuse_light'] = torch.cat([result_dict['diffuse_light'], buffers['shaded'][0, ...,3:4]], dim=2)
            result_dict["specular_light"] = util.rgb_to_srgb(buffers['specular_light'][..., 0:3])[0]
            result_dict['specular_light'] = torch.cat([result_dict['specular_light'], buffers['shaded'][0, ...,3:4]], dim=2)

            result_image = torch.cat([result_image, result_dict["diffuse_light"], result_dict["specular_light"]], axis=1)

        if self.cfg.MODE == "depth" or self.cfg.MODE == "rgbd":
            depth_data = data["depth"].cpu().numpy()    # in mm
            depth_data_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_data, self.cfg.PLOT.DEPTH_NEAR, self.cfg.PLOT.DEPTH_FAR)
            result_dict["depth_data"] = torch.from_numpy(depth_data_cm).float().to(self.cfg.DEVICE) / 255
            result_image = torch.cat([result_image, result_dict["depth_data"]], axis=1)
            
            depth_ren, mask_ren = self.render_depth(opt_mesh.v_pos, data["mvp_depth"])    # (H, W)
            depth_ren = depth_ren.cpu().numpy() * 1000    # m to mm
            depth_ren_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_ren, self.cfg.PLOT.DEPTH_NEAR, self.cfg.PLOT.DEPTH_FAR)
            result_dict["depth_ren"] = torch.from_numpy(depth_ren_cm).float().to(self.cfg.DEVICE) / 255
            result_image = torch.cat([result_image, result_dict["depth_ren"]], axis=1)

            depth_diff_cm_with_alpha = utils.depth_diff_and_cm_with_alpha(depth_data, depth_ren, self.cfg.PLOT.DEPTH_DIFF_MAX_THRESH)
            result_dict["depth_diff"] = torch.from_numpy(depth_diff_cm_with_alpha).float().to(self.cfg.DEVICE) / 255
            result_image = torch.cat([result_image, result_dict["depth_diff"]], axis=1)

        return result_image, result_dict
    
    def train_epoch(self, id_epoch, log_dir):
        num_steps = self.cfg.OPT.EPOCHS * len(self.dataset)
        running_img_loss = 0.0
        running_reg_loss = 0.0
        running_total_loss = 0.0
        for id_data, data in enumerate(self.dataset):
            id_step = id_epoch * len(self.dataset) + id_data

            # ==============================================================================================
            #  Initialize parameters
            # ==============================================================================================
            if not self.cfg.OPT.USE_INIT:
                if id_epoch == 0:
                    raise NotImplementedError
            
            # ==============================================================================================
            #  Forward pass
            # ==============================================================================================
            data = mix_background(data)
            opt_mesh, buffers = self.forward(data, train=True)
            img_loss, reg_loss = self.compute_loss(buffers, data, id_step, num_steps, opt_mesh)
            total_loss = img_loss + reg_loss
            
            # ==============================================================================================
            #  Backpropagate
            # ==============================================================================================
            if self.cfg.OPT.OPTIMIZE_MATERIAL:
                self.optimizer_mat.zero_grad()
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.optimizer_light.zero_grad()
            if self.cfg.OPT.OPTIMIZE_SHAPE:
                self.optimizer_shape.zero_grad()
            if self.cfg.OPT.OPTIMIZE_POSE:
                self.optimizer_pose.zero_grad()
            
            total_loss.backward()

            if self.cfg.OPT.OPTIMIZE_MATERIAL:
                self.optimizer_mat.step()
                self.scheduler_mat.step()
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.lgt.base.grad *= 64
                self.optimizer_light.step()
                self.scheduler_light.step()
            if self.cfg.OPT.OPTIMIZE_SHAPE:
                self.optimizer_shape.step()
                self.scheduler_shape.step()
            if self.cfg.OPT.OPTIMIZE_POSE:
                self.optimizer_pose.step()
                self.scheduler_pose.step()
            
            # ==============================================================================================
            #  Clamp trainables to reasonable range
            # ==============================================================================================
            with torch.no_grad():
                if 'kd' in self.mat:
                    self.mat['kd'].clamp_()
                if 'ks' in self.mat:
                    self.mat['ks'].clamp_()
                if 'normal' in self.mat:
                    self.mat['normal'].clamp_()
                    self.mat['normal'].normalize_()
                if self.lgt is not None:
                    self.lgt.clamp_(min=0.01) # For some reason gradient dissapears if light becomes 0


            # ==============================================================================================
            #  Log output
            # ==============================================================================================
            running_img_loss += img_loss.item()
            running_reg_loss += reg_loss.item()
            running_total_loss += total_loss.item()

            # log
            if id_step % self.cfg.OPT.LOG_INTERVAL == (self.cfg.OPT.LOG_INTERVAL-1):
                with torch.no_grad():
                    opt_mesh, buffers = self.forward(data, train=False)
                    result_image, result_dict = self.render_buffers(data, opt_mesh, buffers)
                util.save_image(f"{log_dir}/{id_step:05d}.png", result_image.detach().cpu().numpy())

                log_str = f"[Step: {id_step:>5d} / {num_steps:>5d}] | img_loss: {running_img_loss / self.cfg.OPT.LOG_INTERVAL:>7f} | reg_loss: {running_reg_loss / self.cfg.OPT.LOG_INTERVAL:>7f} | total_loss: {running_total_loss / self.cfg.OPT.LOG_INTERVAL:>7f}"
                logger.info(log_str)

                running_img_loss = 0.0
                running_reg_loss = 0.0
                running_total_loss = 0.0
        
    @torch.no_grad()
    def test_epoch(self, id_epoch, log_dir):
        mse_values = []
        psnr_values = []
        ssim_values = []
        msssim_values = []
        lpips_values = []

        # Hack validation to use high sample count and no denoiser
        _n_samples = self.cfg.RENDER.N_SAMPLES
        _denoiser = self.denoiser
        self.cfg.RENDER.N_SAMPLES = 32
        self.denoiser = None
        
        out_val_dir = f"{log_dir}/epoch_{id_epoch:02d}"; utils.create_dir(out_val_dir, True)
        logger.info("Running evaluation on test")
        with open(f"{out_val_dir}/metrics.txt", 'a') as fout:
            fout.write('ID, MSE, PSNR, SSIM, MSSIM, LPIPS\n')
            # fout.write(f"Epoch: {id_epoch}\n")
            for id_data, data in enumerate(tqdm(self.dataset)):
                # if self.cfg.GENERATE_FIG:
                #     if not (id_data in self.cfg.FIG_ID_DATA):
                #         continue
                # id_step = id_epoch * len(dataloader_test) + id_data + 1

                data = mix_background(data)
                opt_mesh, buffers = self.forward(data, train=False)
                result_image, result_dict = self.render_buffers(data, opt_mesh, buffers)

                # Compute metrics
                opt = torch.clamp(result_dict['opt'], 0.0, 1.0)[..., :3]
                ref = torch.clamp(result_dict['ref'], 0.0, 1.0)[..., :3]

                mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                mse_values.append(float(mse))
                psnr = util.mse_to_psnr(mse)
                psnr_values.append(float(psnr))
                ssim = calculate_ssim(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :])
                ssim_values.append(float(ssim))
                msssim = calculate_msssim(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :])
                msssim_values.append(float(msssim))
                lpips_value = self.lpips(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :], normalize=True).item()
                lpips_values.append(float(lpips_value))

                # line = "%d, %1.8f, %1.8f \n" % (id_batch, mse, psnr)
                # line = f"[{id_batch:>5d} / {len(dataloader_test)-1:>5d}]  mse: {mse:>7f}  psnr: {psnr:>7f}  ssim: {ssim:>7f}  msssim: {msssim:>7f}  lpips: {lpips_value:>7f}\n"
                line = f"{id_data:>5d} {mse:>7f} {psnr:>7f} {ssim:>7f} {msssim:>7f} {lpips_value:>7f}\n"
                fout.write(str(line))
                util.save_image(f"{out_val_dir}/{id_data:05d}.png", result_image.detach().cpu().numpy())

                # if id_batch > 30:
                # if self.cfg.GENERATE_FIG:
                #     break

            avg_mse = np.mean(np.array(mse_values))
            avg_psnr = np.mean(np.array(psnr_values))
            avg_ssim = np.mean(np.array(ssim_values))
            avg_msssim = np.mean(np.array(msssim_values))
            avg_lpips = np.mean(np.array(lpips_values))
            line = f"Average\n{avg_mse:04f}, {avg_psnr:04f}, {avg_ssim:04f}, {avg_msssim:04f}, {avg_lpips:04f}\n"
            fout.write(str(line))
            logger.info("MSE,      PSNR,       SSIM,      MSSIM,     LPIPS")
            logger.info(line[8:])
        
        # Restore sample count and denoiser
        self.cfg.RENDER.N_SAMPLES = _n_samples
        self.denoiser = _denoiser



    def optimize(self, optim_dir):
        # log_dir = f"{self.cfg.OPT.ROOT_DIR}/{self.cfg.EXPT_NAME}/log";  utils.create_dir(log_dir, True)
        # logger.info(f"Optimization log will be saved at {log_dir}")

        log_train_dir = f"{optim_dir}/train"; utils.create_dir(log_train_dir, True)
        log_test_dir = f"{optim_dir}/test"; utils.create_dir(log_test_dir, True)
        params_dir = f"{optim_dir}/params"; utils.create_dir(params_dir, True)
        
        for id_epoch in range(self.cfg.OPT.EPOCHS):
            logger.info(f"------------- Epoch {id_epoch} -------------")
            self.train_epoch(id_epoch, log_train_dir)
            self.test_epoch(id_epoch, log_test_dir)

            params_epoch_dir = f"{params_dir}/epoch_{id_epoch:02d}";  utils.create_dir(params_epoch_dir, True)
            logger.info(f"Optimized parameters will be saved at {params_epoch_dir}")
            self.save_optimized_parameters(params_epoch_dir)
        logger.info(f"Optimization done!")

    @torch.no_grad()
    def save_optimized_parameters(self, out_dir):
        # out_dir = f"{self.cfg.OPT.ROOT_DIR}/{self.cfg.EXPT_NAME}/out";  utils.create_dir(out_dir, True)
        # logger.info(f"Optimized parameters will be saved at {out_dir}")

        out_beta_dir = f"{out_dir}/beta"; utils.create_dir(out_beta_dir, True)
        out_offset_dir = f"{out_dir}/offset"; utils.create_dir(out_offset_dir, True)
        out_global_rot_dir = f"{out_dir}/global_rot"; utils.create_dir(out_global_rot_dir, True)
        out_global_transl_dir = f"{out_dir}/global_transl"; utils.create_dir(out_global_transl_dir, True)
        out_hand_pose_dir = f"{out_dir}/hand_pose"; utils.create_dir(out_hand_pose_dir, True)
        out_vert_dir = f"{out_dir}/vert"; utils.create_dir(out_vert_dir, True)
        out_mesh_dir = f"{out_dir}/mesh"; utils.create_dir(out_mesh_dir, True)
        out_light_dir = f"{out_dir}/light"; utils.create_dir(out_light_dir, True)

        np.save(f"{out_beta_dir}/beta.npy", self.beta.cpu().numpy())
        np.save(f"{out_offset_dir}/offset.npy", self.offset.cpu().numpy())
        
        for id_data, data in enumerate(tqdm(self.dataset, desc="Saving optimized parameters")):
            np.save(f"{out_global_rot_dir}/{id_data:05d}.npy", self.list_global_rot[id_data].cpu().numpy())
            np.save(f"{out_global_transl_dir}/{id_data:05d}.npy", self.list_global_transl[id_data].cpu().numpy())
            np.save(f"{out_hand_pose_dir}/{id_data:05d}.npy", self.list_hand_pose[id_data].cpu().numpy())
            
            vert, lmk3d = self.forward_mano(data)
            np.save(f"{out_vert_dir}/{id_data:05d}.npy", vert.cpu().numpy())


        mano_output = self.mano(self.beta[None, :], torch.zeros((1, 3), device=self.cfg.DEVICE), torch.zeros((1, 15*3), device=self.cfg.DEVICE), torch.zeros((1, 3), device=self.cfg.DEVICE), self.offset[None, :, :], flat_hand_mean=True)
        vert = mano_output.vertices[0]
        final_mesh = self.get_mesh(vert)
        obj.write_obj(out_mesh_dir, final_mesh)

        light.save_env_map(f"{out_light_dir}/probe.hdr", self.lgt)
