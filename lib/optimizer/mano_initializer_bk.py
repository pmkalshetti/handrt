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
from pytorch3d.ops import sample_points_from_meshes, estimate_pointcloud_normals
import subprocess
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
from pathlib import Path
import open3d as o3d

from .. import utils
from ..mano.mano import Mano
from ..render import optixutils as ou
from ..render import renderutils as ru
from ..render import bilateral_denoiser, light, texture, render, regularizer, util, mesh, material, obj

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

        list_path_to_rgba_seg = list(sorted(Path(f"{lmk_seg_dir}/rgba_seg").glob("*.png")))
        # list_id_data = range(0, len(list_path_to_rgba_seg))

        self.list_id_frame = []
        self.list_rgba = []
        self.list_depth = []
        self.list_pc_pos = []
        self.list_pc_nrm = []
        self.list_K = []        # intrinsic mat
        self.list_lmk2d = []    # 2D landmark in image space
        self.list_lmk3d_rel = []    # 3D landmark in root-relative camera space
        self.list_lmk3d_cam = []    # 3D landmark in camera space

        for id_frame in tqdm(range(len(list_path_to_rgba_seg)), desc="Appending to list"):
            self.list_id_frame.append(id_frame)
            
            rgba = cv.cvtColor(cv.imread(f"{lmk_seg_dir}/rgba_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
            rgba = torch.from_numpy(rgba).float()/255
            # rgba[..., 0:3] = util.srgb_to_rgb(rgba[..., 0:3])
            self.list_rgba.append(rgba)

            depth = np.load(f"{lmk_seg_dir}/depth_seg_npy/{id_frame:05d}.npy")   # (H, W) in mm
            self.list_depth.append(torch.from_numpy(depth).float())

            pc_pos = np.load(f"{lmk_seg_dir}/pc_pos/{id_frame:05d}.npy")   # (N_POINT_SAMPLES, 3) in m
            self.list_pc_pos.append(torch.from_numpy(pc_pos).float())

            pc_nrm = np.load(f"{lmk_seg_dir}/pc_nrm/{id_frame:05d}.npy")   # (N_POINT_SAMPLES, 3) in m
            self.list_pc_nrm.append(torch.from_numpy(pc_nrm).float())

            lmk2d = np.load(f"{lmk_seg_dir}/lmk2d_img/{id_frame:05d}.npy")   # (21, 2) in image space
            self.list_lmk2d.append(torch.from_numpy(lmk2d).float())

            lmk3d_rel = np.load(f"{lmk_seg_dir}/lmk3d_rel/{id_frame:05d}.npy")   # (21, 3) in root-relative camera space
            self.list_lmk3d_rel.append(torch.from_numpy(lmk3d_rel).float())

            lmk3d_cam = np.load(f"{lmk_seg_dir}/lmk3d_cam/{id_frame:05d}.npy")   # (21, 3) in camera space, with units in m
            self.list_lmk3d_cam.append(torch.from_numpy(lmk3d_cam).float())

            K = np.load(f"{lmk_seg_dir}/K/{id_frame:05d}.npy")  # (3, 3)
            # K[0, 0] *= -1
            self.list_K.append(torch.from_numpy(K).float())

        self.num_data = len(self.list_id_frame)

    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, id_frame):
        return {
            "id_frame": self.list_id_frame[id_frame],
            "rgba": self.list_rgba[id_frame].to(self.cfg.DEVICE),
            "depth": self.list_depth[id_frame].to(self.cfg.DEVICE),
            "pc_pos": self.list_pc_pos[id_frame].to(self.cfg.DEVICE),
            "pc_nrm": self.list_pc_nrm[id_frame].to(self.cfg.DEVICE),
            "lmk2d": self.list_lmk2d[id_frame].to(self.cfg.DEVICE),
            "lmk3d_rel": self.list_lmk3d_rel[id_frame].to(self.cfg.DEVICE),
            "lmk3d_cam": self.list_lmk3d_cam[id_frame].to(self.cfg.DEVICE),
            "K": self.list_K[id_frame].to(self.cfg.DEVICE)
        }
    
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
        self.list_mv = []
        self.list_mvp = []
        self.list_campos = []

        for data in self.dataset:
            rgba = data["rgba"].cpu().numpy()
            img_height, img_width = rgba.shape[:2]
            K = data["K"].cpu().numpy()

            # create projection matrix from camera intrinsics
            # Ref: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
            proj_mat = utils.opengl_persp_proj_mat_from_K(K, self.cfg.CAM_NEAR_FAR[0], self.cfg.CAM_NEAR_FAR[1], img_height, img_width)
            proj_mat = torch.from_numpy(proj_mat).float().to(self.cfg.DEVICE)

            mv = torch.eye(4).to(self.cfg.DEVICE)
            mvp = proj_mat @ mv

            campos = torch.linalg.inv(mv)[:3, 3]    # (3,)

            self.list_mv.append(mv)
            self.list_mvp.append(mvp)
            self.list_campos.append(campos)

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
        # depth = dr.antialias(depth, rast, v_clip, self.mano.faces.int())    # (1, H, W, 1)  # Note: this is necessary for gradients wrt silhouette
        depth = depth[0, :, :, 0]   # (H, W)

        # m to mm
        # depth = 1000 * depth

        # invert Z
        depth = -1 * depth

        return depth, mask

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

    @torch.no_grad()
    def initialize_global_pose(self, data):        
        id_frame = data["id_frame"]

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = data["pc_pos"].cpu().numpy(); color_pos = "green"; pts_nrm = data["pc_nrm"].cpu().numpy(); color_nrm = "red"
            scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            skip = 10; scale = 1/100
            list_scat_data_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            fig = go.Figure([scat_data_pos, *list_scat_data_nrm])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()
            exit()

        self.list_global_rot[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        self.list_global_transl[id_frame].data = torch.zeros(3, device=self.cfg.DEVICE)
        vert, lmk3d = self.forward_mano(data)

        # ==============================================================================================
        #  Use landmark to get initial global pose
        # ==============================================================================================
        id_lmk_palm = [
            0,                  # root
            13, 1, 4, 10, 7,    # mcp: thumb, index, middle, ring, pinky
        ]
        lmk3d_mano_palm = lmk3d[id_lmk_palm].cpu().numpy()
        lmk3d_data = data["lmk3d_cam"].cpu().numpy()
        lmk3d_data_palm = lmk3d_data[id_lmk_palm]
        pc_lmk3d_data_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_data_palm))
        pc_lmk3d_mano_palm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lmk3d_mano_palm))
        corres = np.repeat(np.arange(len(lmk3d_data_palm))[:, None], 2, axis=1)
        corres = o3d.utility.Vector2iVector(corres)
        trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T_glob_global = trans_est.compute_transformation(pc_lmk3d_mano_palm, pc_lmk3d_data_palm, corres)

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = data["pc_pos"].cpu().numpy(); color_pos = "green"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            
            # vert_np = utils.apply_proj_mat(vert_np, T_glob_global)
            vert_tranf = utils.apply_proj_mat(vert, torch.from_numpy(T_glob_global).float().to(self.cfg.DEVICE))
            vert_np = vert_tranf.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
            
            pts_pos = lmk3d_data
            scat_lmk3d_data = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=5, color="brown"), showlegend=False)
            
            fig = go.Figure([scat_pc_data_pos, plotlymesh, scat_lmk3d_data])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()
            exit()

        # ==============================================================================================
        #  ICP to register mesh to depth pointcloud
        # ==============================================================================================
        """
        # vert_transf = utils.apply_proj_mat(vert.cpu().numpy(), T_glob_lmk)
        # o3dmesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert_transf), o3d.utility.Vector3iVector(self.base_mesh.t_pos_idx.cpu().numpy().astype(int)))
        o3dmesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert.cpu().numpy()), o3d.utility.Vector3iVector(self.base_mesh.t_pos_idx.cpu().numpy().astype(int)))
        o3dmesh.compute_vertex_normals()
        pc_mano = o3dmesh.sample_points_poisson_disk(self.cfg.INIT.N_SAMPLES_ON_MESH)
        pc_mano = pc_mano.voxel_down_sample(self.cfg.INIT.PC_VOXEL_SIZE)
        # back face culling
        mesh_culled, list_id_pt_vis = pc_mano.hidden_point_removal(np.zeros(3), 10)
        pc_mano = pc_mano.select_by_index(list_id_pt_vis)
        """
        ## Matching: Projection-based
        depth_ren, mask_ren = self.render_depth(utils.apply_proj_mat(vert, torch.from_numpy(T_glob_global).float().to(self.cfg.DEVICE)), self.list_mvp[data["id_frame"]])    # (H, W)
        # depth_ren = depth_ren.cpu().numpy(); mask_ren = mask_ren.cpu().numpy()
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            fig = go.Figure(go.Heatmap(z=depth_ren.cpu().numpy()))
            fig.update_layout(width=depth_ren.shape[1], height=depth_ren.shape[0])
            fig.update_yaxes(autorange='reversed')
            fig.show()
            exit()

        depth_data_raw = data["depth"].cpu().numpy()
        depth_data = cv.bilateralFilter(depth_data_raw, -1, 10, 10)
        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            fig = go.Figure(go.Heatmap(z=depth_data_raw))
            fig.update_layout(width=depth_data_raw.shape[1], height=depth_data_raw.shape[0])
            fig.update_yaxes(autorange='reversed')
            fig.show()

            fig = go.Figure(go.Heatmap(z=depth_data))
            fig.update_layout(width=depth_data.shape[1], height=depth_data.shape[0])
            fig.update_yaxes(autorange='reversed')
            fig.show()
            exit()

        # K = data["K"].cpu().numpy()
        # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # mask_depth_data = data["rgba"][:, :, 3].cpu().numpy().astype(bool)
        # mask_depth_common = mask_ren & mask_depth_data  # use correspondences present in both depth maps (this has the effect of boundary point rejection)
        # xyz_ren = utils.depth_to_pointcloud(depth_ren, mask_depth_common, fx, fy, cx, cy)
        # xyz_data = utils.depth_to_pointcloud(depth_data, mask_depth_common, fx, fy, cx, cy)

        K = data["K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        mask_depth_data = data["rgba"][:, :, 3].bool()
        mask_depth_common = mask_ren & mask_depth_data  # use correspondences present in both depth maps (this has the effect of boundary point rejection)
        xyz_ren = utils.depth_to_xyz(depth_ren, mask_depth_common, fx, fy, cx, cy)
        # xyz_ren = xyz_ren / 1000
        R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
        T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
        xyz_ren = utils.apply_proj_mat(xyz_ren, torch.from_numpy(T_kin_to_gl).float().to(xyz_ren.device))

        
        xyz_data = utils.depth_to_xyz(torch.fliplr(data["depth"]), torch.fliplr(mask_depth_common), fx, fy, cx, cy)
        # data["depth"] is in mm and uses kinect camera convention (kinect (based on plot, after fliplr):       x: right,   y: down,    z: forward)
        # convert it to m and rotate to use opengl camera convention (opengl/blender/nvdiffrast:                x: right,   y: up,      z: backward)
        xyz_data = xyz_data / 1000
        R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
        T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
        xyz_data = utils.apply_proj_mat(xyz_data, torch.from_numpy(T_kin_to_gl).float().to(xyz_data.device))

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


        # # intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_ren.shape[1], depth_ren.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        # # pcd_raw = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image((depth_ren*(-1000)).astype(np.uint16)), intrinsic)    # this function requires depth to be in uint16
        # # xyz_ren = np.asarray(pcd_raw.points)
        # R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
        # T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
        # # xyz_ren = utils.apply_proj_mat(xyz_ren, T_kin_to_gl)

        DEBUG_LOCAL = True
        if DEBUG_LOCAL:
            pts_pos = xyz_data.cpu().numpy(); color_pos = "green"; pts_nrm = nrm_data.cpu().numpy(); color_nrm = "red"
            scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            # list_scat_data_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            pts_pos = xyz_ren.cpu().numpy(); color_pos = "yellow"; pts_nrm = nrm_ren.cpu().numpy(); color_nrm = "magenta"
            scat_ren_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            # list_scat_ren_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            pts_pos = lmk3d_data
            scat_lmk3d_data = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=5, color="brown"), showlegend=False)
            
            # fig = go.Figure([scat_data_pos, *list_scat_data_nrm, scat_ren_pos, *list_scat_ren_nrm, scat_lmk3d_data])
            fig = go.Figure([scat_data_pos, scat_ren_pos, scat_lmk3d_data])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()
            exit()

        
        pc_data = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_data.cpu().numpy()))
        pc_data.normals = o3d.utility.Vector3dVector(nrm_data.cpu().numpy())
        # pc_data = pc_data.voxel_down_sample(self.cfg.INIT.PC_VOXEL_SIZE)

        pc_ren = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_ren.cpu().numpy()))
        pc_ren.normals = o3d.utility.Vector3dVector(nrm_ren.cpu().numpy())
        # pc_ren = pc_ren.voxel_down_sample(self.cfg.INIT.PC_VOXEL_SIZE)

        # fpfh_pc_mano = o3d.pipelines.registration.compute_fpfh_feature(pc_mano, o3d.geometry.KDTreeSearchParamHybrid(radius=self.cfg.INIT.PC_VOXEL_SIZE*5, max_nn=100))
        # fpfh_pc_data = o3d.pipelines.registration.compute_fpfh_feature(pc_data, o3d.geometry.KDTreeSearchParamHybrid(radius=self.cfg.INIT.PC_VOXEL_SIZE*5, max_nn=100))
        # result_global = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(pc_mano, pc_data, fpfh_pc_mano, fpfh_pc_data, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=self.cfg.INIT.PC_VOXEL_SIZE*50))
        # print(result_global)
        # T_glob_global = result_global.transformation.copy()
        # result_local = o3d.pipelines.registration.registration_icp(pc_ren, pc_data, 1000, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # logger.debug(result_local)
        # T_glob_icp = result_local.transformation.copy()

        corres = np.repeat(np.arange(len(xyz_ren))[:, None], 2, axis=1)
        corres = o3d.utility.Vector2iVector(corres)
        trans_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T_glob_icp = trans_est.compute_transformation(pc_ren, pc_data, corres)

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            pts_pos = np.asarray(pc_data.points); color_pos = "green"; pts_nrm = np.asarray(pc_data.normals); color_nrm = "red"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            list_scat_pc_data_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
           
            pc_ren_icp = o3d.geometry.PointCloud(pc_ren).transform(T_glob_icp)
            pts_pos = np.asarray(pc_ren_icp.points); color_pos = "cyan"; pts_nrm = np.asarray(pc_ren_icp.normals); color_nrm = "magenta"
            scat_pc_ren_icp_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            list_scat_pc_ren_icp_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
           
            fig = go.Figure([scat_pc_data_pos, *list_scat_pc_data_nrm, scat_pc_ren_icp_pos, *list_scat_pc_ren_icp_nrm])
            # fig = go.Figure([scat_pc_data_pos, scat_pc_ren_icp_pos])
            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
            fig.show()

            exit()
        
        T_glob = T_glob_icp @ T_glob_global

        R_glob = torch.from_numpy(T_glob[:3, :3]).float().to(self.cfg.DEVICE)
        t_glob = torch.from_numpy(T_glob[:3, 3]).float().to(self.cfg.DEVICE)

        # update parameters
        r_glob = so3_log_map(R_glob[None, :, :])[0]
        self.list_global_rot[id_frame].data = r_glob
        self.list_global_transl[id_frame].data = t_glob

        DEBUG_LOCAL = False
        if DEBUG_LOCAL:
            
            pts_pos = np.asarray(pc_data.points); color_pos = "green"; pts_nrm = np.asarray(pc_data.normals); color_nrm = "red"
            scat_pc_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=3, color=color_pos), showlegend=False)
            skip = 100; scale = 1/100
            list_scat_pc_data_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color=color_nrm, width=2), hoverinfo="none", showlegend=False) for start, end in zip(pts_pos[::skip], pts_pos[::skip]+scale*pts_nrm[::skip])]
            
            vert, lmk3d = self.forward_mano(data)
            vert_np = vert.cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
           
            fig = go.Figure([scat_pc_data_pos, *list_scat_pc_data_nrm, plotlymesh])
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
        # sample points on mesh
        pyt3dmesh = Meshes(verts=vert[None, :, :], faces=self.base_mesh.t_pos_idx[None, :, :])
        sample_v_pos, sample_v_nrm = sample_points_from_meshes(pyt3dmesh, num_samples=self.cfg.INIT.N_SAMPLES_ON_MESH, return_normals=True)  # returned arrays have batch dim (B, N_POINTS, 3)

        # TODO: filter corresponding points based on normals


        loss_pos, loss_nrm = chamfer_distance(
            data["pc_pos"][None, :, :], sample_v_pos,
            x_normals=data["pc_nrm"][None, :, :], y_normals=sample_v_nrm,
            single_directional=True
        )

        # loss_chamf = self.cfg.INIT.W_CHAMFER_POS * loss_pos + self.cfg.INIT.W_CHAMFER_NRM * loss_nrm

        return loss_pos, loss_nrm

    def compute_lmk3d_loss(self, data, lmk3d):
        id_frame = data["id_frame"]
        # normalize landmark to [-1, 1]
        lmk3d_norm_mano = (lmk3d - self.list_global_transl[id_frame]) / self.cfg.INIT.LMK3D_NORM_SCALE
        # lmk3d_norm_data = (data["lmk3d_cam"].clone() - self.list_global_transl[id_frame]) / self.cfg.INIT.LMK3D_NORM_SCALE
        lmk3d_norm_data = data["lmk3d_rel"].clone() / self.cfg.INIT.LMK3D_NORM_SCALE

        DEBUG = False
        if DEBUG:
            pts_pred = lmk3d_norm_mano.detach().cpu().numpy()
            scat_pred = go.Scatter3d(x=pts_pred[:, 0], y=pts_pred[:, 1], z=pts_pred[:, 2], mode="markers", marker=dict(color="blue"), name="pred", customdata=np.arange(len(pts_pred)), hovertemplate="%{customdata}<extra></extra>")
            pts_data = lmk3d_norm_data.cpu().numpy()
            scat_data = go.Scatter3d(x=pts_data[:, 0], y=pts_data[:, 1], z=pts_data[:, 2], mode="markers", marker=dict(color="green"), name="data", customdata=np.arange(len(pts_data)), hovertemplate="%{customdata}<extra></extra>")
            fig = go.Figure([scat_data, scat_pred])
            fig.update_layout(scene=dict(aspectmode="data"))
            fig.show()
            # fig.write_html(f"./output/tmp/lmk3d_norm_data_vs_pred/plot.html")
            exit()

        # only consider palm and fingertips
        id_lmk_fingertips = [
            0,                    # root
            # 13, 1, 4, 10, 7,      # mcp: thumb, index, middle, ring, pinky
            20, 16, 17, 19, 18,   # fingertips [thumb to pinky]
        ]
        lmk3d_norm_data = lmk3d_norm_data[id_lmk_fingertips]
        lmk3d_norm_mano = lmk3d_norm_mano[id_lmk_fingertips]
        
        # l1 norm
        loss_lmk3d = torch.norm(lmk3d_norm_data - lmk3d_norm_mano, dim=1, p=1).mean()
        return loss_lmk3d
    
    @torch.no_grad()
    def compute_lmk3d_error(self, data, lmk3d):
        err_lmk3d = torch.norm(data["lmk3d_cam"] - lmk3d, dim=1, p=2).mean()
        return err_lmk3d

    def compute_lmk2d_loss(self, data, lmk3d):
        lmk2d = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1])
        lmk2d[:, 0], lmk2d[:, 1] = normalize_image_points(lmk2d[:, 0], lmk2d[:, 1], self.cfg.IMG_RES)

        lmk2d_data = data["lmk2d"].clone()
        lmk2d_data[:, 0], lmk2d_data[:, 1] = normalize_image_points(lmk2d_data[:, 0], lmk2d_data[:, 1], self.cfg.IMG_RES)

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
        sil = self.render_sil(vert, self.list_mvp[data["id_frame"]])
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
        
        # return self.cfg.INIT.W_TEMP_HAND_POSE*hand_pose_temp_loss + self.cfg.INIT.W_TEMP_GLOBAL_ROT*global_rot_temp_loss + self.cfg.INIT.W_TEMP_GLOBAL_TRANSL*global_transl_temp_loss
        return hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss

    def compute_beta_reg_loss(self):
        reg_beta = self.beta ** 2
        return reg_beta.sum()
    
    def compute_laplacian_loss(self, vert):
        return regularizer.laplace_regularizer_const(vert, self.mano.faces)


    def compute_loss(self, data, vert, lmk3d):
        loss_chamf_pos, loss_chamf_nrm = self.compute_chamfer_loss(data, vert)
        lmk3d_loss = self.compute_lmk3d_loss(data, lmk3d)
        lmk2d_loss = self.compute_lmk2d_loss(data, lmk3d)
        sil_loss = self.compute_sil_loss(data, vert)
        beta_reg_loss = self.compute_beta_reg_loss()
        laplace_loss = self.compute_laplacian_loss(vert)
        hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss = self.compute_temp_loss(data["id_frame"])

        total_loss = loss_chamf_pos * self.cfg.INIT.W_CHAMFER_POS \
                    + loss_chamf_nrm * self.cfg.INIT.W_CHAMFER_NRM \
                    + lmk3d_loss * self.cfg.INIT.W_LMK3D \
                    + lmk2d_loss * self.cfg.INIT.W_LMK2D \
                    + sil_loss * self.cfg.INIT.W_SIL \
                    + beta_reg_loss * self.cfg.INIT.W_BETA_REG \
                    + laplace_loss * self.cfg.INIT.W_LAPLACE_REG \
                    + hand_pose_temp_loss * self.cfg.INIT.W_TEMP_HAND_POSE \
                    + global_rot_temp_loss * self.cfg.INIT.W_TEMP_GLOBAL_ROT \
                    + global_transl_temp_loss * self.cfg.INIT.W_TEMP_GLOBAL_TRANSL \

        return total_loss, loss_chamf_pos, loss_chamf_nrm, lmk3d_loss, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss

    def optimize(self, log_dir):
        utils.create_dir(log_dir, True)
        logger.info(f"Initialization log will be saved at {log_dir}")

        log_mesh_dir = f"{log_dir}/mesh"; utils.create_dir(log_mesh_dir, True)
        log_lmk3d_data_vs_mano_dir = f"{log_dir}/lmk3d_data_vs_mano"; utils.create_dir(log_lmk3d_data_vs_mano_dir, True)
        log_lmk2d_data_vs_mano_dir = f"{log_dir}/lmk2d_data_vs_mano"; utils.create_dir(log_lmk2d_data_vs_mano_dir, True)
        log_sil_data_vs_mano_dir = f"{log_dir}/sil_data_vs_mano"; utils.create_dir(log_sil_data_vs_mano_dir, True)
        log_pc_vs_mesh_dir = f"{log_dir}/pc_vs_mesh"; utils.create_dir(log_pc_vs_mesh_dir, True)
        path_log_loss_file = f"{log_dir}/loss.txt"

        self.setup_optimizer()

        _W_LMK3D = self.cfg.INIT.W_LMK3D
        _W_LMK2D = self.cfg.INIT.W_LMK2D

        with open(path_log_loss_file, "w") as log_loss_file:
            for data in self.dataset:
                id_frame = data["id_frame"]

                DEBUG = True
                if DEBUG:
                    self.cfg.INIT.W_TEMP = 0
                    if id_frame != 0:
                        continue
                    self.initialize_global_pose(data)
                    self.cfg.INIT.OPTIMIZE_SHAPE = False
                else:
                    if id_frame == 0:
                        self.initialize_global_pose(data)
                    else:
                        self.cfg.INIT.OPTIMIZE_SHAPE = False
                        with torch.no_grad():
                            lmk2d_data_proj = utils.clip_to_img(utils.cam_to_clip_space(data["lmk3d_cam"], self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1])
                            error = reproj_error(lmk2d_data_proj.cpu().numpy(), data["lmk2d"].cpu().numpy())
                            if error > self.cfg.INIT.REPROJ_ERROR_THRESH:
                                # cannot rely on landmark, so use previous frame's pose
                                logger.warning(f"At frame {id_frame}, data reproj error ({error:.3f}) exceeded! Initializing from previous frame.")
                                self.initialize_from_previous_frame(id_frame)
                                self.cfg.INIT.W_LMK3D = _W_LMK3D/10
                                self.cfg.INIT.W_LMK2D = _W_LMK2D/10
                            else:
                                self.initialize_global_pose(data)
                                self.cfg.INIT.W_LMK3D = _W_LMK3D
                                self.cfg.INIT.W_LMK2D = _W_LMK2D
                    

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


                if DEBUG:
                    with torch.no_grad():
                        # plot mesh
                        vert, lmk3d = self.forward_mano(data)
                        rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp[data["id_frame"]]))
                        composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                        cv.imwrite(f"{log_mesh_dir}/init.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                        # plot 3D landmark
                        lmk3d_mano = lmk3d.cpu().numpy()
                        lmk3d_data = data["lmk3d_cam"].cpu().numpy()
                        scat_mano = go.Scatter3d(x=lmk3d_mano[:, 0], y=lmk3d_mano[:, 1], z=lmk3d_mano[:, 2], mode="markers", name="lmk3d_mano", customdata=np.arange(len(lmk3d_mano)), hovertemplate="%{customdata}<extra></extra>")
                        scat_data = go.Scatter3d(x=lmk3d_data[:, 0], y=lmk3d_data[:, 1], z=lmk3d_data[:, 2], mode="markers", name="lmk3d_data", customdata=np.arange(len(lmk3d_data)), hovertemplate="%{customdata}<extra></extra>")
                        fig = go.Figure([scat_data, scat_mano])
                        fig.update_layout(scene=dict(
                            # aspectmode="cube",
                            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
                            # xaxis=dict(range=[-0.1, 0.1]),
                            # yaxis=dict(range=[-0.1, 0.1]),
                            # zaxis=dict(range=[-0.1, 0.1]),
                        ))
                        fig.write_html(f"{log_lmk3d_data_vs_mano_dir}/init.html")

                        # plot 2D landmark
                        lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                        lmk2d_data = data["lmk2d"].cpu().numpy()
                        rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                        rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))
                        cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/init.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))

                        # plot sil
                        sil_mano = self.render_sil(vert, self.list_mvp[data["id_frame"]]).cpu().numpy()
                        sil_data = data["rgba"][:, :, 3].cpu().numpy()
                        composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                        composite_sil[:, :, 1] = sil_data*255
                        composite_sil[:, :, 2] = sil_mano*255
                        cv.imwrite(f"{log_sil_data_vs_mano_dir}/init.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

                        # plot mesh + depth
                        K = data["K"]
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        mask_depth_data = data["rgba"][:, :, 3].bool()
                        xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx, fy, cx, cy) * (-1/1000)
                        pc_pos = xyz_data.cpu().numpy()
                        scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                        vert_np = vert.detach().cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
                        plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
                        # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                        fig = go.Figure([scat_pc_pos, plotlymesh])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.write_html(f"{log_pc_vs_mesh_dir}/init.html")

                
                # multiple iterations per frame
                for id_step in range(self.cfg.INIT.N_STEPS):
                    # ==============================================================================================
                    #  Forward pass
                    # ==============================================================================================
                    vert, lmk3d = self.forward_mano(data)
                    total_loss, loss_chamf_pos, loss_chamf_nrm, lmk3d_loss, lmk2d_loss, sil_loss, beta_reg_loss, laplace_loss, hand_pose_temp_loss, global_rot_temp_loss, global_transl_temp_loss = self.compute_loss(data, vert, lmk3d)

                    # ==============================================================================================
                    #  Backpropagate
                    # ==============================================================================================
                    if self.cfg.OPT.OPTIMIZE_SHAPE:
                        self.optimizer_shape.zero_grad()
                    if self.cfg.OPT.OPTIMIZE_POSE:
                        self.optimizer_pose.zero_grad()

                    total_loss.backward()

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
                    log_loss_str = f"[Frame:{id_frame:>3d}/{len(self.dataset)-1:>3d}] [Step:{id_step:>3d}/{self.cfg.INIT.N_STEPS-1:>3d}]  total_loss:{total_loss.item():>7f} | loss_chamf_pos: {loss_chamf_pos.item():>7f} | loss_chamf_nrm: {loss_chamf_nrm.item():>7f} | lmk3d_loss:{lmk3d_loss.item():>7f} | lmk2d_loss:{lmk2d_loss.item():>7f} | sil_loss:{sil_loss.item():>7f} | beta_reg_loss:{beta_reg_loss.item():>7f} | laplace_loss:{laplace_loss.item():>7f} | hand_pose_temp_loss: {hand_pose_temp_loss.item():>7f} | global_rot_temp_loss: {global_rot_temp_loss.item():>7f} | global_transl_temp_loss: {global_transl_temp_loss.item():>7f}"
                    logger.info(log_loss_str)
                    log_loss_file.write(f"{log_loss_str}\n")

                    if DEBUG:
                        with torch.no_grad():
                            # plot mesh
                            vert, lmk3d = self.forward_mano(data)
                            rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp[data["id_frame"]]))
                            composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                            cv.imwrite(f"{log_mesh_dir}/{id_step:05d}.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                            # plot 3D landmark
                            lmk3d_mano = lmk3d.cpu().numpy()
                            lmk3d_data = data["lmk3d_cam"].cpu().numpy()
                            scat_mano = go.Scatter3d(x=lmk3d_mano[:, 0], y=lmk3d_mano[:, 1], z=lmk3d_mano[:, 2], mode="markers", name="lmk3d_mano", customdata=np.arange(len(lmk3d_mano)), hovertemplate="%{customdata}<extra></extra>")
                            scat_data = go.Scatter3d(x=lmk3d_data[:, 0], y=lmk3d_data[:, 1], z=lmk3d_data[:, 2], mode="markers", name="lmk3d_data", customdata=np.arange(len(lmk3d_data)), hovertemplate="%{customdata}<extra></extra>")
                            fig = go.Figure([scat_data, scat_mano])
                            fig.update_layout(scene=dict(
                                # aspectmode="cube",
                                aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
                                # xaxis=dict(range=[-0.1, 0.1]),
                                # yaxis=dict(range=[-0.1, 0.1]),
                                # zaxis=dict(range=[-0.1, 0.1]),
                            ))
                            fig.write_html(f"{log_lmk3d_data_vs_mano_dir}/{id_step:05d}.html")

                            # plot 2D landmark
                            lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                            lmk2d_data = data["lmk2d"].cpu().numpy()
                            rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                            rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))
                            cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_step:05d}.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))

                            # plot sil
                            sil_mano = self.render_sil(vert, self.list_mvp[data["id_frame"]]).cpu().numpy()
                            sil_data = data["rgba"][:, :, 3].cpu().numpy()
                            composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                            composite_sil[:, :, 1] = sil_data*255
                            composite_sil[:, :, 2] = sil_mano*255
                            cv.imwrite(f"{log_sil_data_vs_mano_dir}/{id_step:05d}.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

                            # plot mesh + depth
                            K = data["K"]
                            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                            mask_depth_data = data["rgba"][:, :, 3].bool()
                            xyz_data = utils.depth_to_xyz(data["depth"], mask_depth_data, fx, fy, cx, cy) * (-1/1000)
                            pc_pos = xyz_data.cpu().numpy()
                            scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                            vert_np = vert.detach().cpu().numpy(); faces_np = self.mano.faces.cpu().numpy()
                            plotlymesh = go.Mesh3d(x=vert_np[:, 0], y=vert_np[:, 1], z=vert_np[:, 2], i=faces_np[:, 0], j=faces_np[:, 1], k=faces_np[:, 2], color="cyan", flatshading=True, hoverinfo="none", opacity=0.5)
                            # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                            fig = go.Figure([scat_pc_pos, plotlymesh])
                            fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                            fig.write_html(f"{log_pc_vs_mesh_dir}/{id_step:05d}.html")

                if DEBUG:
                    exit()

                with torch.no_grad():
                    # plot mesh
                    vert, lmk3d = self.forward_mano(data)
                    rgb, alpha = self.rasterize_mesh(utils.cam_to_clip_space(vert, self.list_mvp[data["id_frame"]]))
                    composite_img = (utils.alpha_composite(data["rgba"].cpu().numpy(), torch.cat([rgb, alpha[:, :, None]], dim=2).cpu().numpy(), 1)*255).astype(np.uint8)
                    cv.imwrite(f"{log_mesh_dir}/{id_frame:05d}.png", cv.cvtColor(composite_img, cv.COLOR_RGBA2BGRA))

                    # plot 3D landmark
                    lmk3d_mano = lmk3d.cpu().numpy()
                    lmk3d_data = data["lmk3d_cam"].cpu().numpy()
                    scat_mano = go.Scatter3d(x=lmk3d_mano[:, 0], y=lmk3d_mano[:, 1], z=lmk3d_mano[:, 2], mode="markers", name="lmk3d_mano", customdata=np.arange(len(lmk3d_mano)), hovertemplate="%{customdata}<extra></extra>")
                    scat_data = go.Scatter3d(x=lmk3d_data[:, 0], y=lmk3d_data[:, 1], z=lmk3d_data[:, 2], mode="markers", name="lmk3d_data", customdata=np.arange(len(lmk3d_data)), hovertemplate="%{customdata}<extra></extra>")
                    fig = go.Figure([scat_data, scat_mano])
                    fig.update_layout(scene=dict(
                        # aspectmode="cube",
                        aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
                        # xaxis=dict(range=[-0.1, 0.1]),
                        # yaxis=dict(range=[-0.1, 0.1]),
                        # zaxis=dict(range=[-0.1, 0.1]),
                    ))
                    fig.write_html(f"{log_lmk3d_data_vs_mano_dir}/{id_frame:05d}.html")

                    # plot 2D landmark
                    lmk2d_mano = utils.clip_to_img(utils.cam_to_clip_space(lmk3d, self.list_mvp[data["id_frame"]]), self.cfg.IMG_RES[0], self.cfg.IMG_RES[1]).cpu().numpy()
                    lmk2d_data = data["lmk2d"].cpu().numpy()
                    rgb_lmk2d_plot = utils.draw_pts_on_img(data["rgba"].cpu().numpy()[:, :, :3]*255, lmk2d_mano, radius=5, color=(0, 0, 255))
                    rgb_lmk2d_plot = utils.draw_pts_on_img(rgb_lmk2d_plot, lmk2d_data, radius=5, color=(0, 255, 0))
                    cv.imwrite(f"{log_lmk2d_data_vs_mano_dir}/{id_frame:05d}.png", cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))

                    # plot sil
                    sil_mano = self.render_sil(vert, self.list_mvp[data["id_frame"]]).cpu().numpy()
                    sil_data = data["rgba"][:, :, 3].cpu().numpy()
                    composite_sil = np.zeros((self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 3), dtype=np.uint8)
                    composite_sil[:, :, 1] = sil_data*255
                    composite_sil[:, :, 2] = sil_mano*255
                    cv.imwrite(f"{log_sil_data_vs_mano_dir}/{id_frame:05d}.png", cv.cvtColor(composite_sil, cv.COLOR_RGB2BGR))

        height_rgba_seg, width_rgba_seg = self.dataset[0]["rgba"].cpu().numpy().shape[:2]
        subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_rgba_seg}", f"{height_rgba_seg}", f"{log_mesh_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{log_lmk2d_data_vs_mano_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{log_sil_data_vs_mano_dir}"])

    @torch.no_grad()
    def save_results(self, out_dir):
        utils.create_dir(out_dir, True)
        logger.info(f"Initialization results will be saved at {out_dir}")

        out_mvp_dir = f"{out_dir}/mvp"; utils.create_dir(out_mvp_dir, True)
        out_beta_dir = f"{out_dir}/beta"; utils.create_dir(out_beta_dir, True)
        out_offset_dir = f"{out_dir}/offset"; utils.create_dir(out_offset_dir, True)
        out_global_rot_dir = f"{out_dir}/global_rot"; utils.create_dir(out_global_rot_dir, True)
        out_global_transl_dir = f"{out_dir}/global_transl"; utils.create_dir(out_global_transl_dir, True)
        out_hand_pose_dir = f"{out_dir}/hand_pose"; utils.create_dir(out_hand_pose_dir, True)
        out_vert_dir = f"{out_dir}/vert"; utils.create_dir(out_vert_dir, True)
        out_rgba_seg_dir = f"{out_dir}/rgba_seg"; utils.create_dir(out_rgba_seg_dir, True)

        np.save(f"{out_beta_dir}/beta.npy", self.beta.cpu().numpy())
        np.save(f"{out_offset_dir}/offset.npy", self.offset.cpu().numpy())
        
        for id_frame, data in enumerate(tqdm(self.dataset, desc="Writing initialization")):
            np.save(f"{out_global_rot_dir}/{id_frame:05d}.npy", self.list_global_rot[id_frame].cpu().numpy())
            np.save(f"{out_global_transl_dir}/{id_frame:05d}.npy", self.list_global_transl[id_frame].cpu().numpy())
            np.save(f"{out_hand_pose_dir}/{id_frame:05d}.npy", self.list_hand_pose[id_frame].cpu().numpy())
            
            np.save(f"{out_mvp_dir}/{id_frame:05d}.npy", self.list_mvp[id_frame].cpu().numpy())

            vert, lmk3d = self.forward_mano(self.dataset[id_frame])
            np.save(f"{out_vert_dir}/{id_frame:05d}.npy", vert.cpu().numpy())

            # segment image using mesh
            if self.cfg.INIT.SEG_USING_MESH:
                rgb = (data["rgba"].cpu().numpy()*255).astype(np.uint8)[:, :, :3]
                img_height, img_width = rgb.shape[:2]
                vert_img = utils.clip_to_img(utils.cam_to_clip_space(vert.cpu().numpy(), self.list_mvp[id_frame].cpu().numpy()), img_height, img_width)
                mask = obtain_mask_from_verts_img(rgb, vert_img.astype(np.int32), self.mano.faces.cpu().numpy())
                mask = mask[:, :, 0] == 255
                rgb_seg = rgb*mask[:, :, None] + np.zeros_like(rgb)*(~mask[:, :, None])   # (H, W, 3)
                rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)
                cv.imwrite(f"{out_rgba_seg_dir}/{id_frame:05d}.png",  cv.cvtColor(rgba_seg, cv.COLOR_RGBA2BGRA))
            else:
                cv.imwrite(f"{out_rgba_seg_dir}/{id_frame:05d}.png",  cv.cvtColor((data["rgba"].cpu().numpy()*255).astype(np.uint8), cv.COLOR_RGBA2BGRA))




