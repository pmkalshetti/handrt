import numpy as np
from pathlib import Path
import cv2 as cv
import subprocess
from tqdm import tqdm
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
import torch
import open3d as o3d
from scipy import interpolate

from .. import utils
from ..render import util
from lib.optimizer.mano_initializer import ManoInitializer, reproj_error

logger = utils.get_logger(__name__)

def download_segment_anything_model(model_path):
    logger.info(f"Downloading Segment Anything Model to {model_path}")
    utils.create_dir(Path(model_path).parent, True)
    link_to_model_file = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    # response = requests.get(link_to_model_file)
    # response.raise_for_status()

    # with open(model_path, "wb") as file:
    #     file.write(response.content)
    subprocess.run(["wget", "--output-document", model_path, link_to_model_file])

def fill_K(fx, fy, cx, cy):
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

    return K

def get_kinecvtv2_stored_depth_camera_intrinsics():
    fx = 366.085
    fy = 366.085
    cx = 259.229
    cy = 207.968

    
    # return fx, fy, cx, cy
    K = fill_K(fx, fy, cx, cy)
    return K


def get_kinecvtv2_stored_color_camera_intrinsics():
    fx = 1081.372
    fy = 1081.372
    cx = 959.500
    cy = 539.500

    # return fx, fy, cx, cy
    K = fill_K(fx, fy, cx, cy)
    return K

def get_depth_to_color_stereo_extrinsic_transform_4x4():
    R = np.array([
        [0.9999589974098119, -0.0058757808714647985, 0.006890478816052757],
        [0.00575462910860141, 0.9998307848051879, 0.01747243542452615],
        [-0.006991977044182909, -0.01743206685944978, 0.9998236020928998]
    ])

    t = np.array([
        0.09800363751345595,
        -0.05476590976028551,
        0.010414303416182925
    ])

    T = utils.create_4x4_trans_mat_from_R_t(R, t)
    return T.astype(np.float32)

def mp_normalized_landmark_to_np_image_landmark(list_landmark_normalized, img_shape):
    list_x = []
    list_y = []
    list_z = []
    for landmark_normalized in list_landmark_normalized:
        # Ref: https://github.com/google/mediapipe/blob/cbf1d97429153fe0797ca5648712f71e0bd70bf8/mediapipe/python/solutions/drawing_utils.py#L49
        x = min(np.floor(landmark_normalized.x * img_shape[1]), img_shape[1]-1)
        y = min(np.floor(landmark_normalized.y * img_shape[0]), img_shape[0]-1)
        
        list_x.append(x)
        list_y.append(y)
        list_z.append(landmark_normalized.z)

    return np.stack([list_x, list_y, list_z], axis=1).astype(np.float32)

def mp_world_landmark_to_np_world_landmark(list_landmark_world):
    list_x = []
    list_y = []
    list_z = []
    for landmark_world in list_landmark_world:
        list_x.append(landmark_world.x)
        list_y.append(landmark_world.y)
        list_z.append(landmark_world.z)
    return np.stack([list_x, list_y, list_z], axis=1).astype(np.float32)

def bbox_from_landmark_img(lmk2d_img, img_shape, pad=10):
    bbox_xyxy = np.zeros(4, dtype=int)
    bbox_xyxy[0] = max(np.amin(lmk2d_img[:, 0]) - pad, 0)
    bbox_xyxy[1] = max(np.amin(lmk2d_img[:, 1]) - pad, 0)
    bbox_xyxy[2] = min(np.amax(lmk2d_img[:, 0]) + pad, img_shape[1]-1)
    bbox_xyxy[3] = min(np.amax(lmk2d_img[:, 1]) + pad, img_shape[0]-1)

    return bbox_xyxy

def get_crop_affine_trans(bbox_xyxy, scale_crop, out_img_res):
    # crop and resize image
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    # aspect ratio preserving bbox
    c_x = x1 + w / 2.
    c_y = y1 + h / 2.
    aspect_ratio = out_img_res[1] / out_img_res[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox_xywh = np.array([
        c_x - w / 2.,
        c_y - h / 2.,
        w,
        h
    ])
    bb_c_x = float(bbox_xywh[0] + 0.5*bbox_xywh[2])
    bb_c_y = float(bbox_xywh[1] + 0.5*bbox_xywh[3])
    bb_width = float(bbox_xywh[2])
    bb_height = float(bbox_xywh[3])

    src_w = bb_width * scale_crop
    src_h = bb_height * scale_crop
    src_center = np.array([bb_c_x, bb_c_y], np.float32)
    src_downdir = np.array([0, src_h*0.5], dtype=np.float32)
    src_rightdir = np.array([src_w*0.5, 0], dtype=np.float32)
    
    dst_w = out_img_res[1]
    dst_h = out_img_res[0]
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_2x3 = cv.getAffineTransform(np.float32(src), np.float32(dst))
    trans_2x3 = trans_2x3.astype(np.float32)

    return trans_2x3, bbox_xywh

def clip_and_normalize_depth_to_cm_w_alpha(depth, depth_near, depth_far):
    mask = (depth > depth_near) & (depth < depth_far)
    depth_clip = np.clip(depth, depth_near, depth_far)
    depth_norm = (depth_clip - depth_near) / (depth_far - depth_near)
    
    depth_cm_single_channel = (depth_norm*255).astype(np.uint8) # (H, W)
    depth_cm_single_channel[~mask] = 0
    depth_cm = np.stack((depth_cm_single_channel,)*3, axis=2)   # (H, W, 3)

    # depth_cm_w_alpha = depth_cm * mask[:, :, None] + np.zeros_like(depth_cm)*(~mask[:, :, None])
    depth_cm_w_alpha = np.concatenate([depth_cm, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

    return depth_cm_w_alpha

def compute_landmark_and_segment_hand(cfg, lmk_seg_dir):
    # ==============================================================================================
    # Load Segment Anything model
    # ==============================================================================================
    if not Path(cfg.SEGMENT_ANYTHING_MODEL_PATH).exists():
        download_segment_anything_model(cfg.SEGMENT_ANYTHING_MODEL_PATH)
    logger.info(f"Load SamPredictor (start)...")
    sam = sam_model_registry["vit_h"](checkpoint=cfg.SEGMENT_ANYTHING_MODEL_PATH)
    sam.to(device=cfg.DEVICE)
    predictor = SamPredictor(sam)
    logger.info(f"Load SamPredictor (complete)")

    # fx, fy, cx, cy = get_kinecvtv2_stored_ir_camera_intrinsics()
    # K_orig = np.array([
    #     [fx,  0, cx],
    #     [ 0, fy, cy],
    #     [ 0,  0,  1]
    # ], dtype=np.float32)

    K_orig_rgb = get_kinecvtv2_stored_color_camera_intrinsics()
    K_orig_depth = get_kinecvtv2_stored_depth_camera_intrinsics()

    # reorder mediapipe landmarks to mano's ordering
    mp_to_mano_lmk_ordering = [
        0,                  # wirst
        5, 6, 7,            # index
        9, 10, 11,          # middle
        17, 18, 19,         # pinky
        13, 14, 15,         # ring
        1, 2, 3,            # thumb
        8, 12, 20, 16, 4    # tips: index, middle, pinky, ring, thumb
    ]

    # ==============================================================================================
    # output directories
    # ==============================================================================================
    out_dir = lmk_seg_dir; utils.create_dir(out_dir, True)
    out_rgb_raw_dir = f"{out_dir}/rgb_raw"; utils.create_dir(out_rgb_raw_dir, True)
    out_rgb_lmk2d_plot_dir = f"{out_dir}/rgb_lmk2d_plot"; utils.create_dir(out_rgb_lmk2d_plot_dir, True)
    out_rgb_bbox_plot_dir = f"{out_dir}/rgb_bbox_plot"; utils.create_dir(out_rgb_bbox_plot_dir, True)
    out_rgba_seg_dir = f"{out_dir}/rgba_seg"; utils.create_dir(out_rgba_seg_dir, True)
    out_rgb_seg_lmk2d_plot_dir = f"{out_dir}/rgb_seg_lmk2d_plot"; utils.create_dir(out_rgb_seg_lmk2d_plot_dir, True)
    # out_lmk3d_img_seg_dir = f"{out_dir}/lmk3d_img_seg"; utils.create_dir(out_lmk3d_img_seg_dir, True)
    out_lmk2d_img_dir = f"{out_dir}/lmk2d_img"; utils.create_dir(out_lmk2d_img_dir, True)
    out_lmk3d_rel_dir = f"{out_dir}/lmk3d_rel"; utils.create_dir(out_lmk3d_rel_dir, True)
    out_lmk3d_cam_dir = f"{out_dir}/lmk3d_cam"; utils.create_dir(out_lmk3d_cam_dir, True)
    out_K_dir = f"{out_dir}/K"; utils.create_dir(out_K_dir, True)
    out_rgb_seg_lmk2d_proj_plot_dir = f"{out_dir}/rgb_seg_lmk2d_proj_plot"; utils.create_dir(out_rgb_seg_lmk2d_proj_plot_dir, True)
    # out_lmk3d_rel_plot_dir = f"{out_dir}/lmk3d_rel_plot"; utils.create_dir(out_lmk3d_rel_plot_dir, True)
    # out_lmk3d_cam_plot_dir = f"{out_dir}/lmk3d_cam_plot"; utils.create_dir(out_lmk3d_cam_plot_dir, True)
    path_reprog_err_file = f"{out_dir}/reprog_err.txt"
    out_depth_raw_npy_dir = f"{out_dir}/depth_raw_npy"; utils.create_dir(out_depth_raw_npy_dir, True)
    out_depth_raw_cm_dir = f"{out_dir}/depth_raw_cm"; utils.create_dir(out_depth_raw_cm_dir, True)
    out_depth_seg_npy_dir = f"{out_dir}/depth_seg_npy"; utils.create_dir(out_depth_seg_npy_dir, True)
    out_depth_seg_cm_dir = f"{out_dir}/depth_seg_cm"; utils.create_dir(out_depth_seg_cm_dir, True)
    # out_pc_plot_dir = f"{out_dir}/pc_plot"; utils.create_dir(out_pc_plot_dir, True)
    # out_pc_pos_dir = f"{out_dir}/pc_pos"; utils.create_dir(out_pc_pos_dir, True)
    # out_pc_nrm_dir = f"{out_dir}/pc_nrm"; utils.create_dir(out_pc_nrm_dir, True)

    logger.info(f"Output will be stored in {out_dir}")

    # ==============================================================================================
    # process frames
    # ==============================================================================================
    in_dir = f"{cfg.KINECT.DATA_ROOT_DIR}/{cfg.KINECT.USER}/{cfg.KINECT.SEQUENCE}"
    list_path_to_frame = sorted(list(Path(f"{in_dir}/color").glob("*.png")))
    if cfg.KINECT.END_FRAME_ID != -1:
        list_path_to_frame = list_path_to_frame[:cfg.KINECT.END_FRAME_ID+1]
    list_path_to_frame = list_path_to_frame[cfg.KINECT.START_FRAME_ID:]

    logger.info(f"Segmenting hand using MediaPipe landmarks")
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands:
        with open(path_reprog_err_file, "w") as file_reprog_err:
            id_frame_out = 0
            for id_frame_in, path_to_frame in enumerate(tqdm(list_path_to_frame)):
                rgb = cv.cvtColor(cv.imread(f"{in_dir}/color_reg/{id_frame_in:05d}.png"), cv.COLOR_BGR2RGB)  # (424, 512, 3)
                depth = np.load(f"{in_dir}/depth_npy/{id_frame_in:05d}.npy")  # (424, 512)
                K = K_orig.copy()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fig = go.Figure(go.Heatmap(z=depth))
                    fig.update_layout(width=depth.shape[1], height=depth.shape[0])
                    fig.update_yaxes(autorange='reversed')
                    fig.show()
                    exit()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    xyz_data = utils.depth_to_xyz_np(depth, depth>0, fx, fy, cx, cy)
                    pts_pos = xyz_data; color_pos = "green"
                    scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                    fig = go.Figure([scat_data_pos])
                    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                    fig.show()
                    exit()
                    # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward
                DEBUG = False
                if DEBUG:
                    fig = go.Figure(utils.rgbd_as_plotly_surface(rgb, depth))
                    fig.show()
                    exit()
                FLIP = True
                if FLIP:
                    rgb = np.fliplr(rgb).copy()
                    depth = np.fliplr(depth).copy()

                    # update intrinsics correspondingly
                    # Ref: https://ksimek.github.io/2012/08/14/decompose/
                    # If the image x-axis and camera x-axis point in opposite directions, negate the first column of K and the first row of R.                
                    # Ref: https://stackoverflow.com/a/74753976
                    # K[0, 0] *= -1

                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth))
                        fig.update_layout(width=depth.shape[1], height=depth.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        exit()
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        xyz_data = utils.depth_to_xyz_np(depth, depth>0, fx, fy, cx, cy)
                        pts_pos = xyz_data; color_pos = "green"
                        scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                        fig = go.Figure([scat_data_pos])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.show()
                        exit()
                        # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward
                        # no need to change K (no need to negate fx)

                # ==============================================================================================
                #  Compute landmark
                # ==============================================================================================
                # to improve performance, optionally mark image as not writeable to pass by reference
                rgb.flags.writeable = False
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    logger.warning(f"No landmark for {path_to_frame}")
                    continue
                rgb.flags.writeable = True

                # results convention: https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md#multi_hand_landmarks
                # multi_hand_landmarks are in image space (x,y in normalized image space, z represents depth wrt wrist (smaller value corresponds to close to camera, magnitude of z is roughly same scale as x))
                lmk3d_img = mp_normalized_landmark_to_np_image_landmark(results.multi_hand_landmarks[0].landmark, rgb.shape)    # this scales x,y to image space
                lmk3d_img = lmk3d_img[mp_to_mano_lmk_ordering]  # (21, 3)
                lmk2d_img = lmk3d_img[:, :2]    # (21, 2)
                
                rgb_lmk2d_plot = utils.draw_pts_on_img(rgb, lmk2d_img, radius=10)

                bbox_xyxy = bbox_from_landmark_img(lmk2d_img, rgb.shape, pad=30).astype(int)    # (4,)
                # plot bbox on img
                rgb_bbox_plot = rgb.copy()
                cv.rectangle(rgb_bbox_plot, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 3)
                
                # 3D landmarks, root relative normalized
                lmk3d_rel = mp_world_landmark_to_np_world_landmark(results.multi_hand_world_landmarks[0].landmark)
                lmk3d_rel = lmk3d_rel[mp_to_mano_lmk_ordering]
                
                DEBUG = False
                if DEBUG:
                    # check if final reprojection is low, and also print initial reprojection error
                    log_dir = f"{out_dir}/landmark_cam_space"; utils.create_dir(log_dir, True)

                    # initial reprojection
                    lmk2d_proj_rel = utils.apply_Krt(lmk3d_rel, K, r=np.zeros(3), t=np.zeros(3))[:, :2]
                    # img_height, img_width = rgb.shape[:2]; proj_mat = utils.opengl_persp_proj_mat_from_K(K, cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1], img_height, img_width); mv = np.eye(4); mvp = proj_mat @ mv
                    # lmk2d_proj_rel = utils.clip_to_img(utils.cam_to_clip_space(lmk3d_rel, mvp), img_height, img_width)
                
                    # chech if it matches with opencv function
                    lmk2d_proj_rel_cvproj, _ = cv.projectPoints(lmk3d_rel, np.zeros(3), np.zeros(3), K, None)
                    lmk2d_proj_rel_cvproj = lmk2d_proj_rel_cvproj[:, 0, :]
                    assert np.allclose(lmk2d_proj_rel, lmk2d_proj_rel_cvproj)
                    # calculate initial reprojection error
                    error = reproj_error(lmk2d_proj_rel, lmk2d_img)
                    logger.debug(f"Reprojection error (init) = {error:.3f}")
                    # plot 3D points
                    pts = lmk3d_rel; name = "lmk3d_rel"
                    fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name))
                    fig.update_layout(scene=dict(aspectmode="data"))
                    fig.write_html(f"{log_dir}/{name}.html")
                    # plot projected points (initial/relative space) on image
                    rgb_plot_lmk2d_proj_rel = utils.draw_pts_on_img(rgb, lmk2d_proj_rel, radius=1)
                    cv.imwrite(f"{log_dir}/rgb_plot_lmk2d_proj_rel.png", cv.cvtColor(rgb_plot_lmk2d_proj_rel, cv.COLOR_RGB2BGR))

                # calculate transformation from object space to camera space
                # Note about cv.solvePnP: Numpy array slices won't work as input because solvePnP requires contiguous arrays (Ref: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
                # (although the below line works without this, why?)
                success, r_pnp, t_pnp = cv.solvePnP(np.ascontiguousarray(lmk3d_rel).astype(np.float32), np.ascontiguousarray(lmk2d_img).astype(np.float32), K, None, flags=cv.SOLVEPNP_SQPNP)
                assert success
                r_pnp = r_pnp[:, 0]; t_pnp = t_pnp[:, 0]    # solvepnp outputs extra dim, i.e., (3, 1) -> (3,) [not required when providing initial guess]
                
                # solvepnp assumes opencv camera convention
                # we use opengl camera convention to project/render/rasterize in nvdiffrast
                # opencv cam                    x: right,   y: down,    z: forward
                # opengl/blender/nvdiffrast:    x: right,   y: up,      z: backward (Ref: https://nvlabs.github.io/nvdiffrast/#coordinate-systems)
                # kinect:                       x: left,    y: up,      z: forward  (Ref: https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn785530(v=ieb.10))
                r_obj_to_cv = r_pnp; t_obj_to_cv = t_pnp
                R_obj_to_cv = Rotation.from_rotvec(r_obj_to_cv).as_matrix()
                T_obj_to_cv = utils.create_4x4_trans_mat_from_R_t(R_obj_to_cv, t_obj_to_cv)
                #
                R_cv_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_cv_to_gl = utils.create_4x4_trans_mat_from_R_t(R_cv_to_gl)
                T_obj_to_gl = T_cv_to_gl @ T_obj_to_cv
                # apply transformation
                lmk3d_cam_cv = lmk3d_rel @ Rotation.from_rotvec(r_pnp).as_matrix().T + t_pnp
                lmk3d_cam = utils.apply_proj_mat(lmk3d_rel, T_obj_to_gl)
                lmk3d_rel = utils.apply_proj_mat(lmk3d_rel, T_cv_to_gl) # ensure this is done at last
                # lmk3d_cam_gl_method2 = (lmk3d_cam_cv - t_pnp) @ R_cv_to_gl.T + t_pnp @ R_cv_to_gl.T
                # assert np.allclose(lmk3d_cam, lmk3d_cam_gl_method2)
                
                if DEBUG:
                    # check final reprojection error

                    # project transformed points (given in camera space)
                    lmk2d_proj_cam_cv = utils.apply_Krt(lmk3d_cam_cv, K, r=np.zeros(3), t=np.zeros(3))[:, :2]
                    img_height, img_width = rgb.shape[:2]; proj_mat = utils.opengl_persp_proj_mat_from_K(K, cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1], img_height, img_width); mv = np.eye(4); mvp = proj_mat @ mv
                    lmk2d_proj_cam = utils.clip_to_img(utils.cam_to_clip_space(lmk3d_cam, mvp), img_height, img_width)
                    assert np.allclose(lmk2d_proj_cam_cv, lmk2d_proj_cam)
                    # calculate initial reprojection error
                    error = reproj_error(lmk2d_proj_cam, lmk2d_img)
                    logger.debug(f"Reprojection error (final) = {error:.3f}")
                    # plot 3D points in camera space
                    pts = lmk3d_cam; name = "lmk3d_cam"
                    fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name))
                    fig.update_layout(scene=dict(aspectmode="data"))
                    fig.write_html(f"{log_dir}/{name}.html")
                    # plot projected points (final/camera space) on image
                    rgb_plot_lmk2d_proj_cam = utils.draw_pts_on_img(rgb, lmk2d_proj_cam, radius=1)
                    cv.imwrite(f"{log_dir}/rgb_plot_lmk2d_proj_cam.png", cv.cvtColor(rgb_plot_lmk2d_proj_cam, cv.COLOR_RGB2BGR))

                    exit()
                
                # log final reprojection error
                # lmk2d_proj_cam_cv = utils.apply_Krt(lmk3d_cam_cv, K, r=np.zeros(3), t=np.zeros(3))[:, :2]
                img_height, img_width = rgb.shape[:2]; proj_mat = utils.opengl_persp_proj_mat_from_K(K, cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1], img_height, img_width); mv = np.eye(4); mvp = proj_mat @ mv
                lmk2d_proj_cam = utils.clip_to_img(utils.cam_to_clip_space(lmk3d_cam, mvp), img_height, img_width)
                error = reproj_error(lmk2d_proj_cam, lmk2d_img)
                # file_reprog_err.write(f"{id_frame_out:>3d} {error:>7f}\n")
                # if error > cfg.INIT.REPROJ_ERROR_THRESH:
                if error > 20:
                    logger.warning(f"Reprojection error (final) = {error:.3f} for frame {id_frame_out}")
                
                # ==============================================================================================
                #  Segment hand using landmark
                # ==============================================================================================
                predictor.set_image(rgb)
                masks, scores, logits = predictor.predict(point_coords=lmk2d_img.astype(int), point_labels=np.ones((len(lmk2d_img))), box=bbox_xyxy, multimask_output=False)
                mask_sam = masks[0] # (H, W)
                
                # remove small blobs
                n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_sam.astype(np.uint8))
                # labels.shape = (H, W)
                # stats.shape = (n_labels, 5) Ref: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
                # centroids.shape = (n_labels, 2)
                if n_labels < 2:    # no component except background
                    logger.warning(f"[connected component] No foreground object in frame {id_frame_out}")
                    continue
                label_ids_sorted_by_area = np.argsort(stats[:, 4])  # 4 represents area of each component
                label_hand_comp = np.arange(n_labels)[label_ids_sorted_by_area][-2]
                mask = labels == label_hand_comp

                # mask using depth
                mask_depth = (depth > cfg.KINECT.DEPTH_NEAR) & (depth < cfg.KINECT.DEPTH_FAR)
                mask = mask & mask_depth

                # segment image using mask
                rgb_seg = rgb * mask[:, :, None] + np.zeros_like(rgb)*(~mask[:, :, None])   # (H, W, 3)
                # add mask in alpha channel
                rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                depth_seg = depth * mask + np.zeros_like(depth)*(~mask)
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fig = go.Figure(go.Heatmap(z=depth_seg))
                    fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                    fig.update_yaxes(autorange='reversed')
                    fig.show()
                    exit()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    xyz_data = utils.depth_to_xyz_np(depth_seg, depth_seg>0, fx, fy, cx, cy)
                    pts_pos = xyz_data; color_pos = "green"
                    scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                    fig = go.Figure([scat_data_pos])
                    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                    fig.show()
                    exit()
                    # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward

                # ==============================================================================================
                #  Crop
                # ==============================================================================================
                CROP = True
                if CROP:
                    threshold = np.amin(depth_seg[mask])    # used for thresholding

                    mask_orig = mask.copy()
                    trans_2x3, bbox_xywh = get_crop_affine_trans(bbox_xyxy, cfg.KINECT.SCALE_CROP, cfg.IMG_RES)
                    rgb_seg = cv.warpAffine(rgba_seg[:, :, :3], trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    mask = cv.warpAffine(mask.astype(np.uint8), trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)

                    rgb_seg = rgb_seg * mask[:, :, None] + np.zeros_like(rgb_seg)*(~mask[:, :, None])   # (H, W, 3)
                    rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                    # depth_seg = cv.bilateralFilter(depth_seg, -1, 1, 10)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        # exit()
                    depth_seg = cv.warpAffine(depth_seg, trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=mask_orig.astype(np.float32)))
                        fig.update_layout(width=mask_orig.shape[1], height=mask_orig.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    mask_lin = cv.warpAffine(mask_orig.astype(np.float32), trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=mask_lin))
                        fig.update_layout(width=mask_lin.shape[1], height=mask_lin.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    depth_seg[mask_lin>0] = depth_seg[mask_lin>0] / mask_lin[mask_lin>0].astype(np.float32)
                    # depth_seg = depth_seg * mask + np.zeros_like(depth_seg)*(~mask)
                    # depth_seg = depth_seg * mask + np.zeros_like(depth_seg)*(~mask)
                    depth_seg = cv.bilateralFilter(depth_seg, -1, 30, 10)
                    # update mask to handle outliers
                    # mask = depth_seg > threshold
                    depth_seg = depth_seg * mask + np.zeros_like(depth_seg)*(~mask)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        # exit()

                    # update intrinsics correspondingly
                    # Ref: https://stackoverflow.com/a/74753976
                    K[0, 0] = K[0, 0] * cfg.IMG_RES[1] / (bbox_xywh[2] * cfg.KINECT.SCALE_CROP)
                    K[1, 1] = K[1, 1] * cfg.IMG_RES[0] / (bbox_xywh[3] * cfg.KINECT.SCALE_CROP)
                    K[:2, 2] = (trans_2x3 @ np.array([K[0, 2], K[1, 2], 1.0]))[:2]
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        xyz_data = utils.depth_to_xyz_np(depth_seg, depth_seg>0, fx, fy, cx, cy)
                        pts_pos = xyz_data; color_pos = "green"
                        scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                        fig = go.Figure([scat_data_pos])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.show()
                        exit()
                        # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward

                    # update 2D landmark points
                    lmk2d_img_seg = (trans_2x3 @ np.concatenate([lmk2d_img, np.ones((lmk2d_img.shape[0], 1))], axis=1).T).T[:, :2]
                    # plot
                    rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_img_seg, radius=5)
                    # rgba_seg_lmk2d_plot = np.concatenate([rgb_seg_lmk2d_plot, mask[:, :, None]], axis=2).astype(np.uint8)   # (H_crop, W_crop, 4)
                    
                else:
                    lmk2d_img_seg = lmk2d_img
                    rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_img_seg, radius=10)

                
                # # use depth image to obtain depth at 2D landmark points
                # # TODO: use nearest neighbor interpolation to obtain depth (in mm) at 2D point, convert it to camera coordinates corresponding to cropped K
                # # Ref (interpolate on grid data): https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python/39596856#39596856
                # interpolate.griddata(())


                # plot projected landmark on segmented image
                # lmk2d_img_seg_proj_cv = utils.apply_Krt(lmk3d_cam_cv, K, r=np.zeros(3), t=np.zeros(3))[:, :2]
                img_height, img_width = rgb_seg.shape[:2]; proj_mat = utils.opengl_persp_proj_mat_from_K(K, cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1], img_height, img_width); mv = np.eye(4); mvp = proj_mat @ mv
                lmk2d_img_seg_proj = utils.clip_to_img(utils.cam_to_clip_space(lmk3d_cam, mvp), img_height, img_width)
                rgb_seg_lmk2d_proj_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_img_seg_proj, radius=5)
                # rgb_seg_lmk2d_proj_plot = np.concatenate([rgb_seg_lmk2d_proj_plot, mask[:, :, None]], axis=2).astype(np.uint8)   # (H_crop, W_crop, 4)

                error = reproj_error(lmk2d_img_seg_proj, lmk2d_img_seg)
                file_reprog_err.write(f"{id_frame_out:>3d} {error:>7f}\n")

                """
                # ==============================================================================================
                #  Depth to pointcloud
                # ==============================================================================================
                intrinsic = o3d.camera.PinholeCameraIntrinsic(depth_seg.shape[1], depth_seg.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                pcd_raw = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_seg.astype(np.uint16)), intrinsic)    # this function requires depth to be in uint16
                # intrinsic = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], K_orig[0, 0], K_orig[1, 1], K_orig[0, 2], K_orig[1, 2])
                # pcd_raw = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth.astype(np.uint16)), intrinsic)    # this function requires depth to be in uint16
                
                # pcd_inlier, mask_selected_points = pcd_raw.remove_radius_outliers(nb_points=20, search_radius=0.01)    # remove points that have less than `nb_points` in a sphere of `search_radius`
                pcd_inlier, list_idx_selected_point = pcd_raw.remove_radius_outlier(nb_points=500, radius=0.01)    # remove points that have less than `nb_points` in a sphere of `radius`
                pcd_inlier.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
                pcd_inlier.orient_normals_towards_camera_location()

                if len(np.asarray(pcd_inlier.points)) > cfg.KINECT.N_SAMPLES_ON_PC:
                    pcd_down = pcd_inlier.farthest_point_down_sample(cfg.KINECT.N_SAMPLES_ON_PC)
                else:
                    pcd_down = pcd_inlier

                # x = pcd_inlier.point.positions.numpy(); xn = pcd_inlier.point.normals.numpy()
                pc_pos = np.asarray(pcd_down.points); pc_nrm = np.asarray(pcd_down.normals)
                # xn = utils.safe_normalize(xn) # not required, it is already normalized

                # transform from kinectv2's coordinate system to opengl coordinate system
                # opengl/blender/nvdiffrast:    x: right,   y: up,      z: backward (Ref: https://nvlabs.github.io/nvdiffrast/#coordinate-systems)
                # kinect:                       x: left,    y: up,      z: forward  (Ref: https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn785530(v=ieb.10))
                # kinect (based on plot):       x: right,   y: down,    z: forward
                R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                pc_pos = utils.apply_proj_mat(pc_pos, T_kin_to_gl)
                pc_nrm = utils.apply_proj_mat(pc_nrm, T_kin_to_gl)

                DEBUG = False
                if DEBUG:
                    # fig = go.Figure(go.Heatmap(z=depth_seg))
                    # fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                    # fig.update_yaxes(autorange='reversed')
                    # fig.show()

                    # logger.debug(f"Number of points in pointcloud: raw = {len(pcd_raw.point.positions.numpy())} | down = {len(pcd_down.point.positions.numpy())} | inlier = {len(pcd_inlier.point.positions.numpy())}")
                    logger.debug(f"Number of points in pointcloud: raw = {len(np.asarray(pcd_raw.points))} | inlier = {len(np.asarray(pcd_inlier.points))} | down = {len(np.asarray(pcd_down.points))}")
                    
                    pos_raw = np.asarray(pcd_raw.points)
                    scat_pos_raw = go.Scatter3d(x=pos_raw[:, 0], y=pos_raw[:, 1], z=pos_raw[:, 2], mode="markers", marker=dict(size=1, color="black"), name="pos_raw")
                    
                    pos_inlier = np.asarray(pcd_inlier.points)
                    scat_pos_inlier = go.Scatter3d(x=pos_inlier[:, 0], y=pos_inlier[:, 1], z=pos_inlier[:, 2], mode="markers", marker=dict(size=1, color="blue"), name="pos_inlier")
                    
                    pos_down = pc_pos
                    scat_pos_down = go.Scatter3d(x=pos_down[:, 0], y=pos_down[:, 1], z=pos_down[:, 2], mode="markers", marker=dict(size=3, color="green"), name="pos_down")
                    
                    # plotting normal lines for large number of points is slow and cluttered, so plot few (around 1000)
                    skip_step = int(len(pos_down)/1000)
                    line_pos_down = pos_down[::skip_step]; line_nrm_down = pc_nrm[::skip_step]
                    logger.debug(f"Number of lines for plotting normal = {len(line_pos_down)}")
                    list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(line_pos_down, line_pos_down+0.005*line_nrm_down)]
                    
                    scat_lmk3d_cam = go.Scatter3d(x=lmk3d_cam[:, 0], y=lmk3d_cam[:, 1], z=lmk3d_cam[:, 2], mode="markers", marker=dict(size=5, color="blue"), name="lmk3d_cam")
                    
                    # fig = go.Figure([scat_pos_raw, scat_pos_inlier, scat_pos_down, *lines_nrm])
                    fig = go.Figure([scat_lmk3d_cam, scat_pos_down, *list_scat_nrm])
                    # fig.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)))
                    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                    fig.show()
                    exit()
                """


                # ==============================================================================================
                #  Save results
                # ==============================================================================================
                cv.imwrite(f"{out_rgb_raw_dir}/{id_frame_out:05d}.png", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_bbox_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_bbox_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgba_seg_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgba_seg, cv.COLOR_RGBA2BGRA))
                cv.imwrite(f"{out_rgb_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
                # np.save(f"{out_lmk3d_img_seg_dir}/{id_frame_out:05d}.npy", lmk3d_img_seg)
                np.save(f"{out_lmk2d_img_dir}/{id_frame_out:05d}.npy", lmk2d_img_seg)
                np.save(f"{out_lmk3d_rel_dir}/{id_frame_out:05d}.npy", lmk3d_rel)
                np.save(f"{out_lmk3d_cam_dir}/{id_frame_out:05d}.npy", lmk3d_cam)
                np.save(f"{out_K_dir}/{id_frame_out:05d}.npy", K)
                cv.imwrite(f"{out_rgb_seg_lmk2d_proj_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_seg_lmk2d_proj_plot, cv.COLOR_RGB2BGR))
                np.save(f"{out_depth_raw_npy_dir}/{id_frame_out:05d}.npy", depth)
                cv.imwrite(f"{out_depth_raw_cm_dir}/{id_frame_out:05d}.png", clip_and_normalize_depth_to_cm_w_alpha(depth, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR))
                np.save(f"{out_depth_seg_npy_dir}/{id_frame_out:05d}.npy", depth_seg)
                cv.imwrite(f"{out_depth_seg_cm_dir}/{id_frame_out:05d}.png", clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR))
                # np.save(f"{out_pc_pos_dir}/{id_frame_out:05d}.npy", pc_pos)
                # np.save(f"{out_pc_nrm_dir}/{id_frame_out:05d}.npy", pc_nrm)

                # plot
                # fig = go.Figure(go.Scatter3d(x=lmk3d_rel[:, 0], y=lmk3d_rel[:, 1], z=lmk3d_rel[:, 2], mode="markers", name="lmk3d_rel"))
                # fig.update_layout(scene=dict(
                #     aspectmode="cube",
                #     xaxis=dict(range=[-0.1, 0.1]),
                #     yaxis=dict(range=[-0.1, 0.1]),
                #     zaxis=dict(range=[-0.1, 0.1]),
                # ))
                # fig.write_html(f"{out_lmk3d_rel_plot_dir}/{id_frame_out:05d}.html")

                # fig = go.Figure(go.Scatter3d(x=lmk3d_cam[:, 0], y=lmk3d_cam[:, 1], z=lmk3d_cam[:, 2], mode="markers", name="lmk3d_cam"))
                # fig.update_layout(scene=dict(
                #     # aspectmode="cube",
                #     aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
                #     # xaxis=dict(range=[-0.1, 0.1]),
                #     # yaxis=dict(range=[-0.1, 0.1]),
                #     # zaxis=dict(range=[-0.1, 0.1]),
                # ))
                # fig.write_html(f"{out_lmk3d_cam_plot_dir}/{id_frame_out:05d}.html")

                # scat_pc_pos = go.Scatter3d(x=pc_pos[:, 0], y=pc_pos[:, 1], z=pc_pos[:, 2], mode="markers", marker=dict(size=3, color="green"), showlegend=False)
                # list_scat_nrm = [go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode="lines", line=dict(color="red", width=2), hoverinfo="none", showlegend=False) for start, end in zip(pc_pos, pc_pos+0.005*pc_nrm)]
                # scat_lmk3d_cam = go.Scatter3d(x=lmk3d_cam[:, 0], y=lmk3d_cam[:, 1], z=lmk3d_cam[:, 2], mode="markers", marker=dict(size=5, color="blue"), name="lmk3d_cam")
                # fig = go.Figure([scat_lmk3d_cam, scat_pc_pos, *list_scat_nrm])
                # fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                # fig.update_layout(scene=dict(
                #     xaxis=dict(showbackground=False, showticklabels=False, title="", visible=False),
                #     yaxis=dict(showbackground=False, showticklabels=False, title="", visible=False),
                #     zaxis=dict(showbackground=False, showticklabels=False, title="", visible=False),
                # ))
                # fig.write_image(f"{out_pc_plot_dir}/{id_frame_out:05d}.png", width=1000, height=1000)

                id_frame_out += 1
                
                # break

    # height_rgb_raw, width_rgb_raw = rgb.shape[:2]
    height_rgba_seg, width_rgba_seg = rgba_seg.shape[:2]
    subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_raw_dir}"])
    subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_lmk2d_plot_dir}"])
    subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_bbox_plot_dir}"])
    subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_rgba_seg}", f"{height_rgba_seg}", f"{out_rgba_seg_dir}"])
    subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_seg_lmk2d_plot_dir}"])
    subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_seg_lmk2d_proj_plot_dir}"])
    height_depth_raw, width_depth_raw = depth.shape[:2]
    subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_depth_raw}", f"{height_depth_raw}", f"{out_depth_raw_cm_dir}"])
    height_depth_seg, width_depth_seg = depth_seg.shape[:2]
    subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_depth_seg}", f"{height_depth_seg}", f"{out_depth_seg_cm_dir}"])
    # subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_pc_plot_dir}"])

def compute_initial_params(cfg, lmk_seg_dir, init_dir):
    mano_initializer = ManoInitializer(cfg, lmk_seg_dir)
    mano_initializer.optimize(f"{init_dir}/log")
    mano_initializer.save_results(f"{init_dir}/out")


def preprocess_kinect(cfg, preprocess_dir):
    # ==============================================================================================
    # Calculate landmark points and segment hand region
    # ============================================================================================== 
    lmk_seg_dir = f"{preprocess_dir}/preprocess"
    if (not Path(lmk_seg_dir).exists()) or cfg.KINECT.FORCE_PREPROCESS_LMK_SEG:
        logger.info("Compute landmark and segment hand (start)...")
        compute_landmark_and_segment_hand(cfg, lmk_seg_dir)
        logger.info("Compute landmark and segment hand (complete)")
    
    # exit()  # TODO: remove this line
    
    # ==============================================================================================
    # Initialize MANO params and camera matrices for each frame
    # ============================================================================================== 
    init_dir = f"{preprocess_dir}/initialization"
    if (not Path(init_dir).exists()) or cfg.KINECT.FORCE_PREPROCESS_INIT:
        logger.info("Initialization (start)...")
        compute_initial_params(cfg, lmk_seg_dir, init_dir)
        logger.info("Initialization (complete)")

    # ==============================================================================================
    # Save initialization
    # ============================================================================================== 
    

class KinectDataset(torch.utils.data.Dataset):
    def __init__(self, preprocess_dir, cfg):
        super().__init__()
        self.cfg = cfg

        list_path_to_rgba_seg = list(sorted(Path(f"{preprocess_dir}/rgba_seg").glob("*.png")))
        # list_id_data = range(0, len(list_path_to_rgba_seg))

        self.list_id_frame = []
        self.list_rgba = []
        self.list_K = []        # intrinsic mat
        self.list_lmk2d = []    # 2D landmark in image space
        self.list_lmk3d = []    # 3D landmark in camera space

        for id_frame in tqdm(range(len(list_path_to_rgba_seg)), desc="Appending to list"):
            self.list_id_frame.append(id_frame)
            
            rgba = cv.cvtColor(cv.imread(f"{preprocess_dir}/rgba_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
            rgba = torch.from_numpy(rgba).float()/255
            rgba[..., 0:3] = util.srgb_to_rgb(rgba[..., 0:3])
            self.list_rgba.append(rgba)

            lmk2d = np.load(f"{preprocess_dir}/lmk2d_img/{id_frame:05d}.npy")   # (21, 2) in image space
            self.list_lmk2d.append(torch.from_numpy(lmk2d).float())

            lmk3d = np.load(f"{preprocess_dir}/lmk3d_cam/{id_frame:05d}.npy")   # (21, 3) in camera space, with units in m
            self.list_lmk3d.append(torch.from_numpy(lmk3d).float())

            K = np.load(f"{preprocess_dir}/K/{id_frame:05d}.npy")  # (3, 3)
            self.list_K.append(torch.from_numpy(K).float())

        self.num_data = len(self.list_id_frame)

    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, id_frame):
        return {
            "id_frame": self.list_id_frame[id_frame],
            "rgba": self.list_rgba[id_frame].to(self.cfg.DEVICE),
            "lmk2d": self.list_lmk2d[id_frame].to(self.cfg.DEVICE),
            "lmk3d": self.list_lmk3d[id_frame].to(self.cfg.DEVICE),
            "K": self.list_K[id_frame].to(self.cfg.DEVICE)
        }
    

