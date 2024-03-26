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

# def fill_K(fx, fy, cx, cy):
#     K = np.array([
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1]
#     ], dtype=np.float32)

#     return K

# def get_kinecvtv2_stored_depth_camera_intrinsics():
#     # fx = 366.085
#     # fy = 366.085
#     # cx = 259.229
#     # cy = 207.968

    
#     # # return fx, fy, cx, cy
#     # K_depth = fill_K(fx, fy, cx, cy)

#     K_depth = np.array(
#         [[370.2707923820256, 0, 252.43761208619625],
#         [0, 370.58955085768406, 218.11201894640524],
#         [0, 0, 1]]
#     , dtype=np.float32)
#     return K_depth


# def get_kinecvtv2_stored_color_camera_intrinsics():
#     # fx = 1081.372
#     # fy = 1081.372
#     # cx = 959.500
#     # cy = 539.500

#     # # return fx, fy, cx, cy
#     # K_color = fill_K(fx, fy, cx, cy)

#     K_color = np.array(
#         [[951.9449796327856, 0, 949.5328134419224],
#         [0, 950.7157511344377, 533.7178363479932],
#         [0, 0, 1]]
#     , dtype=np.float32)
#     return K_color

# def get_depth_to_color_stereo_extrinsic_transform_4x4():
#     # T_depth_to_color = np.array([
#     #     [0.9997502924829441, 0.01244422239337516, 0.018560549811138097, -0.05076114808196847],
#     #     [-0.012330113370313842, 0.9999044498654309, -0.0062497554820337324, 0.007607216445136995],
#     #     [-0.01863654969522862, 0.006019341187723377, 0.9998082048808776, -0.015544198619326833],
#     #     [0, 0, 0, 1]
#     # ], dtype=np.float32)

#     T_depth_to_color = np.array(
#         [[0.9999379870791254, 0.011091306469704321, -0.001002455456346108, -1.584101191842083],
#         [-0.01105789108560517, 0.9995250902234889, 0.02876311976238475, 0.027073655328168095],
#         [0.0013209999567587777, -0.02875025103406037, 0.9995857532120958, -1.2063097219199503],
#         [0, 0, 0, 1]]
#     , dtype=np.float32)

#     return T_depth_to_color

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





def estimate_lmk_and_segment(cfg, lmk_seg_dir):
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

    if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
        K_rgb_orig = utils.get_calibrated_kinectv2_K_rgb()
    # if cfg.USE_DEPTH_TO_RGB_EXTRINSIC
    if cfg.MODE == "rgbd" or cfg.MODE == "depth":
        K_depth_orig = utils.get_calibrated_kinectv2_K_ir()
    # T_rgb_to_depth_orig = get_color_to_depth_stereo_extrinsic_transform_4x4()

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
    
    if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
        out_rgb_raw_dir = f"{out_dir}/rgb_raw"; utils.create_dir(out_rgb_raw_dir, True)
        out_rgb_lmk2d_plot_dir = f"{out_dir}/rgb_lmk2d_plot"; utils.create_dir(out_rgb_lmk2d_plot_dir, True)
        out_rgb_bbox_plot_dir = f"{out_dir}/rgb_bbox_plot"; utils.create_dir(out_rgb_bbox_plot_dir, True)
        out_rgba_seg_dir = f"{out_dir}/rgba_seg"; utils.create_dir(out_rgba_seg_dir, True)
        out_rgb_seg_lmk2d_plot_dir = f"{out_dir}/rgb_seg_lmk2d_plot"; utils.create_dir(out_rgb_seg_lmk2d_plot_dir, True)
        out_lmk2d_rgb_dir = f"{out_dir}/lmk2d_rgb"; utils.create_dir(out_lmk2d_rgb_dir, True)
        if cfg.MODE == "rgb":
            out_lmk3d_cam_dir = f"{out_dir}/lmk3d_cam"; utils.create_dir(out_lmk3d_cam_dir, True)
        out_K_rgb_dir = f"{out_dir}/K_rgb"; utils.create_dir(out_K_rgb_dir, True)

    if cfg.MODE == "rgbd" or cfg.MODE == "depth":
        out_rgb_reg_raw_dir = f"{out_dir}/rgb_reg_raw"; utils.create_dir(out_rgb_reg_raw_dir, True)
        out_rgb_reg_lmk2d_plot_dir = f"{out_dir}/rgb_reg_lmk2d_plot"; utils.create_dir(out_rgb_reg_lmk2d_plot_dir, True)
        out_rgb_reg_bbox_plot_dir = f"{out_dir}/rgb_reg_bbox_plot"; utils.create_dir(out_rgb_reg_bbox_plot_dir, True)
        out_rgba_reg_seg_dir = f"{out_dir}/rgba_reg_seg"; utils.create_dir(out_rgba_reg_seg_dir, True)
        out_rgb_reg_seg_lmk2d_plot_dir = f"{out_dir}/rgb_reg_seg_lmk2d_plot"; utils.create_dir(out_rgb_reg_seg_lmk2d_plot_dir, True)
        out_lmk2d_rgb_reg_dir = f"{out_dir}/lmk2d_rgb_reg"; utils.create_dir(out_lmk2d_rgb_reg_dir, True)

        out_depth_raw_npy_dir = f"{out_dir}/depth_raw_npy"; utils.create_dir(out_depth_raw_npy_dir, True)
        out_depth_raw_cm_dir = f"{out_dir}/depth_raw_cm"; utils.create_dir(out_depth_raw_cm_dir, True)
        out_depth_seg_npy_dir = f"{out_dir}/depth_seg_npy"; utils.create_dir(out_depth_seg_npy_dir, True)
        out_depth_seg_cm_dir = f"{out_dir}/depth_seg_cm"; utils.create_dir(out_depth_seg_cm_dir, True)
        out_depth_seg_lmk2d_plot_dir = f"{out_dir}/depth_seg_lmk2d_plot"; utils.create_dir(out_depth_seg_lmk2d_plot_dir, True)
        out_lmk2d_depth_dir = f"{out_dir}/lmk2d_depth"; utils.create_dir(out_lmk2d_depth_dir, True)
        out_K_depth_dir = f"{out_dir}/K_depth"; utils.create_dir(out_K_depth_dir, True)
    
    # out_T_depth_to_rgb_dir = f"{out_dir}/T_depth_to_rgb"; utils.create_dir(out_T_depth_to_rgb_dir, True)

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
    # with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands:
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands_rgb, mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands_depth:
        id_frame_out = 0
        # for id_frame_in, path_to_frame in enumerate(tqdm(list_path_to_frame)):
        for path_to_frame in tqdm(list_path_to_frame):
            id_frame_in = int(path_to_frame.stem)
            # ==============================================================================================
            #  Read images and camera intrinsics
            # ==============================================================================================
            if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                rgb = cv.cvtColor(cv.imread(f"{in_dir}/color/{id_frame_in:05d}.png"), cv.COLOR_BGR2RGB)  # (1080, 1920, 3)
                K_rgb = K_rgb_orig.copy()
            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                rgb_reg = cv.cvtColor(cv.imread(f"{in_dir}/color_reg/{id_frame_in:05d}.png"), cv.COLOR_BGR2RGB)  # (424, 512, 3)
                depth = np.load(f"{in_dir}/depth_npy/{id_frame_in:05d}.npy")  # (424, 512)
                K_depth = K_depth_orig.copy()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fig = go.Figure(go.Heatmap(z=depth))
                    fig.update_layout(width=depth.shape[1], height=depth.shape[0])
                    fig.update_yaxes(autorange='reversed')
                    fig.show()
                    exit()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                    xyz_data = utils.depth_to_xyz_np(depth, depth>0, fx, fy, cx, cy)
                    pts_pos = xyz_data; color_pos = "green"
                    scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                    fig = go.Figure([scat_data_pos])
                    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                    fig.show()
                    exit()
                    # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward
        
            FLIP = True
            if FLIP:
                if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                    rgb = np.fliplr(rgb).copy()
                if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                    rgb_reg = np.fliplr(rgb_reg).copy()
                    depth = np.fliplr(depth).copy()

                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth))
                        fig.update_layout(width=depth.shape[1], height=depth.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        exit()
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                        xyz_data = utils.depth_to_xyz_np(depth, depth>0, fx, fy, cx, cy)
                        pts_pos = xyz_data; color_pos = "green"
                        scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                        fig = go.Figure([scat_data_pos])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.show()
                        exit()
                        # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward
                        # no need to change K (no need to negate fx)

            
            if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                # ==============================================================================================
                #  Compute landmark (rgb)
                # ==============================================================================================
                # to improve performance, optionally mark image as not writeable to pass by reference
                rgb.flags.writeable = False
                results = hands_rgb.process(rgb)
                if not results.multi_hand_landmarks:
                    logger.warning(f"No landmark for {path_to_frame}")
                    continue
                rgb.flags.writeable = True

                # results convention: https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md#multi_hand_landmarks
                # multi_hand_landmarks are in image space (x,y in normalized image space, z represents depth wrt wrist (smaller value corresponds to close to camera, magnitude of z is roughly same scale as x))
                lmk3d_rgb = mp_normalized_landmark_to_np_image_landmark(results.multi_hand_landmarks[0].landmark, rgb.shape)    # this scales x,y to image space
                lmk3d_rgb = lmk3d_rgb[mp_to_mano_lmk_ordering]  # (21, 3)
                lmk2d_rgb = lmk3d_rgb[:, :2]    # (21, 2)
                
                rgb_lmk2d_plot = utils.draw_pts_on_img(rgb, lmk2d_rgb, radius=10)

                bbox_xyxy = bbox_from_landmark_img(lmk2d_rgb, rgb.shape, pad=30).astype(int)    # (4,)
                # plot bbox on img
                rgb_bbox_plot = rgb.copy()
                cv.rectangle(rgb_bbox_plot, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 3)
                
                if cfg.MODE == "rgb":
                    # 3D landmarks, root relative normalized
                    lmk3d_rel = mp_world_landmark_to_np_world_landmark(results.multi_hand_world_landmarks[0].landmark)
                    lmk3d_rel = lmk3d_rel[mp_to_mano_lmk_ordering]

                    # calculate transformation from object space to camera space
                    # Note about cv.solvePnP: Numpy array slices won't work as input because solvePnP requires contiguous arrays (Ref: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
                    # (although the below line works without this, why?)
                    success, r_pnp, t_pnp = cv.solvePnP(np.ascontiguousarray(lmk3d_rel).astype(np.float32), np.ascontiguousarray(lmk2d_rgb).astype(np.float32), K_rgb, None, flags=cv.SOLVEPNP_SQPNP)
                    assert success
                    r_pnp = r_pnp[:, 0]; t_pnp = t_pnp[:, 0]    # solvepnp outputs extra dim, i.e., (3, 1) -> (3,) [not required when providing initial guess]
                    
                    # solvepnp assumes opencv camera convention
                    # we use opengl camera convention to project/render/rasterize in nvdiffrast
                    # opencv cam                    x: right,   y: down,    z: forward
                    # opengl/blender/nvdiffrast:    x: right,   y: up,      z: backward (Ref: https://nvlabs.github.io/nvdiffrast/#coordinate-systems)
                    r_obj_to_cv = r_pnp; t_obj_to_cv = t_pnp
                    R_obj_to_cv = Rotation.from_rotvec(r_obj_to_cv).as_matrix()
                    T_obj_to_cv = utils.create_4x4_trans_mat_from_R_t(R_obj_to_cv, t_obj_to_cv)

                    # apply transformation
                    lmk3d_cam_cv = lmk3d_rel @ Rotation.from_rotvec(r_pnp).as_matrix().T + t_pnp
                    lmk3d_cam = utils.apply_proj_mat(lmk3d_rel, T_obj_to_cv)

                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        # check final reprojection error

                        # project transformed points (given in camera space)
                        lmk2d_proj_cam_cv = utils.apply_Krt(lmk3d_cam_cv, K_rgb, r=np.zeros(3), t=np.zeros(3))[:, :2]
                        R_kin_to_gl = Rotation.from_euler("X", np.pi).as_matrix()
                        T_kin_to_gl = utils.create_4x4_trans_mat_from_R_t(R_kin_to_gl)
                        img_height, img_width = rgb.shape[:2]; proj_mat = utils.opengl_persp_proj_mat_from_K(K_rgb, cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1], img_height, img_width); mv = T_kin_to_gl; mvp = proj_mat @ mv
                        lmk2d_proj_cam = utils.clip_to_img(utils.cam_to_clip_space(lmk3d_cam, mvp), img_height, img_width)
                        assert np.allclose(lmk2d_proj_cam_cv, lmk2d_proj_cam)
                        # calculate initial reprojection error
                        error = reproj_error(lmk2d_proj_cam, lmk2d_rgb)
                        logger.debug(f"Reprojection error (final) = {error:.3f}")
                        # plot 3D points in camera space
                        pts = lmk3d_cam; name = "lmk3d_cam"
                        fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name))
                        fig.update_layout(scene=dict(aspectmode="data"))
                        fig.show()
                        # plot projected points (final/camera space) on image
                        rgb_plot_lmk2d_proj_cam = utils.draw_pts_on_img(rgb, lmk2d_proj_cam, radius=1)
                        fig = go.Figure(go.Image(z=rgb_plot_lmk2d_proj_cam))
                        fig.show()
                        # cv.imwrite(f"{log_dir}/rgb_plot_lmk2d_proj_cam.png", cv.cvtColor(rgb_plot_lmk2d_proj_cam, cv.COLOR_RGB2BGR))

                        exit()

                # ==============================================================================================
                #  Segment hand using landmark (rgb)
                # ==============================================================================================
                predictor.set_image(rgb)
                masks, scores, logits = predictor.predict(point_coords=lmk2d_rgb.astype(int), point_labels=np.ones((len(lmk2d_rgb))), box=bbox_xyxy, multimask_output=False)
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

                # # mask using depth
                # mask_depth = (depth > cfg.KINECT.DEPTH_NEAR) & (depth < cfg.KINECT.DEPTH_FAR)
                # mask = mask & mask_depth

                # segment image using mask
                rgb_seg = rgb * mask[:, :, None] + np.zeros_like(rgb)*(~mask[:, :, None])   # (H, W, 3)
                # add mask in alpha channel
                rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
            
                # ==============================================================================================
                #  Compute landmark (rgb_reg)
                # ==============================================================================================
                # to improve performance, optionally mark image as not writeable to pass by reference
                rgb_reg.flags.writeable = False
                results = hands_depth.process(rgb_reg)
                if not results.multi_hand_landmarks:
                    logger.warning(f"(rgb_reg) No landmark for {path_to_frame}")
                    continue
                rgb_reg.flags.writeable = True

                # results convention: https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md#multi_hand_landmarks
                # multi_hand_landmarks are in image space (x,y in normalized image space, z represents depth wrt wrist (smaller value corresponds to close to camera, magnitude of z is roughly same scale as x))
                lmk3d_rgb_reg = mp_normalized_landmark_to_np_image_landmark(results.multi_hand_landmarks[0].landmark, rgb_reg.shape)    # this scales x,y to image space
                lmk3d_rgb_reg = lmk3d_rgb_reg[mp_to_mano_lmk_ordering]  # (21, 3)
                lmk2d_rgb_reg = lmk3d_rgb_reg[:, :2]    # (21, 2)
                
                rgb_reg_lmk2d_plot = utils.draw_pts_on_img(rgb_reg, lmk2d_rgb_reg, radius=5)

                bbox_xyxy_reg = bbox_from_landmark_img(lmk2d_rgb_reg, rgb_reg.shape, pad=10).astype(int)    # (4,)
                # plot bbox on img
                rgb_reg_bbox_plot = rgb_reg.copy()
                cv.rectangle(rgb_reg_bbox_plot, (bbox_xyxy_reg[0], bbox_xyxy_reg[1]), (bbox_xyxy_reg[2], bbox_xyxy_reg[3]), (0, 255, 0), 3)
                
                # ==============================================================================================
                #  Segment hand using landmark (rgb_reg)
                # ==============================================================================================
                predictor.set_image(rgb_reg)
                masks, scores, logits = predictor.predict(point_coords=lmk2d_rgb_reg.astype(int), point_labels=np.ones((len(lmk2d_rgb_reg))), box=bbox_xyxy_reg, multimask_output=False)
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
                mask_reg = labels == label_hand_comp

                # mask using depth
                mask_depth = (depth > cfg.KINECT.DEPTH_NEAR) & (depth < cfg.KINECT.DEPTH_FAR)
                mask_reg = mask_reg & mask_depth

                # segment image using mask
                rgb_reg_seg = rgb_reg * mask_reg[:, :, None] + np.zeros_like(rgb_reg)*(~mask_reg[:, :, None])   # (H, W, 3)
                # add mask in alpha channel
                rgba_reg_seg = np.concatenate([rgb_reg_seg, 255*mask_reg[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                depth_seg = depth * mask_reg + np.zeros_like(depth)*(~mask_reg)
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fig = go.Figure(go.Heatmap(z=depth_seg))
                    fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                    fig.update_yaxes(autorange='reversed')
                    fig.show()
                    exit()
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                    xyz_data = utils.depth_to_xyz_np(depth_seg, depth_seg>0, fx, fy, cx, cy)
                    pts_pos = xyz_data; color_pos = "green"
                    scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                    fig = go.Figure([scat_data_pos])
                    fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                    fig.show()
                    exit()
                    # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward

            CROP = True
            # ==============================================================================================
            #  Crop (rgb)
            # ==============================================================================================
            if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                if CROP:
                    trans_2x3, bbox_xywh = get_crop_affine_trans(bbox_xyxy, cfg.KINECT.SCALE_CROP, cfg.IMG_RES)
                    rgb_seg = cv.warpAffine(rgba_seg[:, :, :3], trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    mask = cv.warpAffine(mask.astype(np.uint8), trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)

                    rgb_seg = rgb_seg * mask[:, :, None] + np.zeros_like(rgb_seg)*(~mask[:, :, None])   # (H, W, 3)
                    rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                    # update intrinsics correspondingly
                    # Ref: https://stackoverflow.com/a/74753976
                    K_rgb[0, 0] = K_rgb[0, 0] * cfg.IMG_RES[1] / (bbox_xywh[2] * cfg.KINECT.SCALE_CROP)
                    K_rgb[1, 1] = K_rgb[1, 1] * cfg.IMG_RES[0] / (bbox_xywh[3] * cfg.KINECT.SCALE_CROP)
                    K_rgb[:2, 2] = (trans_2x3 @ np.array([K_rgb[0, 2], K_rgb[1, 2], 1.0]))[:2]
                    
                    # update 2D landmark points
                    lmk2d_rgb_seg = (trans_2x3 @ np.concatenate([lmk2d_rgb, np.ones((lmk2d_rgb.shape[0], 1))], axis=1).T).T[:, :2]
                    # plot
                    rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_rgb_seg, radius=5)
                    
                else:
                    lmk2d_rgb_seg = lmk2d_rgb
                    rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_rgb_seg, radius=10)

                
            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                # ==============================================================================================
                #  Crop (depth)
                # ==============================================================================================
                if CROP:
                    mask_reg_orig = mask_reg.copy()
                    trans_2x3_reg, bbox_xywh_reg = get_crop_affine_trans(bbox_xyxy_reg, cfg.KINECT.SCALE_CROP, cfg.IMG_RES)
                    rgb_reg_seg = cv.warpAffine(rgba_reg_seg[:, :, :3], trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    mask_reg = cv.warpAffine(mask_reg.astype(np.uint8), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)

                    rgb_reg_seg = rgb_reg_seg * mask_reg[:, :, None] + np.zeros_like(rgb_reg_seg)*(~mask_reg[:, :, None])   # (H, W, 3)
                    rgba_reg_seg = np.concatenate([rgb_reg_seg, 255*mask_reg[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                    # depth_seg = cv.bilateralFilter(depth_seg, -1, 1, 10)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        # exit()
                    depth_seg = cv.warpAffine(depth_seg, trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=mask_reg_orig.astype(np.float32)))
                        fig.update_layout(width=mask_reg_orig.shape[1], height=mask_reg_orig.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    mask_reg_lin = cv.warpAffine(mask_reg_orig.astype(np.float32), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=mask_reg_lin))
                        fig.update_layout(width=mask_reg_lin.shape[1], height=mask_reg_lin.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                    depth_seg[mask_reg_lin>0] = depth_seg[mask_reg_lin>0] / mask_reg_lin[mask_reg_lin>0].astype(np.float32)
                    # depth_seg = depth_seg * mask + np.zeros_like(depth_seg)*(~mask)
                    # depth_seg = depth_seg * mask + np.zeros_like(depth_seg)*(~mask)
                    depth_seg = cv.bilateralFilter(depth_seg, -1, 30, 10)
                    # update mask to handle outliers
                    # mask = depth_seg > threshold
                    depth_seg = depth_seg * mask_reg + np.zeros_like(depth_seg)*(~mask_reg)
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fig = go.Figure(go.Heatmap(z=depth_seg))
                        fig.update_layout(width=depth_seg.shape[1], height=depth_seg.shape[0])
                        fig.update_yaxes(autorange='reversed')
                        fig.show()
                        # exit()

                    # update intrinsics correspondingly
                    # Ref: https://stackoverflow.com/a/74753976
                    K_depth[0, 0] = K_depth[0, 0] * cfg.IMG_RES[1] / (bbox_xywh_reg[2] * cfg.KINECT.SCALE_CROP)
                    K_depth[1, 1] = K_depth[1, 1] * cfg.IMG_RES[0] / (bbox_xywh_reg[3] * cfg.KINECT.SCALE_CROP)
                    K_depth[:2, 2] = (trans_2x3_reg @ np.array([K_depth[0, 2], K_depth[1, 2], 1.0]))[:2]
                    DEBUG_LOCAL = False
                    if DEBUG_LOCAL:
                        fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
                        xyz_data = utils.depth_to_xyz_np(depth_seg, depth_seg>0, fx, fy, cx, cy)
                        pts_pos = xyz_data; color_pos = "green"
                        scat_data_pos = go.Scatter3d(x=pts_pos[:, 0], y=pts_pos[:, 1], z=pts_pos[:, 2], mode="markers", marker=dict(size=1, color=color_pos), showlegend=False)
                        fig = go.Figure([scat_data_pos])
                        fig.update_layout(scene=dict(aspectmode="data", aspectratio=dict(x=1, y=1, z=1)))
                        fig.show()
                        exit()
                        # kinect 3D plot shows the camera coordinate system convention is x: right, y: down, z: forward

                    # update 2D landmark points
                    lmk2d_rgb_reg_seg = (trans_2x3_reg @ np.concatenate([lmk2d_rgb_reg, np.ones((lmk2d_rgb_reg.shape[0], 1))], axis=1).T).T[:, :2]
                    # plot
                    rgb_reg_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_reg_seg[:, :, :3], lmk2d_rgb_reg_seg, radius=5)
                    depth_seg_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR)
                    depth_seg_lmk2d_plot = utils.draw_pts_on_img(depth_seg_cm[:, :, :3], lmk2d_rgb_reg_seg, radius=5)
                    
                else:
                    lmk2d_rgb_reg_seg = lmk2d_rgb_reg
                    depth_seg_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR)
                    depth_seg_lmk2d_plot = utils.draw_pts_on_img(depth_seg_cm[:, :, :3], lmk2d_rgb_reg_seg, radius=5)
                    
            
            # ==============================================================================================
            #  Save results
            # ==============================================================================================
            if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
                cv.imwrite(f"{out_rgb_raw_dir}/{id_frame_out:05d}.png", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_bbox_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_bbox_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgba_seg_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgba_seg, cv.COLOR_RGBA2BGRA))
                cv.imwrite(f"{out_rgb_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
                np.save(f"{out_lmk2d_rgb_dir}/{id_frame_out:05d}.npy", lmk2d_rgb_seg)
                if cfg.MODE == "rgb":
                    np.save(f"{out_lmk3d_cam_dir}/{id_frame_out:05d}.npy", lmk3d_cam)
                np.save(f"{out_K_rgb_dir}/{id_frame_out:05d}.npy", K_rgb)

            if cfg.MODE == "rgbd" or cfg.MODE == "depth":
                cv.imwrite(f"{out_rgb_reg_raw_dir}/{id_frame_out:05d}.png", cv.cvtColor(rgb_reg, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_reg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_reg_lmk2d_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgb_reg_bbox_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_reg_bbox_plot, cv.COLOR_RGB2BGR))
                cv.imwrite(f"{out_rgba_reg_seg_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgba_reg_seg, cv.COLOR_RGBA2BGRA))
                cv.imwrite(f"{out_rgb_reg_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_reg_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
                np.save(f"{out_lmk2d_rgb_reg_dir}/{id_frame_out:05d}.npy", lmk2d_rgb_reg_seg)

                np.save(f"{out_depth_raw_npy_dir}/{id_frame_out:05d}.npy", depth)
                cv.imwrite(f"{out_depth_raw_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR))
                np.save(f"{out_depth_seg_npy_dir}/{id_frame_out:05d}.npy", depth_seg)
                cv.imwrite(f"{out_depth_seg_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.KINECT.DEPTH_NEAR, cfg.KINECT.DEPTH_FAR))
                cv.imwrite(f"{out_depth_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(depth_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
                np.save(f"{out_lmk2d_depth_dir}/{id_frame_out:05d}.npy", lmk2d_rgb_reg_seg)
                np.save(f"{out_K_depth_dir}/{id_frame_out:05d}.npy", K_depth)
                # np.save(f"{out_T_depth_to_rgb_dir}/{id_frame_out:05d}.npy", T_depth_to_rgb_orig)
            
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
    if cfg.MODE == "rgb" or (cfg.MODE == "rgbd" and (not cfg.USE_REG_RGB)):
        height_rgba_seg, width_rgba_seg = rgba_seg.shape[:2]
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_raw_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_lmk2d_plot_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_bbox_plot_dir}"])
        subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_rgba_seg}", f"{height_rgba_seg}", f"{out_rgba_seg_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_seg_lmk2d_plot_dir}"])
    
    if cfg.MODE == "rgbd" or cfg.MODE == "depth":
        height_rgba_reg_seg, width_rgba_reg_seg = rgba_reg_seg.shape[:2]
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_reg_raw_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_reg_lmk2d_plot_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_reg_bbox_plot_dir}"])
        subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_rgba_reg_seg}", f"{height_rgba_reg_seg}", f"{out_rgba_reg_seg_dir}"])
        subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_rgb_reg_seg_lmk2d_plot_dir}"])
        
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
    lmk_seg_dir = f"{preprocess_dir}/lmk_seg"
    if (not Path(lmk_seg_dir).exists()) or cfg.KINECT.FORCE_PREPROCESS_LMK_SEG:
        logger.info("Compute landmark and segment hand (start)...")
        estimate_lmk_and_segment(cfg, f"{lmk_seg_dir}")
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
