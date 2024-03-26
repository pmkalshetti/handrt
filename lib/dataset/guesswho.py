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

    K_rgb_orig = utils.get_guesswho_K()
    K_depth_orig = utils.get_guesswho_K()
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
    out_rgb_raw_dir = f"{out_dir}/rgb_raw"; utils.create_dir(out_rgb_raw_dir, True)
    out_rgb_lmk2d_plot_dir = f"{out_dir}/rgb_lmk2d_plot"; utils.create_dir(out_rgb_lmk2d_plot_dir, True)
    out_rgb_bbox_plot_dir = f"{out_dir}/rgb_bbox_plot"; utils.create_dir(out_rgb_bbox_plot_dir, True)
    out_rgba_seg_dir = f"{out_dir}/rgba_seg"; utils.create_dir(out_rgba_seg_dir, True)
    out_rgb_seg_lmk2d_plot_dir = f"{out_dir}/rgb_seg_lmk2d_plot"; utils.create_dir(out_rgb_seg_lmk2d_plot_dir, True)
    out_lmk2d_rgb_dir = f"{out_dir}/lmk2d_rgb"; utils.create_dir(out_lmk2d_rgb_dir, True)
    out_K_rgb_dir = f"{out_dir}/K_rgb"; utils.create_dir(out_K_rgb_dir, True)

    out_depth_raw_npy_dir = f"{out_dir}/depth_raw_npy"; utils.create_dir(out_depth_raw_npy_dir, True)
    out_depth_raw_cm_dir = f"{out_dir}/depth_raw_cm"; utils.create_dir(out_depth_raw_cm_dir, True)
    out_depth_seg_npy_dir = f"{out_dir}/depth_seg_npy"; utils.create_dir(out_depth_seg_npy_dir, True)
    out_depth_seg_cm_dir = f"{out_dir}/depth_seg_cm"; utils.create_dir(out_depth_seg_cm_dir, True)
    out_depth_seg_lmk2d_plot_dir = f"{out_dir}/depth_seg_lmk2d_plot"; utils.create_dir(out_depth_seg_lmk2d_plot_dir, True)
    out_lmk2d_depth_dir = f"{out_dir}/lmk2d_depth"; utils.create_dir(out_lmk2d_depth_dir, True)
    out_K_depth_dir = f"{out_dir}/K_depth"; utils.create_dir(out_K_depth_dir, True)

    out_hadjust_depth_npy_dir = f"{out_dir}/hadjust_depth_npy"; utils.create_dir(out_hadjust_depth_npy_dir, True)
    out_hadjust_depth_cm_dir = f"{out_dir}/hadjust_depth_cm"; utils.create_dir(out_hadjust_depth_cm_dir, True)
    out_honline_depth_npy_dir = f"{out_dir}/honline_depth_npy"; utils.create_dir(out_honline_depth_npy_dir, True)
    out_honline_depth_cm_dir = f"{out_dir}/honline_depth_cm"; utils.create_dir(out_honline_depth_cm_dir, True)
    
    # out_T_depth_to_rgb_dir = f"{out_dir}/T_depth_to_rgb"; utils.create_dir(out_T_depth_to_rgb_dir, True)

    logger.info(f"Output will be stored in {out_dir}")

    # ==============================================================================================
    # process frames
    # ==============================================================================================
    in_dir = f"{cfg.GUESSWHO.DATA_ROOT_DIR}/{cfg.GUESSWHO.USER}/sensor_data"
    in_hadjust_dir = f"{cfg.GUESSWHO.DATA_ROOT_DIR}/{cfg.GUESSWHO.USER}/hadjust_rendered"
    in_honline_dir = f"{cfg.GUESSWHO.DATA_ROOT_DIR}/{cfg.GUESSWHO.USER}/honline_rendered"
    list_path_to_frame = sorted(list(Path(f"{in_dir}").glob("color-*.png")))
    if cfg.GUESSWHO.END_FRAME_ID != -1:
        list_path_to_frame = list_path_to_frame[:cfg.GUESSWHO.END_FRAME_ID+1]
    list_path_to_frame = list_path_to_frame[cfg.GUESSWHO.START_FRAME_ID:]

    logger.info(f"Segmenting hand using MediaPipe landmarks")
    # with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands:
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands_rgb, mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.8) as hands_depth:
        id_frame_out = 0
        # for id_frame_in, path_to_frame in enumerate(tqdm(list_path_to_frame)):
        for path_to_frame in tqdm(list_path_to_frame):
            id_frame_in = int(path_to_frame.stem[6:])   # skip prefix "color-"
            # ==============================================================================================
            #  Read images and camera intrinsics
            # ==============================================================================================
            rgb = cv.imread(f"{in_dir}/color-{id_frame_in:07d}.png")  # (240, 320, 3)   # no need of BGR2RGB as it is already stored inverted
            # rgb_reg = cv.cvtColor(cv.imread(f"{in_dir}/color_reg/{id_frame_in:05d}.png"), cv.COLOR_BGR2RGB)  # (424, 512, 3)
            rgb_reg = rgb.copy()
            depth = cv.imread(f"{in_dir}/depth-{id_frame_in:07d}.png", cv.IMREAD_ANYDEPTH).astype(np.float32)  # (240, 320) uint16 in mm
            mask_gt = cv.imread(f"{in_dir}/mask-{id_frame_in:07d}.png", cv.IMREAD_GRAYSCALE).astype(bool)  # (240, 320) uint8 [0, 255]
            
            depth_hadjust = cv.imread(f"{in_hadjust_dir}/depth-{id_frame_in:07d}.png", cv.IMREAD_ANYDEPTH).astype(np.float32)  # (240, 320) uint16 in mm
            depth_hadjust[depth_hadjust > 10000] = 0
            depth_honline = cv.imread(f"{in_honline_dir}/depth-{id_frame_in:07d}.png", cv.IMREAD_ANYDEPTH).astype(np.float32)  # (240, 320) uint16 in mm
            depth_honline[depth_honline > 10000] = 0

            
            K_rgb = K_rgb_orig.copy()
            K_depth = K_depth_orig.copy()
            DEBUG_LOCAL = False
            if DEBUG_LOCAL:
                fig = go.Figure(go.Heatmap(z=depth))
                fig.update_layout(width=2*depth.shape[1], height=2*depth.shape[0])
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
        
            FLIP = False
            if FLIP:
                rgb = np.fliplr(rgb).copy()
                rgb_reg = np.fliplr(rgb_reg).copy()
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
            
            # ==============================================================================================
            #  Segment hand using landmark (rgb)
            # ==============================================================================================
            predictor.set_image(rgb)
            masks, scores, logits = predictor.predict(point_coords=lmk2d_rgb.astype(int), point_labels=np.ones((len(lmk2d_rgb))), box=bbox_xyxy, multimask_output=False)
            mask_sam = masks[0] # (H, W)
            
            # # remove small blobs
            # n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_sam.astype(np.uint8))
            # # labels.shape = (H, W)
            # # stats.shape = (n_labels, 5) Ref: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
            # # centroids.shape = (n_labels, 2)
            # if n_labels < 2:    # no component except background
            #     logger.warning(f"[connected component] No foreground object in frame {id_frame_out}")
            #     continue
            # label_ids_sorted_by_area = np.argsort(stats[:, 4])  # 4 represents area of each component
            # label_hand_comp = np.arange(n_labels)[label_ids_sorted_by_area][-2]
            # mask = labels == label_hand_comp
            mask = mask_sam & mask_gt

            # # mask using depth
            # mask_depth = (depth > cfg.KINECT.DEPTH_NEAR) & (depth < cfg.KINECT.DEPTH_FAR)
            # mask = mask & mask_depth

            # segment image using mask
            rgb_seg = rgb * mask[:, :, None] + np.zeros_like(rgb)*(~mask[:, :, None])   # (H, W, 3)
            # add mask in alpha channel
            rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

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

            bbox_xyxy_reg = bbox_from_landmark_img(lmk2d_rgb_reg, rgb_reg.shape, pad=30).astype(int)    # (4,)
            # plot bbox on img
            rgb_reg_bbox_plot = rgb_reg.copy()
            cv.rectangle(rgb_reg_bbox_plot, (bbox_xyxy_reg[0], bbox_xyxy_reg[1]), (bbox_xyxy_reg[2], bbox_xyxy_reg[3]), (0, 255, 0), 3)
            
            # ==============================================================================================
            #  Segment hand using landmark (rgb_reg)
            # ==============================================================================================
            predictor.set_image(rgb_reg)
            masks, scores, logits = predictor.predict(point_coords=lmk2d_rgb_reg.astype(int), point_labels=np.ones((len(lmk2d_rgb_reg))), box=bbox_xyxy_reg, multimask_output=False)
            mask_sam = masks[0] # (H, W)
            
            # # remove small blobs
            # n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_sam.astype(np.uint8))
            # # labels.shape = (H, W)
            # # stats.shape = (n_labels, 5) Ref: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
            # # centroids.shape = (n_labels, 2)
            # if n_labels < 2:    # no component except background
            #     logger.warning(f"[connected component] No foreground object in frame {id_frame_out}")
            #     continue
            # label_ids_sorted_by_area = np.argsort(stats[:, 4])  # 4 represents area of each component
            # label_hand_comp = np.arange(n_labels)[label_ids_sorted_by_area][-2]
            # mask_reg = labels == label_hand_comp
            mask_reg = mask_sam & mask_gt

            # mask using depth
            mask_depth = (depth > cfg.GUESSWHO.DEPTH_NEAR) & (depth < cfg.GUESSWHO.DEPTH_FAR)
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

            # ==============================================================================================
            #  Crop (rgb)
            # ==============================================================================================
            CROP = True
            if CROP:
                trans_2x3, bbox_xywh = get_crop_affine_trans(bbox_xyxy, cfg.GUESSWHO.SCALE_CROP, cfg.IMG_RES)
                rgb_seg = cv.warpAffine(rgba_seg[:, :, :3], trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                mask = cv.warpAffine(mask.astype(np.uint8), trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)

                rgb_seg = rgb_seg * mask[:, :, None] + np.zeros_like(rgb_seg)*(~mask[:, :, None])   # (H, W, 3)
                rgba_seg = np.concatenate([rgb_seg, 255*mask[:, :, None]], axis=2).astype(np.uint8)   # (H, W, 4)

                # update intrinsics correspondingly
                # Ref: https://stackoverflow.com/a/74753976
                K_rgb[0, 0] = K_rgb[0, 0] * cfg.IMG_RES[1] / (bbox_xywh[2] * cfg.GUESSWHO.SCALE_CROP)
                K_rgb[1, 1] = K_rgb[1, 1] * cfg.IMG_RES[0] / (bbox_xywh[3] * cfg.GUESSWHO.SCALE_CROP)
                K_rgb[:2, 2] = (trans_2x3 @ np.array([K_rgb[0, 2], K_rgb[1, 2], 1.0]))[:2]
                
                # update 2D landmark points
                lmk2d_rgb_seg = (trans_2x3 @ np.concatenate([lmk2d_rgb, np.ones((lmk2d_rgb.shape[0], 1))], axis=1).T).T[:, :2]
                # plot
                rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_rgb_seg, radius=5)
                
            else:
                lmk2d_rgb_seg = lmk2d_rgb
                rgb_seg_lmk2d_plot = utils.draw_pts_on_img(rgba_seg[:, :, :3], lmk2d_rgb_seg, radius=10)

            
            # ==============================================================================================
            #  Crop (depth)
            # ==============================================================================================
            if CROP:
                mask_reg_orig = mask_reg.copy()
                trans_2x3_reg, bbox_xywh_reg = get_crop_affine_trans(bbox_xyxy_reg, cfg.GUESSWHO.SCALE_CROP, cfg.IMG_RES)
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
                K_depth[0, 0] = K_depth[0, 0] * cfg.IMG_RES[1] / (bbox_xywh_reg[2] * cfg.GUESSWHO.SCALE_CROP)
                K_depth[1, 1] = K_depth[1, 1] * cfg.IMG_RES[0] / (bbox_xywh_reg[3] * cfg.GUESSWHO.SCALE_CROP)
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
                depth_seg_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR)
                depth_seg_lmk2d_plot = utils.draw_pts_on_img(depth_seg_cm[:, :, :3], lmk2d_rgb_reg_seg, radius=5)

                mask_hadjust = depth_hadjust > 0
                mask_hadjust_orig = mask_hadjust.copy()
                mask_hadjust = cv.warpAffine(mask_hadjust.astype(np.uint8), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)
                mask_hadjust_lin = cv.warpAffine(mask_hadjust_orig.astype(np.float32), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                depth_hadjust = cv.warpAffine(depth_hadjust, trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                depth_hadjust[mask_hadjust_lin>0] = depth_hadjust[mask_hadjust_lin>0] / mask_hadjust_lin[mask_hadjust_lin>0].astype(np.float32)
                depth_hadjust = cv.bilateralFilter(depth_hadjust, -1, 30, 10)
                depth_hadjust = depth_hadjust * mask_hadjust + np.zeros_like(depth_hadjust)*(~mask_hadjust)
                DEBUG_LOCAL = False
                if DEBUG_LOCAL:
                    fig = go.Figure(go.Heatmap(z=depth_hadjust))
                    fig.update_layout(width=depth_hadjust.shape[1], height=depth_hadjust.shape[0])
                    fig.update_yaxes(autorange='reversed')
                    fig.show()
                    exit()
                
                mask_honline = depth_honline > 0
                mask_honline_orig = mask_honline.copy()
                mask_honline = cv.warpAffine(mask_honline.astype(np.uint8), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_NEAREST).astype(bool)
                mask_honline_lin = cv.warpAffine(mask_honline_orig.astype(np.float32), trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                depth_honline = cv.warpAffine(depth_honline, trans_2x3_reg, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
                depth_honline[mask_honline_lin>0] = depth_honline[mask_honline_lin>0] / mask_honline_lin[mask_honline_lin>0].astype(np.float32)
                depth_honline = cv.bilateralFilter(depth_honline, -1, 30, 10)
                depth_honline = depth_honline * mask_honline + np.zeros_like(depth_honline)*(~mask_honline)
                
                
            else:
                lmk2d_rgb_reg_seg = lmk2d_rgb_reg
                depth_seg_cm = utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR)
                depth_seg_lmk2d_plot = utils.draw_pts_on_img(depth_seg_cm[:, :, :3], lmk2d_rgb_reg_seg, radius=5)
                
            
            # ==============================================================================================
            #  Save results
            # ==============================================================================================
            cv.imwrite(f"{out_rgb_raw_dir}/{id_frame_out:05d}.png", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
            cv.imwrite(f"{out_rgb_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_lmk2d_plot, cv.COLOR_RGB2BGR))
            cv.imwrite(f"{out_rgb_bbox_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_bbox_plot, cv.COLOR_RGB2BGR))
            cv.imwrite(f"{out_rgba_seg_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgba_seg, cv.COLOR_RGBA2BGRA))
            cv.imwrite(f"{out_rgb_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(rgb_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
            np.save(f"{out_lmk2d_rgb_dir}/{id_frame_out:05d}.npy", lmk2d_rgb_seg)
            np.save(f"{out_K_rgb_dir}/{id_frame_out:05d}.npy", K_rgb)

            np.save(f"{out_depth_raw_npy_dir}/{id_frame_out:05d}.npy", depth)
            cv.imwrite(f"{out_depth_raw_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR))
            np.save(f"{out_depth_seg_npy_dir}/{id_frame_out:05d}.npy", depth_seg)
            cv.imwrite(f"{out_depth_seg_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth_seg, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR))
            cv.imwrite(f"{out_depth_seg_lmk2d_plot_dir}/{id_frame_out:05d}.png",  cv.cvtColor(depth_seg_lmk2d_plot, cv.COLOR_RGB2BGR))
            np.save(f"{out_lmk2d_depth_dir}/{id_frame_out:05d}.npy", lmk2d_rgb_reg_seg)
            np.save(f"{out_K_depth_dir}/{id_frame_out:05d}.npy", K_depth)
            # np.save(f"{out_T_depth_to_rgb_dir}/{id_frame_out:05d}.npy", T_depth_to_rgb_orig)

            np.save(f"{out_hadjust_depth_npy_dir}/{id_frame_out:05d}.npy", depth_hadjust)
            cv.imwrite(f"{out_hadjust_depth_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth_hadjust, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR))
            np.save(f"{out_honline_depth_npy_dir}/{id_frame_out:05d}.npy", depth_honline)
            cv.imwrite(f"{out_honline_depth_cm_dir}/{id_frame_out:05d}.png", utils.clip_and_normalize_depth_to_cm_w_alpha(depth_honline, cfg.GUESSWHO.DEPTH_NEAR, cfg.GUESSWHO.DEPTH_FAR))

            
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
    height_depth_raw, width_depth_raw = depth.shape[:2]
    subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_depth_raw}", f"{height_depth_raw}", f"{out_depth_raw_cm_dir}"])
    height_depth_seg, width_depth_seg = depth_seg.shape[:2]
    subprocess.run(["lib/utils/create_video_from_transparent_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{width_depth_seg}", f"{height_depth_seg}", f"{out_depth_seg_cm_dir}"])
    # subprocess.run(["lib/utils/create_video_from_frames.sh", "-f", "25", "-s", f"{0}", "-w", "%05d.png", f"{out_pc_plot_dir}"])

def compute_initial_params(cfg, lmk_seg_dir, init_dir):
    mano_initializer = ManoInitializer(cfg, lmk_seg_dir)
    mano_initializer.optimize(f"{init_dir}/log")
    mano_initializer.save_results(f"{init_dir}/out")


def preprocess_guesswho(cfg, preprocess_dir):
    # ==============================================================================================
    # Calculate landmark points and segment hand region
    # ============================================================================================== 
    lmk_seg_dir = f"{preprocess_dir}/lmk_seg"
    if (not Path(lmk_seg_dir).exists()) or cfg.GUESSWHO.FORCE_PREPROCESS_LMK_SEG:
        logger.info("Compute landmark and segment hand (start)...")
        estimate_lmk_and_segment(cfg, f"{lmk_seg_dir}")
        logger.info("Compute landmark and segment hand (complete)")
    
    # exit()  # TODO: remove this line
    
    # ==============================================================================================
    # Initialize MANO params and camera matrices for each frame
    # ============================================================================================== 
    init_dir = f"{preprocess_dir}/initialization"
    if (not Path(init_dir).exists()) or cfg.GUESSWHO.FORCE_PREPROCESS_INIT:
        logger.info("Initialization (start)...")
        compute_initial_params(cfg, lmk_seg_dir, init_dir)
        logger.info("Initialization (complete)")

    # ==============================================================================================
    # Save initialization
    # ============================================================================================== 
    

class GuesswhoDataset(torch.utils.data.Dataset):
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
    

