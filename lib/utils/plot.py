import cv2 as cv
import numpy as np
from PIL import Image as PImage
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def draw_pts_on_img(img, list_pt2d, radius=5, color=(0, 255, 255), thickness=-1):
    img_plot = img.copy()
    for pt2d in list_pt2d:
        img_plot = cv.circle(img_plot, pt2d.astype(int), radius, color, thickness)
    return img_plot

def alpha_composite(bg_rgba, fg_rgba, alpha):
    composite_img = bg_rgba.copy()
    composite_img[:, :, :3] = bg_rgba[:, :, :3] * (1 - fg_rgba[:, :, 3:4]*(1-alpha)) + fg_rgba[:, :, :3] * fg_rgba[:, :, 3:4]*alpha
    composite_img[:, :, :3] = np.clip(composite_img[:, :, :3], 0, 1)
    composite_img[:, :, 3] = np.maximum(composite_img[:, :, 3], fg_rgba[:, :, 3])
    return composite_img


def rgbd_as_plotly_surface(rgb, depth, min_depth_threshold=100, **kwargs):
    # Ref: https://github.com/plotly/plotly.py/issues/1827
    # depth = depth.copy().astype(np.float32)
    # depth[depth < min_depth_threshold] = np.nan

    eight_bit_img = PImage.fromarray(rgb).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    trace = go.Surface(
        z=depth, 
        surfacecolor=np.array(eight_bit_img), 
        cmin=0, cmax=255,
        colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    )
    
    return trace

def clip_and_normalize_depth(depth, depth_near, depth_far):
    m_bg = (depth < depth_near) | (depth > depth_far)
    depth_clip = np.clip(depth, depth_near, depth_far)
    depth_norm = (depth_clip - depth_near) / (depth_far - depth_near)
    depth_cm = (depth_norm*255).astype(np.uint8)
    depth_cm[m_bg] = 255

    return depth_cm

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

# unused
def calculate_depth_diff_img(depth_data, depth_ren, diff_threshold):
    m_model_in_front_of_D = depth_data > (depth_ren+diff_threshold)
    m_model_behind_D = depth_ren > (depth_data+diff_threshold)
    depth_diff_cm = np.full((depth_data.shape[0], depth_data.shape[1], 3), 255, dtype=np.uint8)
    depth_diff_cm[m_model_in_front_of_D] = np.array([255, 0, 0])
    depth_diff_cm[m_model_behind_D] = np.array([0, 0, 255])
    return depth_diff_cm

def depth_diff_and_cm_with_alpha(depth_data, depth_ren, depth_diff_max_thresh, cmap=plt.get_cmap("bwr")):
    depth_diff = depth_data - depth_ren
    depth_diff = np.clip(depth_diff, -depth_diff_max_thresh, depth_diff_max_thresh)
    depth_diff_norm = (depth_diff + depth_diff_max_thresh) / (2*depth_diff_max_thresh)
    depth_diff_cm = (cmap(depth_diff_norm)[:, :, :3] * 255).astype(np.uint8)

    mask = (depth_data > 0) | (depth_ren > 0)
    depth_diff_cm_with_alpha = np.concatenate([depth_diff_cm, 255*mask[:, :, None]], axis=2).astype(np.uint8)

    return depth_diff_cm_with_alpha
    

# unused
def get_range_of_plotly_fig_traces(traces):
    pt_x_min, pt_x_max = [], []
    pt_y_min, pt_y_max = [], []
    pt_z_min, pt_z_max = [], []
    for trace in traces:
        pt_x_min.append(min(trace.x)); pt_x_max.append(max(trace.x))
        pt_y_min.append(min(trace.y)); pt_y_max.append(max(trace.y))
        pt_z_min.append(min(trace.z)); pt_z_max.append(max(trace.z))
    pt_x_min = min(pt_x_min); pt_x_max = max(pt_x_max)
    pt_y_min = min(pt_y_min); pt_y_max = max(pt_y_max)
    pt_z_min = min(pt_z_min); pt_z_max = max(pt_z_max)

    return pt_x_min, pt_x_max, pt_y_min, pt_y_max, pt_z_min, pt_z_max
