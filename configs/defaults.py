# this file is the one-stop reference point for all configurable options. It should be very well documented and provide sensible defaults for all options.
# Ref: https://github.com/rbgirshick/yacs#usage

from yacs.config import CfgNode as CN

_C = CN()

_C.EXPT_NAME = "reg_pratik_01"
_C.DEVICE = "cuda:0"
_C.SEGMENT_ANYTHING_MODEL_PATH = "data/segment_anything_model/sam_vit_h_4b8939.pth"
_C.IMG_RES = [1024, 1024]
# frustrum
_C.CAM_NEAR_FAR = [0.001, 1000.0] # near far plane in m
_C.OUT_DIR = "output"
_C.MODE = "rgbd"
_C.USE_REG_RGB = False

_C.PLOT = CN()
_C.PLOT.DEPTH_DIFF_MAX_THRESH = 50 # mm
_C.PLOT.DEPTH_NEAR = 200
_C.PLOT.DEPTH_FAR = 1000

_C.KINECT = CN()
_C.KINECT.DATA_ROOT_DIR = "data/kinect_rgbd"
_C.KINECT.PREPROCESS_ROOT_DIR = "output/kinect/preprocess"
_C.KINECT.USER = "pratik"
_C.KINECT.SEQUENCE = "01"
_C.KINECT.START_FRAME_ID = 0
_C.KINECT.END_FRAME_ID = -1
_C.KINECT.SCALE_CROP = 1.3
_C.KINECT.FORCE_PREPROCESS_LMK_SEG = False
_C.KINECT.FORCE_PREPROCESS_INIT = False
_C.KINECT.DEPTH_NEAR = 500 # mm
_C.KINECT.DEPTH_FAR = 1000 # mm
_C.KINECT.N_SAMPLES_ON_PC = 1000

_C.GUESSWHO = CN()
_C.GUESSWHO.DATA_ROOT_DIR = "data/guess-who"
_C.GUESSWHO.PREPROCESS_ROOT_DIR = "output/guesswho/preprocess"
_C.GUESSWHO.USER = "1"
_C.GUESSWHO.START_FRAME_ID = 0
_C.GUESSWHO.END_FRAME_ID = -1
_C.GUESSWHO.SCALE_CROP = 1.0
_C.GUESSWHO.FORCE_PREPROCESS_LMK_SEG = False
_C.GUESSWHO.FORCE_PREPROCESS_INIT = False
_C.GUESSWHO.DEPTH_NEAR = 200 # mm
_C.GUESSWHO.DEPTH_FAR = 1000 # mm
_C.GUESSWHO.N_SAMPLES_ON_PC = 1000

# MANO
_C.MANO = CN()
_C.MANO.PKL_PATH = "data/mano/models/MANO_RIGHT.pkl"
_C.MANO.HTML_UV = "data/html/TextureBasis/uvs_right.pkl"
_C.MANO.HTML_KD = "data/html/TextureSet/shadingRemoval/001_R.png"
_C.MANO.OBJ_DIR = "output/mano_obj"
_C.MANO.SUBDIVIDE = 2

# render
_C.RENDER = CN()
_C.RENDER.PROBE_RES = 256  # env map probe resolution
_C.RENDER.N_SAMPLES = 12
_C.RENDER.DECORRELATED = False # use decorrelated sampling in forward and backward passes
_C.RENDER.DENOISER_DEMODULATE = True
_C.RENDER.SPP = 1
_C.RENDER.LAYERS = 1

_C.INIT = CN()
# _C.INIT.FORCE_RECOMPUTE = True
_C.INIT.N_STEPS = 100
_C.INIT.REPROJ_ERROR_THRESH = 20
# _C.INIT.STEPS_WITH_PREV_POSE = 10
# _C.INIT.STEPS_WITH_GLOBAL_POSE = 100
_C.INIT.OPTIMIZE_SHAPE = True
_C.INIT.OPTIMIZE_POSE = True
_C.INIT.LR_BETA = 0.1
_C.INIT.LR_OFFSET = 0.00001
_C.INIT.LR_GLOBAL_ROT = 0.1
_C.INIT.LR_GLOBAL_TRANSL = 0.0001
_C.INIT.LR_HAND_POSE = 0.6
_C.INIT.LR_GEOM = 0.02 # unused
_C.INIT.N_SAMPLES_ON_MESH = 1000
_C.INIT.W_CHAMFER_POS = 1.0
_C.INIT.W_CHAMFER_NRM = 0.1
_C.INIT.W_LMK3D = 1.0
_C.INIT.W_LMK2D = 1.0
_C.INIT.W_SIL = 0.1
_C.INIT.W_BETA_REG = 0.1
_C.INIT.W_LAPLACE_REG = 100000.0
# _C.INIT.W_TEMP = 1.0  # TODO: this increases instability, why?, recheck loss
_C.INIT.W_TEMP_HAND_POSE = 0.01
_C.INIT.W_TEMP_GLOBAL_ROT = 0.01
_C.INIT.W_TEMP_GLOBAL_TRANSL = 0.01
_C.INIT.LMK3D_NORM_SCALE = 1.0  # 300 mm bbox
_C.INIT.SEG_USING_MESH = True
_C.INIT.N_SAMPLES_ON_MESH = 5000
_C.INIT.PC_VOXEL_SIZE = 0.001
_C.INIT.CHAMF_FILTER_NRM_DIST = True
_C.INIT.CHAMF_FILTER_POS_DIST = True
_C.INIT.CHAMF_POS_THRESH = 0.02 # 20 mm
_C.INIT.CHAMF_NRM_THRESH = 90 # degree
_C.INIT.CHAMF_W_REN = 1.0
_C.INIT.CHAMF_W_DATA = 1.0
_C.INIT.N_SAMPLES_ON_PC = 1000


_C.OPT = CN() 
_C.OPT.ROOT_DIR = "output/optimization"
_C.OPT.LOG_INTERVAL = 10
_C.OPT.EPOCHS = 100
_C.OPT.OPTIMIZE_MATERIAL = True
_C.OPT.OPTIMIZE_LIGHT = True
_C.OPT.OPTIMIZE_SHAPE = True
_C.OPT.OPTIMIZE_POSE = True
_C.OPT.USE_INIT = True
_C.OPT.USE_CALIBRATED = False
_C.OPT.CALIB_OUT_DIR = "output/optimization/pratik_calib_01_mesh_seg/out"

_C.OPT.LEARNING_RATE_LGT = 0.005
_C.OPT.LEARNING_RATE_MAT = 0.001
_C.OPT.LR_BETA = 0.1
_C.OPT.LR_OFFSET = 0.005
_C.OPT.LR_GLOBAL_ROT = 0.01
_C.OPT.LR_GLOBAL_TRANSL = 0.0001
_C.OPT.LR_HAND_POSE = 0.01
_C.OPT.LR_GEOM = 0.02 # unused

_C.OPT.LAMBDA_KD = 0.1
_C.OPT.LAMBDA_KS = 0.05
_C.OPT.LAMBDA_NRM = 0.025
_C.OPT.LAMBDA_DIFFUSE = 0.15
_C.OPT.LAMBDA_SPECULAR = 0.0025
_C.OPT.LAMBDA_CHROMA = 0.025
_C.OPT.LAMBDA_NRM2 = 0.25
_C.OPT.LAPLACE = "absolute"
_C.OPT.LAPLACE_SCALE = 100000.0
_C.OPT.LAMBDA_LPIPS = 0.0
_C.OPT.W_LGT_REG = 0.05
_C.OPT.W_CHAMFER_POS = 1.0
_C.OPT.W_CHAMFER_NRM = 0.1
_C.OPT.CHAMF_FILTER_NRM_DIST = True
_C.OPT.CHAMF_FILTER_POS_DIST = True
_C.OPT.CHAMF_POS_THRESH = 0.02 # 20 mm
_C.OPT.CHAMF_NRM_THRESH = 90 # degree
_C.OPT.CHAMF_W_REN = 1.0
_C.OPT.CHAMF_W_DATA = 1.0
_C.OPT.N_SAMPLES_ON_PC = 1000

_C.MAT = CN()
_C.MAT.KD_MIN = [0.03, 0.03, 0.03]
_C.MAT.KD_MAX = [0.8, 0.8, 0.8]
_C.MAT.KS_MIN = [0.0, 0.3, 0.0]
_C.MAT.KS_MAX = [0.0, 0.7, 0.1]
_C.MAT.NRM_MIN = [-1.0, -1.0, 0.0]
_C.MAT.NRM_MAX = [1.0, 1.0, 1.0]
_C.MAT.TEXTURE_RES = [2048, 2048]
_C.MAT.NO_PERTURBED_NRM = False
_C.MAT.BSDF = 'pbr'

_C.RESULTS = CN()
_C.RESULTS.ROOT_DIR = "output/results"
_C.RESULTS.ACR_OUT_DIR = "../experiments/comparison_with_acr/output"
_C.RESULTS.MESHGRAPHORMER_OUT_DIR = "../experiments/comparison_with_meshgraphormer/output"
_C.RESULTS.EXPT_NAME = "pratik_track_01"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`
