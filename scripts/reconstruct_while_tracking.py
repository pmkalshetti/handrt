import numpy as np
from pathlib import Path
import logging
import cv2 as cv
import subprocess
import torch
import pickle
from tqdm import tqdm
import argparse

import lib.utils as utils
from configs.defaults import get_cfg_defaults
from lib.dataset.kinect import preprocess_kinect
from lib.dataset.guesswho import preprocess_guesswho, GuesswhoDataset
from lib.optimizer.optimizer import Optimizer

# torch.manual_seed(0)
# np.random.seed(0)
# Ref: https://github.com/NVlabs/nvdiffrast/issues/13#issue-794011496
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

logger = utils.get_logger("lib", level=logging.DEBUG, root=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file", default="configs/expts/guesswho.yaml")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    logger.info(f"Experiment: {cfg.EXPT_NAME}")
    
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if "guesswho" in cfg.EXPT_NAME:
        cfg.PLOT.DEPTH_NEAR = cfg.GUESSWHO.DEPTH_NEAR
        cfg.PLOT.DEPTH_FAR = cfg.GUESSWHO.DEPTH_FAR
        preprocess_dir = f"{cfg.GUESSWHO.PREPROCESS_ROOT_DIR}/{cfg.GUESSWHO.USER}"
        if (not Path(f"{preprocess_dir}/initialization").exists()) or (cfg.GUESSWHO.FORCE_PREPROCESS_LMK_SEG or cfg.GUESSWHO.FORCE_PREPROCESS_INIT):
            logger.info("Preprocess guesswho data (start)...")
            with torch.cuda.device(cfg.DEVICE):
                preprocess_guesswho(cfg, preprocess_dir)
            logger.info("Preprocess guesswho data (complete)")
    else:
        cfg.PLOT.DEPTH_NEAR = cfg.KINECT.DEPTH_NEAR
        cfg.PLOT.DEPTH_FAR = cfg.KINECT.DEPTH_FAR
        # preprocess_dir = f"{cfg.KINECT.PREPROCESS_ROOT_DIR}/{cfg.KINECT.USER}/{cfg.KINECT.SEQUENCE}"
        preprocess_dir = f"{cfg.OUT_DIR}/{cfg.EXPT_NAME}/preprocess"
        if (not Path(f"{preprocess_dir}/initialization").exists()) or (cfg.KINECT.FORCE_PREPROCESS_LMK_SEG or cfg.KINECT.FORCE_PREPROCESS_INIT):
            logger.info("Preprocess kinect data (start)...")
            with torch.cuda.device(cfg.DEVICE):
                preprocess_kinect(cfg, preprocess_dir)
            logger.info("Preprocess kinect data (complete)")

    # exit()  # TODO: remove this line

    if cfg.MODE == "rgb" or cfg.MODE == "rgbd":
        with torch.cuda.device(cfg.DEVICE):
            optimizer = Optimizer(cfg, preprocess_dir)
            optim_dir = f"{cfg.OUT_DIR}/{cfg.EXPT_NAME}/optimization";  utils.create_dir(optim_dir, True)
            logger.info(f"Optimization output will be saved at {optim_dir}")
            optimizer.optimize(optim_dir)
            # optimizer.save_optimized_parameters()

if __name__ == "__main__":
    main()
