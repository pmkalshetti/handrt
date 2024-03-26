#!/bin/sh

# you need to be in the project's root directory when sourcing this bash script
# use `source filename.sh` instead of `bash filename.sh`
# because `conda activate xxx` fails as the functions are not available in subshell (Ref: # https://github.com/conda/conda/issues/7980#issuecomment-441358406)
env_name=rgbd_hand_track
conda activate $env_name
path_our=`realpath our`
export PYTHONPATH=$path_our:$PYTHONPATH