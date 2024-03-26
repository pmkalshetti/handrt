#!/bin/sh

# Usage: `source create_env.sh` instead of `bash create_env.sh`
# because `conda activate xxx` fails as the functions are not available in subshell (Ref: # https://github.com/conda/conda/issues/7980#issuecomment-441358406)


env_name=handrt
conda env create --name $env_name --file environment.yaml --force
# --force is used to rebuild environment from scratch

conda activate $env_name
