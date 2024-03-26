# Overview
This directory contains the code for our paper "HandRT: Simultaneous Hand Shape and Appearance Reconstruction with Pose Tracking from Monocular RGB-D Video".

## Usage Instructions
1. Setup environment

```
cd env
source create_env.sh
source activate_dev_env.sh
cd ..
```

Optional: Follow instructions in `env/setup_kinect.sh` to install libfreenect2 that is used to access Kinectv2.

2. Download data to `./data`

2.1 Download MANO hand model from https://mano.is.tue.mpg.de/
2.2 Download HTML model from https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/
2.3 Download Segment Anything model from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
2.4 Download GuessWho dataset from http://lgg.epfl.ch/publications/2017/HOnline/guess-who.zip

The folder structure should be as follows:
```
${ROOT}
|-- data
|   |-- html
|   |   |-- TextureBasis
|   |   |-- TextureSet
|   |   |-- ...
|   |-- guess-who
|   |   |-- comparison
|   |   |-- user1
|   |   |-- user2
|   |   |-- ...
|   |-- mano
|   |   |-- models
|   |   |-- webuser
|   |   |-- ...
|   |-- segment_anything_model
|   |   |-- sam_vit_h_4b8939.pth
```


3. Reconstruct hand avatar while tracking pose for the GuessWho dataset (user 1).
```
python scripts/reconstruct_while_tracking.py
```

The preprocessed results will be stored in `output/guesswho/preprocess`, while the final results will be stored in `output/guesswho/optimization`. 

## References
1. Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM TOG, 36(6):245:1–245:17, 2017.
2. Neng Qian, Jiayi Wang, Franziska Mueller, Florian Bernard, Vladislav Golyanik, and Christian Theobalt. Html: A parametric hand texture model for 3d hand reconstruction and personalization. In ECCV, page 54–71, Berlin, Heidelberg, 2020. Springer-Verlag.
3. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick, Segment Anything, In ICCV, 2023.
4. Anastasia Tkach, Andrea Tagliasacchi, Edoardo Remelli, Mark Pauly, and Andrew Fitzgibbon. Online generative model personalization for hand tracking. ACM TOG, 36 (6):1–11, 2017.