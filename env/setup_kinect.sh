#!/bin/sh

env_name=handrt
conda activate $env_name

# Kinect sensor drivers are provided by libfreenect2
# Ref: https://github.com/OpenKinect/libfreenect2#linux
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build
sudo apt install cmake libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev libva-dev libjpeg-dev libopenni2-dev
cmake .. 
make -j7
sudo make install
# Set up udev rules for device access
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
# Run the test program
./bin/Protonect

export LIBFREENECT2_INSTALL_PREFIX="/usr/local"
export LD_LIBRARY_PATH="/usr/local/lib"
# A python interface for libfreenect2 (https://github.com/r9y9/pylibfreenect2)
pip install cython numpy
pip install pylibfreenect2