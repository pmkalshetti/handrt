#!/bin/bash

# make it executable so that subprocess.run can call it: chmod +x code/utils/create_video_from_frames.sh
# usage: bash code/utils/create_video_from_frames.sh [-f framerate] [-s start_number] [-w wildcard_string] dir_frames 

# Ref: https://stackoverflow.com/questions/18115885/getopts-not-working-in-bash-script/18116001
framerate=30
start_number=0
wildcard="%d.png"
output_filename="video.mp4"
while getopts "f:s:w:o:" opt
do
    case $opt in 
        f) framerate=${OPTARG}
           ;;
        s) start_number=${OPTARG}
           ;;
        w) wildcard=${OPTARG}
           ;;
        o) output_filename=${OPTARG}
           ;; 
        \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
    esac
done
shift $(( OPTIND - 1 )) # remove optional parameter numbers so that $1 works correctly for subsequent lines

dir_frames=$1

# Ref: https://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion
# ffmpeg -v quiet -stats -f image2 -framerate $framerate -start_number 0 -i $dir_frames/%05d.png $dir_frames/video.mp4
ffmpeg -v quiet -stats -f image2 -framerate $framerate -start_number $start_number -i $dir_frames/$wildcard -pix_fmt yuv420p $dir_frames/$output_filename