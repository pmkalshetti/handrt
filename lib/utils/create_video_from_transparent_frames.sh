#!/bin/bash

# make it executable so that subprocess.run can call it: chmod +x create_video_from_transparent_frames.sh
# usage: bash create_video_from_transparent_frames.sh [-f framerate] [-s start_number] [-w wildcard_string] [-o output_filename] width height dir_frames 

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

width=$1
height=$2
dir_frames=$3

# Ref: https://ffmpeg.org/ffmpeg.html#Video-and-Audio-file-format-conversion
# ffmpeg -v quiet -stats -f image2 -framerate $framerate -start_number 0 -i $dir_frames/%05d.png $dir_frames/video.mp4
# ffmpeg -v quiet -stats -f image2 -framerate $framerate -start_number $start_number -i $dir_frames/$wildcard -pix_fmt yuv420p $dir_frames/$output_filename
# ffmpeg -f image2  -framerate $framerate -start_number $start_number -i $dir_frames/$wildcard -vcodec libx264 -crf 15 -pix_fmt yuva420p $dir_frames/$output_filename
# ffmpeg -f image2  -framerate $framerate -start_number $start_number -i $dir_frames/$wildcard -crf 15 -pix_fmt yuva420p $dir_frames/$output_filename

ffmpeg -v quiet -stats -f lavfi -i color=c=white:s=${width}x${height} -i $dir_frames/$wildcard -shortest -filter_complex "[0:v][1:v]overlay=shortest=1,format=yuv420p[out]" -map "[out]" $dir_frames/tmp.mp4
ffmpeg -v quiet -stats -r $framerate -i $dir_frames/tmp.mp4 $dir_frames/$output_filename
rm $dir_frames/tmp.mp4

# ffmpeg -f lavfi -i testsrc=duration=10:size=854x480:rate=60 -vf "drawtext=text=%{n}:fontsize=72:r=60:x=(w-tw)/2: y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000099" test.mp4