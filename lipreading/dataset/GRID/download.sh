#!/bin/bash

#================================================================
#   God Bless You. 
#   
#   file name: download_script.sh
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/05/26
#   description: 
#
#================================================================


# directory dataset will save to.
DOWNLOAD_DIR=`pwd`/downloaded

VIDEO_DIR=video
VIDEO_URL="http://spandh.dcs.shef.ac.uk/gridcorpus/s#/video/s#.mpg_vcd.zip"
ALIGN_DIR=align
ALIGN_URL="http://spandh.dcs.shef.ac.uk/gridcorpus/s#/align/s#.tar"

mkdir -p $DOWNLOAD_DIR/$VIDEO_DIR
mkdir -p $DOWNLOAD_DIR/$ALIGN_DIR

for i in `seq 1 34`; 
do

    if [ $i != 21 ]; then

        # download video
        video_url=${VIDEO_URL//#/$i}
        if [ ! -f $DOWNLOAD_DIR/$VIDEO_DIR/s$i.mpg_vcd.zip ]; then
            wget $video_url -P $DOWNLOAD_DIR/$VIDEO_DIR
        else
            echo ALREADY DOWNLOAD VIDEO: $video_url
        fi

        # download align
        align_url=${ALIGN_URL//#/$i} 
        if [ ! -f $DOWNLOAD_DIR/$ALIGN_DIR/s$i.tar ]; then
            wget $align_url -P $DOWNLOAD_DIR/$ALIGN_DIR
        else
            echo ALREADY DOWNLOAD ALIGN: $align_url
        fi
    fi
done

