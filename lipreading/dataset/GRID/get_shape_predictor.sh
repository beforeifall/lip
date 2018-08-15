#!/bin/bash

#================================================================
#   God Bless You. 
#   
#   file name: get_shape_predictor.sh
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/06/03
#   description: 
#
#================================================================

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
