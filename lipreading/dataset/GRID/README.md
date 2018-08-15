# GIRD

## 1. What is GRID?
GRID is a large multitalker audiovisual sentence corpus to support joint computational-behavioral studies in speech perception. In brief, the corpus consists of high-quality audio and video (facial) recordings of 1000 sentences spoken by each of 34 talkers (18 male, 16 female). Sentences are of the form "put red at G9 now".  The corpus, together with transcriptions, is freely available for research use. GRID is described in more detail in this paper.

project page: [http://spandh.dcs.shef.ac.uk/gridcorpus/#docs](http://spandh.dcs.shef.ac.uk/gridcorpus/#docs)

## 2. Download
    
```
bash download.sh
```

Change the download dir if needed. The default download directory is `./downloaded`

## 3. Decompress

```
bash decompress.sh
```

Change the directory if needed. The default decompressed directory is `./decompressed`

## 4. Deframe (Optional)

```
python extract_frame.py
```

deframe videos. All videos in dir `./decompressed/video` will be deframe to `./decompressed/frame`. The sub-directory structure will stay the same.

## 5. Extract mouth

1. download dlib shape predictor

```
./get_shape_predictor.sh

```

2. extract mouth using `extract_mouth.py`

For example:
```
python extract_mouth.py ../decompressed/video *.mpg ../decompressed/mouth shape_predictor_68_face_landmarks.dat
```

## 6. Split train and test
There are two types of split. 

  1. Overlapped: 255 videos of each speaker chosen to test set. The remainings are in training set.
  2. Unseen: Speaker [s1,s2,s20,s22] are chosen to test set and the remainings are in training set.

Pre-splited sets are in `./db-split`

## 7. Create tfrecord
Create tfrecord using `create_tfrecord.py`. See usage use `python create_tfrecord.py -h`
An example:

```
python create_tfrecord.py --split_file db-split/overlapped_test.txt --save_file ../../tf-records/overlapped_test.tfrecord --video_dir ../../decompressed/mouth --align_dir ../../decompressed/align
```
