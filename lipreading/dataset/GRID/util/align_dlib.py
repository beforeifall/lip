#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: align_dlib.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/05/30
#   description:
#
#================================================================

import os
import cv2
import dlib
import numpy as np

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


class AlignDlib(object):
    """ Align faces use dlib.

    Align faces keep eye and nose static
    """

    # Landmark indices.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, shape_predictor_path):
        """Init AlignDlib with shape_predictor_path

        Args:
            shape_predictor_path: the dlib shape predictor path.
        """
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(shape_predictor_path)

    def _get_all_face_bounding_boxes(self, rgb_img):
        """Find all face boundings in an image.

        Args:
            rgb_img: Numpy.ndarray. image array.

        Returns:
            A list of dlib.rectangle.

        """
        assert rgb_img is not None

        try:
            return self._detector(rgb_img, 1)
        except Exception as e:
            print('warning: {}'.format(e))
            return []

    def get_largest_face_bounding_boxes(self, rgb_img, skip_multi=False):
        """Find largest face bounding boxes in an image.

        Args:
            rgb_img: Numpy.ndarray. image array.

        Kwargs:
            skip_multi: Skip image if more than one face detected.

        Returns:
            dlib.rectangle. Represent the face bounding box.

        """
        assert rgb_img is not None

        faces = self._get_all_face_bounding_boxes(rgb_img)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def find_face_landmarks(self, rgb_img, bounding_box):
        """Find 68 face landmarks in image given bounding_box.

        Args:
            rgb_img: Numpy.ndarray. image array.
            bounding_box: dlib.rectangle. The detected face bounding box.

        Returns:
            A list of 68 face find_face_landmarks. For example:
            [(x1, y1), (x2, y2), ..., (x68, y68)].

        """
        assert rgbImg is not None
        assert bounding_box is not None

        points = self._predictor(rgbImg, bounding_box)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self,
              rgb_img,
              img_dim,
              ref_landmarks,
              bounding_box=None,
              landmarks=None,
              landmark_indices=INNER_EYES_AND_BOTTOM_LIP,
              skip_multi=False):
        """Align img using perspective transformation.

        Args:
            rgb_img: Numpy.ndarray. image array.
            img_dim: The dim of return img. return shape is (img_dim, img_dim, 3)
            ref_landmarks: List with length 68 of Tuple (x, y). The reference landmarks. 
                The detected landmarks are transformed to the ref_landmarks.

        Kwargs:
            bounding_box: dlib.rectangle. If given, detect landmarks in given bounding_box.
            landmarks: List with length 68 of Tuple (x, y). The landmarks in given rgb_img.
                If given, do not perform landmark detection in rgb_img.
            landmark_indices: List with length 3 of Tupe (x, y). Align along the landmark 
                points. Default is (left_eye_inner, right_eye_inner, nose)
            skip_multi: skip image if true.

        Returns:
            A aligned img of shape (img_dim, img_dim, 3)

        """
        assert img_dim is not None
        assert rgb_img is not None
        assert ref_landmarks is not None
        assert landmark_indices is not None

        if bounding_box is None:
            bounding_box = self.get_largest_face_bounding_boxes(
                rgb_img, skipMulti)
            if bounding_box is None:
                return None

        if landmarks is None:
            landmarks = self.find_face_landmarks(rgb_img, bounding_box)

        np_landmarks = np.float32(landmarks)
        np_ref_landmarks = np.float32(ref_landmarks)

        H = cv2.getAffineTransform(np_landmarks[landmark_indices],
                                   np_ref_landmarks[landmark_indices])
        transformed_img = cv2.warpAffine(rgb_img, H, (img_dim, img_dim))
        return transformed_img
