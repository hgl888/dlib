#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
#   if you have a CPU that supports AVX instructions, you have an Nvidia GPU
#   and you have CUDA installed since this makes things run *much* faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys

import dlib
from skimage import io

if len(sys.argv) < 3:
    print(
        "Call this program like this:\n"
        "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        "You can get the mmod_human_face_detector.dat file from:\n"
        "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    exit()

cnn_face_detection_model = dlib.cnn_face_detection_model_v1(sys.argv[1])
win = dlib.image_window()

for f in sys.argv[2:]:
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = cnn_face_detection_model.cnn_face_detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
