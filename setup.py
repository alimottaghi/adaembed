#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="adaembed",
    version="1.0",
    author="Ali Mottaghi",
    url="unknown",
    description="AdaEmbed: Semi-supervised Domain Adaptation in the Embedding Space",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "opencv-python",
        "pandas",
        "torchvision>=0.4.2",
        "Pillow",
        "pytorchvideo",
        "scikit-learn",
        "einops",
        "tensorboard",
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
