#!/usr/bin/bash
# This script assumes that your in the activated environment. How to activate if using poetry:
# 	https://python-poetry.org/docs/managing-environments/#bash-csh-zsh

python ./examples/main_3brain.py play ../data/stim/gaussian_checkerboard_768l-8s-20Hz-60min.h5 --loops 2 --config ../_configs/fpspy.toml -vv
