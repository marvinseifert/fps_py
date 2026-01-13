#!/usr/bin/bash
# This script assumes that your in the activated environment. How to activate if using poetry:
# 	https://python-poetry.org/docs/managing-environments/#bash-csh-zsh

python examples/gui_cal_3brain.py --config ../_configs/fpspy.toml ../data/stim/gaussian_checkerboard_768l-16s-10Hz-5min.h5 --out-dir ../out/cal --no-triggers
