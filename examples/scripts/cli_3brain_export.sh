#!/usr/bin/bash
# This script assumes that your in the activated environment. How to activate if using poetry:
# 	https://python-poetry.org/docs/managing-environments/#bash-csh-zsh

python ./examples/main_3brain.py export ./examples/resources/example_program.py --loops 2 --config ./examples/resources/fpspy.toml -vv
