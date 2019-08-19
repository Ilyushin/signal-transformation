#!/usr/bin/env bash


rm -rf ./signal_transformation.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
pip3 uninstall -y signal-transformation
pip3 install ./dist/signal_transformation-1.0.17-py3-none-any.whl