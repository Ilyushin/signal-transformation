#!/usr/bin/env bash


rm -rf ./signal_transformation.egg-info ./dist ./build
python3 setup.py sdist bdist_wheel
# TODO need to uncomment
python3 -m twine upload dist/*

# TODO need to comment
#pip3 uninstall -y signal-transformation
#pip3 install ./dist/signal_transformation-1.0.15-py3-none-any.whl