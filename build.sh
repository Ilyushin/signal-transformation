#!/usr/bin/env bash


rm -rf ./signal_transformation.egg-info ./dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
