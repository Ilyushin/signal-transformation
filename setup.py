# pylint: disable=line-too-long, invalid-name, missing-docstring

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="signal_transformation",
    version="2.3.0",
    author="Eugene Ilyushin",
    author_email="eugene.ilyushin@gmail.com",
    description="The package allows performing a transformation of an audio signal using TensorFlow or LibROSA",
    long_description="The package allows performing a transformation of an audio signal using TensorFlow or LibROSA",
    long_description_content_type="text/markdown",
    url="https://github.com/Ilyushin/signal-transformation",
    packages=setuptools.find_packages(),
    scripts=[
        'signal_transformation/images/bin/imagenet_to_tf_records',
        'signal_transformation/voice/bin/wav_to_tf_records'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
