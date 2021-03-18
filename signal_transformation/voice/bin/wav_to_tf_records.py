#!/usr/bin/python3

r"""Script to convert Voice dataset to TensorFlow records.

To run the script setup a virtualenv with the following libraries installed.
- `tensorflow`: Install with `pip3 install tensorflow`

To run the script to preprocess the raw dataset as TFRecords,
run the following command:

```
python3 wav_to_tf_records \
  --metadata_path="./vox1_metadata.gzip" \
  --spec_format=3 \
  --sample_rate=16000 \
  --num_mfcc=13 \
  --spect_shape=[300, 200, 1] \
  --num_shards=512 \
  --output_dir="/dataset/tf_records"
```
"""

import pandas as pd
from absl import app
from absl import flags
from absl import logging

from signal_transformation.voice.tf_transformation import wav_to_tf_records, SpecFormat

flags.DEFINE_string(
    'metadata_path',
    None,
    'Path to a parquet file with the metadata'
    'Example: --metadata_path "./vox1_metadata.gzip"'
)
flags.DEFINE_enum(
    'spect_format',
    '3',
    ['1', '2', '3', '4', '5'],
    'Format of a spectrogram. PCM = 1, STFT = 2, MAGNITUDE = 3, MEL_SPEC = 4, LOG_MEL_SPEC = 5, MFCC = 6'
)
flags.DEFINE_integer(
    'sample_rate',
    16000,
    'Sample rate.'
)
flags.DEFINE_integer(
    'num_mfcc',
    13,
    'Number of the Mel Frequency Cepstral Coefficients.'
)
flags.DEFINE_list(
    'spect_shape',
    [300, 200, 1],
    'A shape of the output spectrogram.'
)
flags.DEFINE_integer(
    'num_shards',
    1024,
    'Number of output shards.'
)
flags.DEFINE_string(
    'output_dir', None, 'Scratch directory path for temporary files.'
)

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.metadata_path is None:
        raise AssertionError('Need to specify a path to the metadata file.')

    spec_format = None
    if FLAGS.spect_format == '1':
        spec_format = SpecFormat.PCM
    elif FLAGS.spect_format == '2':
        spec_format = SpecFormat.STFT
    elif FLAGS.spect_format == '3':
        spec_format = SpecFormat.MAGNITUDE
    elif FLAGS.spect_format == '4':
        spec_format = SpecFormat.MEL_SPEC
    elif FLAGS.spect_format == '5':
        spec_format = SpecFormat.LOG_MEL_SPEC

    metadata = pd.read_parquet(FLAGS.metadata_path)
    wav_to_tf_records(metadata=metadata,
                      sample_rate=FLAGS.sample_rate,
                      output_dir=FLAGS.output_dir,
                      spec_format=spec_format,
                      num_mfcc=FLAGS.num_mfcc,
                      spec_shape=FLAGS.spect_shape,
                      num_shards=FLAGS.num_shards)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
