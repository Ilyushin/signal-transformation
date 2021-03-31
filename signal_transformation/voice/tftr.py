# pylint: disable=too-many-arguments, too-many-locals

'''
Transformation of a signal which based on TensorFlow
'''

import os
import math
from typing import Iterable, List, Mapping, Union, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import signal_transformation.helpers as helpers
from signal_transformation.voice.SpectFormat import SpectFormat


def wav_to_pcm(wav_file):
    '''
    Transform a waf file to PCM
    :param wav_file:Path to a wav file
    :return: PCM
    '''
    raw_audio = tf.io.read_file(wav_file)
    waveform = tf.audio.decode_wav(
        raw_audio,
        desired_channels=1
    )

    return waveform


def pcm_to_stft(
        signals,
        frame_length=1024,
        frame_step=256,
        fft_length=1024,
        pad_end=True
):
    '''
    Transform signal to STFT
    :param signals:
    :param frame_length:
    :param frame_step:
    :param fft_length:
    :param pad_end:
    :return: STFT
    '''
    stft = tf.signal.stft(
        signals,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        pad_end=pad_end
    )

    return stft


def stft_to_magnitude(stft):
    '''
    Transform stft to a magnitude spectrogram
    :param stft:
    :return:
    '''

    return tf.abs(stft)


def magnitude_to_mel_spec(
        magnitude_spectrograms,
        sample_rate=16000,
        fmin=80,
        fmax=8000,
        num_mel_bins=80
):
    '''
    Transfrom STFT to a mel spectrogram
    :param magnitude_spectrograms:
    :param sample_rate:
    :param fmin:
    :param fmax:
    :param num_mel_bins:
    :return:
    '''

    # magnitude_spectrograms = stft_to_magnitude(stft)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        fmin,
        fmax
    )

    mel_spec = tf.tensordot(
        magnitude_spectrograms,
        linear_to_mel_weight_matrix,
        1
    )

    return mel_spec


def mel_spec_to_mfcc(mel_spec, num_mfccs=13, log_offset=1e-6):
    '''
    Return mfcc
    :param mel_spec:
    :param num_mfccs:
    :param log_offset:
    :return:
    '''

    log_mel_spec = tf.math.log(mel_spec + log_offset)

    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spec)[..., :num_mfccs]

    return mfcc


def wrap_int64(value):
    '''
    Return int64 list
    :param value:
    :return:
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_list_int64(value):
    '''
    Return int64 list
    :param value:
    :return:
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_float(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def wrap_bytes(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def crop_pcm(value, sample_rate=16000, time=5):
    """Returns a cropped part of PCM."""
    start = sample_rate
    end = time * sample_rate + start

    contain_time = int(value.shape[0] / sample_rate)
    if time > contain_time or end > value.shape[0]:
        return value

    return value[start:end]


def parse_wav(file_path: str,
              sample_rate: int,
              num_mfcc: int,
              spec_format: SpectFormat,
              result_shape: Tuple[int, int, int]):
    '''
    Parse a wav file in an appropriate format
    :param file_path:
    :param sample_rate:
    :param num_mfcc:
    :param spec_format:
    :param result_shape:
    :return: need representation of a signal as a tensor
    '''
    pcm = wav_to_pcm(file_path).audio
    pcm = crop_pcm(pcm, sample_rate=sample_rate)
    signals = tf.reshape(pcm, [1, -1])

    # Step 1 : signals->stfts
    # `stft` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1 = 513.
    stfts = pcm_to_stft(signals)

    # Step2 : stfts->magnitude_spectrograms
    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    magnitude_spectrograms = stft_to_magnitude(stfts)

    # Step 3 : magnitude_spectrograms->mel_spectrograms
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    mel_spectrograms = magnitude_to_mel_spec(magnitude_spectrograms)

    # Step 4 : mel_spectrograms->log_mel_spectrograms
    log_offset = 1e-6
    log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

    # Step 5 : log_mel_spectrograms->mfccs
    # Keep the first `num_mfccs` MFCCs.
    spectrogram = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfcc]

    spect = signals.numpy()
    if spec_format == SpectFormat.STFT:
        spect = stfts
    elif spec_format == SpectFormat.MAGNITUDE:
        spect = magnitude_spectrograms
    elif spec_format == SpectFormat.MEL_SPEC:
        spect = mel_spectrograms
    elif spec_format == SpectFormat.LOG_MEL_SPEC:
        spect = log_mel_spectrograms
    elif spec_format == SpectFormat.MFCC:
        spect = spectrogram

    if not spec_format == SpectFormat.PCM:
        x, y, z = None, None, None
        x = max(spect.shape)
        if len(spect.shape) > 1:
            z = min(spect.shape)

        if len(spect.shape) == 3:
            set_dif = set(spect.shape).difference(set([x, z]))
            y = set_dif.pop() if len(set_dif) else x

        spect = np.reshape(np.array(spect), (x, y, z))

        if spect.shape[0] < result_shape[0] or spect.shape[1] < result_shape[1]:
            return None

        spect = spect[:result_shape[0], :result_shape[1], :result_shape[2]]

    return spect


def parse_chunk(chunk_files: pd.DataFrame,
                sample_rate=16000,
                output_file=None,
                spec_format=SpectFormat.PCM,
                num_mfcc=13,
                spec_shape=(300, 200, 1)):
    '''
    Covert a chunk of files to tf records
    :param chunk_files:
    :param sample_rate:l
    :param output_file:
    :param spec_format:
    :param num_mfcc:
    :param spec_shape:
    :return:
    '''

    writer = tf.io.TFRecordWriter(output_file)
    for file_path, label, class_id in zip(chunk_files.file_path, chunk_files.label,
                                          chunk_files.class_id):
        spect = parse_wav(file_path, sample_rate, num_mfcc, spec_format, spec_shape)

        if spect is None:
            continue

        # Create a dict with the data we want to save
        data = {
            'spectrogram': wrap_bytes(spect.flatten().tostring()),
            'label': wrap_int64(label),
            'class_id': wrap_int64(class_id),
            'height': wrap_int64(spect.shape[0] if len(spect.shape) >= 1 else 0),
            'width': wrap_int64(spect.shape[1] if len(spect.shape) >= 2 else 0),
            'channels': wrap_int64(spect.shape[2] if len(spect.shape) >= 3 else 0)
        }
        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)
        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)
        # Serialize the data.
        serialized = example.SerializeToString()
        # Write the serialized data to the TFRecords file.
        writer.write(serialized)

    writer.close()


# def prase_sub_df():


def wav_to_tf_records(metadata=None,
                      sample_rate=16000,
                      output_dir=None,
                      spec_format=SpectFormat.PCM,
                      num_mfcc=13,
                      spec_shape=(300, 200, 1),
                      num_shards=512):
    '''
    Convert wav files to TFRecords
    :param metadata: DataFrame, that contains information about files and labels
    :param sample_rate:
    :param output_dir: Path to an output directory
    :param spec_format: Needed format of a spectrogram
    :param num_mfcc:
    :param spec_shape:
    :param pattern:
    :param size: How many files needs to parse
    :param num_shards: How many shards needs to create from a source files
    :return:
    '''

    # Number of images. Used when printing the progress.
    num_files = len(metadata)
    chunksize = num_files // num_shards

    # Iterate over all the image-paths and class-labels.
    print('Started parsing to TFRecord')
    for shard in range(num_shards):
        chunk_files = metadata[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(
            output_dir, '%.5d-of-%.5d.tfrecords' % (shard, num_shards)
        )
        helpers.create_dir(os.path.dirname(output_file))
        parse_chunk(chunk_files, sample_rate, output_file, spec_format, num_mfcc, spec_shape)


def wav_to_numpy_arrays(audio_path=None,
                        label=None,
                        sample_rate=16000,
                        out_path=None,
                        spec_format=SpectFormat.PCM,
                        num_mfcc=13,
                        spec_shape=(300, 200, 1),
                        pattern=['.wav', ],
                        size=None
                        ):
    '''
    Convert wav files to Numpy arrays
    :param audio_path: Path to wav files
    :param label:
    :param sample_rate:
    :param out_path: Path to
    :param spec_format: Need a format of a spectrogram
    :param num_mfcc:
    :param spec_shape:
    :param pattern:
    :param size: How many files need to parse
    :return:
    '''

    # Number of images. Used when printing the progress.
    source_files = [item for item in helpers.find_files(audio_path, pattern=pattern)]
    num_files = len(source_files)

    # Iterate over all the image-paths and class-labels.
    print()
    print('Started parsing to Numpy arrays')
    i = 0
    for file_path in source_files:
        # Print the percentage-progress.
        helpers.print_progress(count=i, total=num_files - 1)
        i += 1

        if size and i > size:
            break

        speaker_id = file_path.split('/')[-3]
        spect = parse_wav(file_path, sample_rate, num_mfcc, spec_format, spec_shape)

        if spect is None:
            continue

        result_file = file_path.replace(audio_path, out_path).replace('.wav', '.npy')
        dir_path = '/'.join(result_file.split('/')[:len(result_file.split('/')) - 1])
        helpers.create_dir(dir_path)

        spect = spect.astype(float)

        np.save(result_file, spect)


def parse_vox_tf_example(tf_example: tf.train.Example,
                         normalize=True,
                         to_categorical=False,
                         num_classes=0):
    """
    Parse a TF example of VoxCeleb TF record
    :param num_classes:
    :param to_categorical:
    :param normalize:
    :param tf_example:
    :return:
    """

    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_example = tf.io.parse_single_example(tf_example, feature_description)
    spect = tf.reshape(
        tf.io.decode_raw(parsed_example['spectrogram'], tf.float32),
        [parsed_example['height'], parsed_example['width'], parsed_example['channels']]
    )
    y = parsed_example['class_id']

    if normalize:
        mean = tnp.mean(spect, dtype=tf.float32, keepdims=True)
        std = tnp.std(spect, 0, keepdims=True)
        spect = (spect - mean) / (std + 1e-5)

    if to_categorical and num_classes > 0:
        y = tf.one_hot(y, num_classes)

    return spect, y
