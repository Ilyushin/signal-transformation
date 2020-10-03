# pylint: disable=too-many-arguments, too-many-locals

'''
Transformation of a signal which based on TensorFlow
'''

import os
from enum import Enum
import numpy as np
import tensorflow as tf
import signal_transformation.helpers as helpers


class SpecFormat(Enum):
    '''
    Types of spectrogram's formats
    '''
    PCM = 1
    STFT = 2
    MEL_SPEC = 3
    LOG_MEL_SPEC = 4
    MFCC = 5


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


def stft_to_mel_spec(
        stfts,
        sample_rate=16000,
        fmin=80,
        fmax=8000,
        num_mel_bins=80
):
    '''
    Transfrom STFT to a mel spectrogram
    :param stfts:
    :param sample_rate:
    :param fmin:
    :param fmax:
    :param num_mel_bins:
    :return:
    '''
    magnitude_spectrograms = tf.abs(stfts)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    # num_mel_bins = num_mel_bins

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

    contain_time = int(value.shape[0]/sample_rate)
    if time > contain_time or end > value.shape[0]:
        return value

    return value[start:end]


def wav_to_tf_records(
        audio_path=None,
        label=None,
        sample_rate=16000,
        out_path=None,
        spec_format=SpecFormat.PCM,
        num_mfcc=13,
        spec_shape=(300, 200, 1),
        pattern=['.wav', ],
        size=None
):
    '''
    Convert wav files to TFRecords
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
    print('Started parsing to TFRecord')
    i = 0
    for file_path in source_files:
        # Print the percentage-progress.
        helpers.print_progress(count=i, total=num_files - 1)
        i += 1

        if size and i > size:
            break

        speaker_id = file_path.split('/')[-3]
        pcm = wav_to_pcm(file_path).audio
        pcm = crop_pcm(pcm, sample_rate=16000)
        signals = tf.reshape(pcm, [1, -1])

        # Step 1 : signals->stfts
        # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
        # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
        # where fft_unique_bins = fft_length // 2 + 1 = 513.
        stfts = pcm_to_stft(signals)

        # Step2 : stfts->magnitude_spectrograms
        # An energy spectrogram is the magnitude of the complex-valued STFT.
        # A float32 Tensor of shape [batch_size, ?, 513].
        # Step 3 : magnitude_spectrograms->mel_spectrograms
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        mel_spectrograms = stft_to_mel_spec(stfts)

        # Step 4 : mel_spectrograms->log_mel_spectrograms
        log_offset = 1e-6
        log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

        # Step 5 : log_mel_spectrograms->mfccs
        # Keep the first `num_mfccs` MFCCs.
        spectrogram = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)[..., :num_mfcc]

        spect = signals.numpy()
        if spec_format == SpecFormat.STFT:
            spect = stfts
        elif spec_format == SpecFormat.MEL_SPEC:
            spect = mel_spectrograms
        elif spec_format == SpecFormat.LOG_MEL_SPEC:
            spect = log_mel_spectrograms
        elif spec_format == SpecFormat.MFCC:
            spect = spectrogram

        if not spec_format == SpecFormat.PCM:
            x, y, z = None, None, None
            x = max(spect.shape)
            if len(spect.shape) > 1:
                z = min(spect.shape)

            if len(spect.shape) == 3:
                set_dif = set(spect.shape).difference(set([x, z]))
                y = set_dif.pop() if len(set_dif) else x

            spect = np.reshape(np.array(spect), (x, y, z))

            if spect.shape[0] < spec_shape[0] or spect.shape[1] < spec_shape[1]:
                continue

            spect = spect[:spec_shape[0], :spec_shape[1], :spec_shape[2]]

        # Create a dict with the data we want to save
        data = {
            'spectrogram': wrap_float(spect.flatten()),
            'label': wrap_int64(int(speaker_id.replace('id', ''))),
            'height': wrap_int64(spect.shape[0] if len(spect.shape) >= 1 else 0),
            'width': wrap_int64(spect.shape[1] if len(spect.shape) >= 2 else 0),
            'depth': wrap_int64(spect.shape[2] if len(spect.shape) >= 3 else 0)
        }
        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=data)
        # Wrap again as a TensorFlow Example.
        example = tf.train.Example(features=feature)
        # Serialize the data.
        serialized = example.SerializeToString()
        # Write the serialized data to the TFRecords file.
        # Open a TFRecordWriter for the output-file.
        result_file = file_path.replace(audio_path, out_path).replace('.wav', '.tfrecords')
        dir_path = '/'.join(result_file.split('/')[:len(result_file.split('/')) - 1])
        helpers.create_dir(dir_path)

        with tf.io.TFRecordWriter(result_file) as writer:
            writer.write(serialized)


def wav_to_numpy_arrays(
        audio_path=None,
        label=None,
        sample_rate=16000,
        out_path=None,
        spec_format=SpecFormat.PCM,
        num_mfcc=13,
        spec_shape=(300, 200, 1),
        pattern=['.wav', ],
        size=None
):
    '''
    Convert wav files to TFRecords
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
        pcm = wav_to_pcm(file_path).audio
        pcm = crop_pcm(pcm, sample_rate=16000)
        signals = tf.reshape(pcm, [1, -1])

        # Step 1 : signals->stfts
        # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
        # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
        # where fft_unique_bins = fft_length // 2 + 1 = 513.
        stfts = pcm_to_stft(signals)

        # Step2 : stfts->magnitude_spectrograms
        # An energy spectrogram is the magnitude of the complex-valued STFT.
        # A float32 Tensor of shape [batch_size, ?, 513].
        # Step 3 : magnitude_spectrograms->mel_spectrograms
        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        mel_spectrograms = stft_to_mel_spec(stfts)

        # Step 4 : mel_spectrograms->log_mel_spectrograms
        log_offset = 1e-6
        log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

        # Step 5 : log_mel_spectrograms->mfccs
        # Keep the first `num_mfccs` MFCCs.
        spectrogram = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)[..., :num_mfcc]

        spect = signals.numpy()
        if spec_format == SpecFormat.STFT:
            spect = stfts
        elif spec_format == SpecFormat.MEL_SPEC:
            spect = mel_spectrograms
        elif spec_format == SpecFormat.LOG_MEL_SPEC:
            spect = log_mel_spectrograms
        elif spec_format == SpecFormat.MFCC:
            spect = spectrogram

        if not spec_format == SpecFormat.PCM:
            x, y, z = None, None, None
            x = max(spect.shape)
            if len(spect.shape) > 1:
                z = min(spect.shape)

            if len(spect.shape) == 3:
                set_dif = set(spect.shape).difference(set([x, z]))
                y = set_dif.pop() if len(set_dif) else x

            spect = np.reshape(np.array(spect), (x, y, z))

            if spect.shape[0] < spec_shape[0] or spect.shape[1] < spec_shape[1]:
                continue

            spect = spect[:spec_shape[0], :spec_shape[1], :spec_shape[2]]

        result_file = file_path.replace(audio_path, out_path).replace('.wav', '.npy')
        dir_path = '/'.join(result_file.split('/')[:len(result_file.split('/')) - 1])
        helpers.create_dir(dir_path)

        spect = spect.astype(float)

        np.save(result_file, spect)
