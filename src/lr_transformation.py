import tensorflow as tf


def wav_to_pcm(wav_file):
    '''
    Transform a waf file to PCM
    :param wav_file:Path to a wav file
    :return: PCM
    '''
    audio_binary = tf.read_file(wav_file)
    waveform = tf.contrib.ffmpeg.decode_audio(
        audio_binary,
        file_format='wav',
        samples_per_second=c.SAMPLE_RATE,
        channel_count=1
    )

    return waveform


def signal_to_stft(
        signals,
        frame_length=c.FRAME_LENGTH,
        frame_step=c.FRAME_STEP,
        fft_length=c.FFT_LENGTH,
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
    stft = tf.contrib.signal.stft(
        signals,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        pad_end=pad_end
    )

    return stft


def stfts_to_mel_spec(stfts):
    '''
    Transfrom STFT to a mel spectrogram
    :param stfts:
    :return:
    '''
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # num_spectrogram_bins = 300

    num_mel_bins = c.SPEC_SHAPE[0]

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        c.SAMPLE_RATE,
        c.FMIN,
        c.FMAX
    )

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms,
        linear_to_mel_weight_matrix,
        1
    )

    return mel_spectrograms
