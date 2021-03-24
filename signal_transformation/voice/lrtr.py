'''
Transformation of a signal which based on Librosa
'''
import librosa


def wav_to_pcm(wav_file, sample_rate=16000):
    '''
    Transform a waf file to PCM
    :param wav_file: Path to a wav file
    :param sample_rate:
    :return: PCM
    '''
    waveform, _ = librosa.core.load(
        wav_file,
        sr=sample_rate
    )

    y_trimmed, _ = librosa.effects.trim(waveform)

    return y_trimmed


def pcm_to_stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024
):
    '''
    Transform signal to STFT
    :param waveform:
    :param n_fft:
    :param hop_length:
    :param win_length:
    :return: STFT
    '''
    stft = librosa.core.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    return stft


def pcm_to_mel_spec(
        waveform,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128
):
    '''
    Transfrom wav to a mel spectrogram
    :param waveform:
    :param sample_rate:
    :param n_fft:
    :param hop_length:
    :param n_mels:
    :return:
    '''
    mel_spec = librosa.feature.melspectrogram(
        waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    return mel_spec


def pcm_to_mfcc(waveform, sample_rate=16000, hop_length=256, n_mfcc=13):
    '''
    Compute MFCC features from the pcm
    :param sample_rate:
    :param waveform:
    :param hop_length:
    :param n_mfcc:
    :return:
    '''

    return librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
        n_mfcc=n_mfcc
    )
