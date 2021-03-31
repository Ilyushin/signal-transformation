'''
Transformation of a signal which based on Librosa
'''
import librosa


def wav_to_pcm(wav_file, sample_rate=16000, trimmed=False):
    '''
    Transform a waf file to PCM
    :param wav_file: Path to a wav file
    :param sample_rate:
    :param trimmed:
    :return: PCM
    '''
    pcm, sr = librosa.core.load(
        wav_file,
        sr=sample_rate
    )

    if trimmed:
        pcm, _ = librosa.effects.trim(pcm)

    return pcm


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


def wav_to_magnitude(
        path_wav_file,
        sample_rate=16000,
        trimmed=False,
        n_fft=1024,
        hop_length=256,
        win_length=1024
):
    '''
    Extract magnitude from a signal
    :param path_wav_file:
    :param sample_rate:
    :param trimmed:
    :param n_fft:
    :param hop_length:
    :param win_length:
    :return: STFT
    '''

    pcm = wav_to_pcm(path_wav_file, sample_rate=sample_rate, trimmed=trimmed)
    stft = pcm_to_stft(
        pcm,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    magnitude, _ = librosa.magphase(stft)

    return magnitude
