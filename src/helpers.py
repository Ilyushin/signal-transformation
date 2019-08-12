import pydub


def create_overlapping_signal(signal1, signal2, read=False):
    '''
    Mix two signals in one
    :param signal1: Can be a path or AudioSegment
    :param signal2: Can be a path or AudioSegment
    :param read: If it is True, siganl1 and signal2 are paths
    :return: AudioSegment of overlapping signal
    '''
    speech1 = signal1
    speech2 = signal2
    if read:
        speech1 = pydub.AudioSegment.from_wav(signal1)
        speech2 = pydub.AudioSegment.from_wav(signal2)

    return speech1.overlay(speech2)