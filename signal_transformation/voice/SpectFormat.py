
from enum import Enum

class SpectFormat(Enum):
    '''
    Types of spectrogram's formats
    '''
    PCM = 1
    STFT = 2
    MAGNITUDE = 3
    MEL_SPEC = 4
    LOG_MEL_SPEC = 5
    MFCC = 6