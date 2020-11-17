import os
import sys
import time
from itertools import permutations
import pickle
from pydub import AudioSegment


def print_progress(count, total):
    '''
    Print a progress in the terminal
    :param count:
    :param total:
    :return:
    '''
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def find_files(directory, pattern=['.wav']):
    '''
    Recursively finds all files matching the pattern
    :param directory: Path to a directory with files
    :param pattern: extension of the files
    :return: Generator via files
    '''
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if len(pattern):
                for exten in pattern:
                    if filename.endswith(exten):
                        yield os.path.join(root, filename)
            else:
                yield os.path.join(root, filename)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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
        speech1 = AudioSegment.from_wav(signal1)
        speech2 = AudioSegment.from_wav(signal2)

    return speech1.overlay(speech2)


def create_overlapping_dataset(
        input_folder,
        output_folder,
        size=None,
        speech_time=2,
        silence_time=1,
        speakers_number=2,
        overlapping=False,
        only_overlapping=False,
        pattern='.wav'
):
    '''
    :param input_folder:
    :param output_folder:
    :param size: Number of examples
    :param speech_time:
    :param silence_time: Time of silence which need to add between speakers
    :param speakers_number: How many speakers need to combine into one example
    :param overlapping: Mix some speakers into one period of time
    :param only_overlapping:
    :param pattern:
    :return: Dictionary with information about examples
    '''

    file_names = find_files(input_folder, pattern)

    already_added = set()
    files = []
    for file_name in file_names:
        user_id = file_name.split('/')[-3]
        if user_id not in already_added:
            files.append(file_name)
            already_added.add(user_id)

    perms = permutations(files, speakers_number)
    perms = list(perms)

    num_files = len(perms)

    result = {}
    time = speech_time * 1000
    silence = AudioSegment.silent(int(silence_time * 1000))
    counter = 0
    for speakers in perms:
        if size and counter > size:
            break

        counter += 1
        print_progress(count=counter, total=num_files - 1)

        # Read wav files and put them into list as tuple (speaker_id, source data)
        speech = []
        text_key = ''
        check_len = True
        for wav_path in speakers:
            speaker_id = wav_path.split('/')[-3]
            source_data = AudioSegment.from_wav(wav_path)
            speech.append((speaker_id, source_data))
            text_key += speaker_id if not len(text_key) else '_{0}'.format(speaker_id)
            if len(source_data) + 20 < time:
                check_len = False
                break

        # If any file has len shorter than needed, discard this example
        if not check_len:
            continue

        # Init info in log in ground truth file
        if text_key not in result:
            result[text_key] = {
                item[0]: [] for item in speech
            }

        # Put piece of speech in the example
        example = None
        for item in speech:
            if example:
                example += silence + item[1][10:time + 10]
            else:
                example = item[1][10:time + 10]

            result[text_key][item[0]].append((10, time + 10))

        # Add piece of overlapping speech
        if overlapping and not only_overlapping:
            example = example + silence + \
                      (
                          create_overlapping_signal(speech[0][1], speech[1][1])
                      )[0:time]
        elif only_overlapping:
            example = (create_overlapping_signal(speech[0][1], speech[1][1]))[0:time]

        file_path = os.path.join(
            output_folder,
            '{0}.wav'.format(text_key)
        )
        example.export(file_path, format='wav')

    if len(result.keys()):
        with open(os.path.join(output_folder, 'result.pickle'), 'wb') as file_result:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(result, file_result, pickle.HIGHEST_PROTOCOL)

    return result


def mp3_to_wav(input_file, output_file, channels=1):
    if not input_file:
        print('Path to an input file is empty')
        return

    if not output_file:
        print('Path to an output file is empty')
        return

    try:
        AudioSegment.from_file(
            input_file,
            format="mp3",
            channels=channels
        ).export(
            output_file,
            format='wav'
        )
    except:
        print('Can not transform mp3 to wav!')


def clock():
    try:
        return time.perf_counter()  # Python 3
    except:
        return time.clock() # Python 2


class Timer(object):
    # Begin of `with` block
    def __enter__(self, granularity='s'):
        '''
        :param granularity: can be s - seconds, m - minuts, h - hours
        :return:
        '''
        self.granularity = granularity
        self.start_time = clock()
        self.end_time = None
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        if self.granularity == 'm':
            self.end_time = clock()/60
        elif self.granularity == 'h':
            self.end_time = (clock()/60)/60
        else:
            self.end_time = clock()

    def elapsed_time(self):
        """Return elapsed time in seconds"""
        if self.end_time is None:
            # still running
            return clock() - self.start_time
        else:
            return self.end_time - self.start_time

