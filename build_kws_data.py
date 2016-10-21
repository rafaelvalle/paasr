import os
import ntpath
import glob2 as glob
import numpy as np

import tgt


def find_short_files(filepath, min_length=1.0):
    files = []
    for l in tuple(open(filepath, 'r')):
        if float(l.split('=')[2]) < min_length:
            files.append((l.split(' ')[0]).split('=')[1])
    return files


def get_data(target_glob_str, other_glob_str, file_durations_path, min_length):
    targets_paths = glob.glob(target_glob_str)
    others_paths = glob.glob(other_glob_str)

    short_files = find_short_files(file_durations_path, min_length)
    short_files = [os.path.splitext(ntpath.basename(p))[0] for p in short_files]

    # filter files that are too small
    targets_paths = [
        p for p in targets_paths
        if os.path.splitext(ntpath.basename(p))[0] not in short_files]
    others_paths = [
        p for p in others_paths
        if os.path.splitext(ntpath.basename(p))[0] not in short_files]

    targets = np.array([np.load(p) for p in targets_paths])
    others = np.array([np.load(p) for p in others_paths])

    return targets, others


def data_iterator(targets, others, batch_size, ratio=0.5):
    """
        All items in targets must have length equal to sample size
        Every item in others must be of length at least equal to sample
        size!

    """
    sample_size = targets[0].shape[0]
    while True:
        t = targets[np.random.randint(0, len(targets), int(batch_size*ratio))]
        o = others[np.random.randint(0, len(others), int(batch_size*(1-ratio)))]
        lbls = np.ones(batch_size).astype(int)
        lbls[int(batch_size*ratio):] = 0
        o_sliced = []
        for i in xrange(len(o)):
            rand_start = np.random.randint(0, len(others[i]) - sample_size)
            o_sliced.append(others[i][rand_start:rand_start+sample_size])
        # shuffle order
        ids = np.random.randint(0, batch_size, batch_size)

        yield np.vstack((t, np.array(o_sliced)))[ids], lbls[ids]


def create_target_audio(glob_str, tgt_word, frame_size, left_context,
                        right_context, save_folder):
    """
    glob_str: str
        Glob str to find TextGrid files
    tgt_word : str
    frame_size: f
        In milliseconds
    left_context : f
        In number of frames
    right_context : f
        In number of frames
    save_folder: str
        Folder path where to save target files
    """
    sample_length = frame_size * (left_context + 1 + right_context)

    # iterate over text-grid data
    for filepath in glob.glob(glob_str):
        # declare required paths
        filename = os.path.splitext(ntpath.basename(filepath))[0]
        folder_path = os.path.dirname(filepath)
        save_path = os.path.join(save_folder, "target_{}.wav".format(filename))
        input_path = os.path.join(folder_path, filename+".wav")

        regex = r"(?i)\b{}\b".format(tgt_word)
        words = tgt.read_textgrid(filepath).get_tier_by_name('words')
        for t in words.get_annotations_with_matching_text(regex, regex=True):
            # declare center and desired start and end time
            center_time = (t.end_time + t.start_time) / 2.
            start_time = center_time - (left_context+.5)*frame_size
            end_time = center_time + (right_context+.5)*frame_size

            # word is not long enough for left and right contexts
            if (start_time < words.start_time and end_time > words.end_time):
                params = "-i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    words.start_time,
                    words.end_time,
                    words.end_time / sample_length,
                    save_path)
            # is word long enough
            elif (start_time >= words.start_time and end_time < words.end_time):
                params = "-i {} -ss {} -to {} -acodec copy {}".format(
                    input_path,
                    start_time,
                    end_time,
                    save_path)
            # is there enough right context
            elif end_time < words.end_time:
                params = "-i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    words.start_time,
                    end_time,
                    (end_time - words.start_time) / sample_length,
                    save_path)
            # is there enough left context
            else:
                params = "-i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    start_time,
                    words.end_time,
                    (words.end_time - start_time) / sample_length,
                    save_path)
            # run command
            os.system("ffmpeg -y " + params)
