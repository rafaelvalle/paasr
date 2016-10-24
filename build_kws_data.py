import os
import ntpath
import cPickle as pkl
import glob2 as glob
import numpy as np
import tgt


def find_short_files(filepath, min_length=1.0):
    files = []
    for line in list(open(filepath, 'r')):
        filename, dur = line.strip().split(' ')
        if float(dur) < min_length:
            files.append(filename)
    return files


def create_lab_from_filename_text(filepath, savepath):
    def echo(line):
        filename, text = line.strip().split(' ', 1)
        savefolder = os.path.join(savepath, filename+'/')

        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        params = '"{}" > {}'.format(
            text, os.path.join(savefolder, filename+'.lab'))
        print params
        os.system("echo " + params)

    map(echo, list(open(filepath, 'r')))


def create_keyword_audio_from_filename_time(filepath, audiopath, savepath):
    def ffmpeg(line):
        filename, start_time, end_time = line.strip().split(' ')
        params = "-i {} -ss {} -to {} {}".format(
            os.path.join(os.path.join(audiopath, filename), filename+".wav"),
            start_time,
            end_time,
            os.path.join(savepath, "target_"+filename+".wav"))
        os.system("ffmpeg -y " + params)

    map(ffmpeg, list(open(filepath, 'r')))


def compute_scaler(target_glob_str, other_glob_str, file_durations_path,
                   min_length, savepath):
    from sklearn.preprocessing import StandardScaler
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
    data = []
    [data.extend(np.load(p)) for p in targets_paths]
    [data.extend(np.load(p)) for p in others_paths]

    ss = StandardScaler()
    ss.fit(data)
    pkl.dump(ss, open(savepath, "wb"))


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


def data_iterator(targets, others, t_batch_size, o_batch_size, transpose=False):
    """
        All items in targets must have length equal to sample size
        Every item in others must be of length at least equal to sample
        size!

    """
    sample_size = targets[0].shape[0]
    while True:
        t = targets[np.random.randint(0, len(targets), t_batch_size)]
        o = others[np.random.randint(0, len(others), o_batch_size)]
        lbls = np.ones(t_batch_size + o_batch_size).astype(int)
        lbls[t_batch_size:] = 0

        # hack in case target is too long, MUST FIX!
        t_sliced = []
        for i in xrange(len(t)):
            rand_start = np.random.randint(0, len(t[i]) - sample_size + 1)
            if transpose:
                t_sliced.append(t[i][rand_start:rand_start+sample_size].T)
            else:
                t_sliced.append(t[i][rand_start:rand_start+sample_size])

        # select random slice from others
        o_sliced = []
        for i in xrange(len(o)):
            rand_start = np.random.randint(0, len(o[i]) - sample_size + 1)
            if transpose:
                o_sliced.append(o[i][rand_start:rand_start+sample_size].T)
            else:
                o_sliced.append(o[i][rand_start:rand_start+sample_size])

        # shuffle order
        data = np.vstack((np.array(t_sliced), np.array(o_sliced)))
        ids = np.random.randint(0, data.shape[0], data.shape[0])
        yield np.vstack((np.array(t_sliced), np.array(o_sliced)))[ids], lbls[ids]


def create_target_audio(glob_str, tgt_word, frame_size, left_context,
                        right_context, save_folder):
    """
    glob_str: str
        Glob str to find TextGrid files
    tgt_word : str
    frame_size: f
        In seconds
    left_context : f
        In number of frames
    right_context : f
        In number of frames
    save_folder: str
        Folder path where to save target files
    """
    # sample_length = frame_size * (left_context + 1 + right_context)

    # iterate over text-grid data
    for filepath in glob.glob(glob_str):
        # declare required paths
        filename = os.path.splitext(ntpath.basename(filepath))[0]
        folder_path = os.path.dirname(filepath)
        input_path = os.path.join(folder_path, filename+".wav")

        # find audios where target word exist
        regex = r"(?i)\b{}\b".format(tgt_word)
        words = tgt.read_textgrid(filepath).get_tier_by_name('words')
        matches = words.get_annotations_with_matching_text(regex, regex=True)

        # process word ocurrences case by case
        for i in xrange(len(matches)):
            save_path = os.path.join(save_folder, "target_{}_{}.wav".format(
                filename, i))
            t = matches[i]
            # compute center and required start and end time
            center_time = (t.start_time + t.end_time) / 2.
            start_time = center_time - (left_context+.5)*frame_size
            end_time = center_time + (right_context+.5)*frame_size

            # is audio longer than start time and end time
            if (start_time >= words.start_time and end_time <= words.end_time):
                # trim beginning and end
                cmd = "sox {} {} trim {} {}".format(
                    input_path,
                    save_path,
                    start_time,
                    end_time - start_time)
            # if word start and end are shorter than start and end time
            elif (start_time < words.start_time and end_time > words.end_time):
                # zero pad beginning and end
                cmd = "sox {} {} pad {} {}".format(
                    input_path,
                    save_path,
                    words.start_time - start_time,
                    end_time - words.end_time)
                """
                cmd = "-i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    words.start_time,
                    words.end_time,
                    words.end_time / sample_length,
                    save_path)
                """
            # is there enough right context
            elif end_time <= words.end_time:
                # zero pad beginning
                cmd = "sox {} {} trim 0 {} pad {} 0".format(
                    input_path,
                    save_path,
                    end_time,
                    words.start_time - start_time)
                """
                # stretch
                cmd = "ffmpeg -y -i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    words.start_time,
                    end_time,
                    (end_time - words.start_time) / sample_length,
                    save_path)
                """
            # is there enough left context
            else:
                # zero pad end
                cmd = "sox {} {} trim {} pad 0 {}".format(
                    input_path,
                    save_path,
                    start_time,
                    end_time - words.end_time)
                """
                # stretch
                cmd = "-i {} -ss {} -to {} -filter:a atempo='{}' {}".format(
                    input_path,
                    start_time,
                    words.end_time,
                    (words.end_time - start_time) / sample_length,
                    save_path)
                """
            # run command
            os.system(cmd)
