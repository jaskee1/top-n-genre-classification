import sys
import time

import project_name.data.data_loader as dl           # noqa: E402
import project_name.data.feature_extractor as fe     # noqa: E402
import project_name.data.feature_recorder as fr      # noqa: E402

DEBUG = False

if __name__ == '__main__':

    data_type = 'fma'
    fma_set = 'small'

    # Some options for datatype and fma_set on the command line
    if len(sys.argv) > 1:
        data_type = sys.argv[1]
    if len(sys.argv) > 2:
        fma_set = sys.argv[2]

    # Set up sampling rate based on which dataset we're using to avoid
    # doing costly audio resampling as much as possible.
    # Also assign filetype so we know which type of files to look for.
    if data_type == 'gtzan':
        sample_rate = fe.FeatureExtractor.SR_LOW
        filetype = '.au'
    else:
        sample_rate = fe.FeatureExtractor.SR_STANDARD
        filetype = '.mp3'

    # Used to gather up all files and associated them with labels.
    loader = dl.DataLoader(data_type=data_type, fma_set=fma_set)
    # Used to do the actual feature extraction.
    extractor = fe.FeatureExtractor(sample_rate=sample_rate, variant="C")
    # Used to record extracted features in .tfrecord files
    recorder = fr.FeatureRecorder()

    # This returns a pandas dataframe with filenames and labels.
    raw_data = loader.gather_data(filetype)

    start_time = time.time()

    for index, row in raw_data.iterrows():
        file_path = row['filename']
        label = row['label']
        try:
            # Extract the features
            features = extractor.extract(file_path)
            # Pack features and label into a protobuf for writing to disk
            protobuf = recorder.packIntoProtobuf([features, label])
            # Write them to a binary .tfrecord file
            recorder.write_tfrecord(protobuf, file_path, '.c.tfrecord')

            if DEBUG and index < 10:
                print(recorder.read_tfrecord(file_path, '.c.tfrecord'))
            if DEBUG and index == 10:
                break
        except Exception:
            print(f'Could not extract features from {file_path}')

    stop_time = time.time()
    count = raw_data.shape[0]

    print("processed {} audio tracks: {} seconds".format(
        count, (stop_time - start_time)))
    print("{} seconds per track".format((stop_time - start_time)/count))
