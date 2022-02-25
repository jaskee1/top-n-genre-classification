import sys
import time

import project_name.data.data_loader as dl
import project_name.data.feature_extractor as fe
import project_name.data.feature_recorder as fr

DEBUG = False

if __name__ == '__main__':

    data_type = 'fma'
    fma_set = 'medium'

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
    elif data_type == 'fma':
        sample_rate = fe.FeatureExtractor.SR_STANDARD
        filetype = '.mp3'
    elif data_type == 'prop':
        sample_rate = fe.FeatureExtractor.SR_STANDARD
        filetype = '.wav'

    # Used to gather up all files and associated them with labels.
    loader = dl.DataLoader(data_type=data_type, fma_set=fma_set)
    # Used to do the actual feature extraction.
    extractor = fe.FeatureExtractor(sample_rate=sample_rate, variant="C")
    # Used to record extracted features in .tfrecord files
    recorder = fr.FeatureRecorder()

    raw_data = loader.gather_data(filetype)

    start_time = time.time()
    count = raw_data.shape[0]

    for index, row in raw_data.iterrows():
        file_path = row['filename']
        label = row['label']
        try:
            # Extract the features
            features = extractor.extract(file_path)
            # Pack the features into a protobuf
            protobufs = [recorder.packIntoProtobuf([features, label])]
            # Write the protobuf to a binary .tfrecord file
            recorder.write_tfrecord(protobufs, file_path, '.c.tfrecord')

            if DEBUG:
                print(recorder.read_tfrecord(file_path, '.c.tfrecord'))
            if DEBUG and index == 1:
                break

        except Exception:
            print(f'Could not extract features from {file_path}')

        if index % 1000 == 0:
            print(f'Completed {index} / {count}')

    stop_time = time.time()

    print("processed {} audio tracks: {} seconds".format(
        count, (stop_time - start_time)))
    print("{} seconds per track".format((stop_time - start_time)/count))
