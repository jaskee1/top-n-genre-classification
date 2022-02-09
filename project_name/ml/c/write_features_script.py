import sys
# import os
import time

# script_dir = os.path.dirname(__file__)
# mymodule_dir = os.path.join(script_dir, '..', '..')
# sys.path.append(mymodule_dir)

import project_name.data.data_loader as dl           # noqa: E402
import project_name.data.feature_extractor as fe     # noqa: E402
import project_name.data.feature_recorder as fr      # noqa: E402

DEBUG = False

if __name__ == '__main__':

    start_time = time.time()

    data_type = 'gtzan'

    if len(sys.argv) > 1:
        data_type = sys.argv[1]
    if len(sys.argv) > 2:
        fma_set = sys.argv[2]

    loader = dl.DataLoader(data_type=data_type)
    extractor = fe.FeatureExtractor(variant="C")
    recorder = fr.FeatureRecorder()

    raw_data = loader.gather_data('au')

    for index, row in raw_data.iterrows():
        file_path = row['data']
        label = row['label']
        # Extract the features
        features = extractor.extract(file_path)
        # Pack features and label into a protobuf for writing to disk
        protobuf = recorder.packIntoProtobuf([features, label])
        # Write them to a binary .tfrecord file
        recorder.write_tfrecord(protobuf, file_path)

        if DEBUG and index < 10:
            print(recorder.read_tfrecord(file_path))

        if DEBUG and index == 10:
            break

    stop_time = time.time()
    count = raw_data.shape[0]

    print("processed {} audio tracks: {} seconds".format(
        count, (stop_time - start_time)))
    print("{} seconds per track".format((stop_time - start_time)/count))
