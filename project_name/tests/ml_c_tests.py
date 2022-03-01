from pathlib import Path

import pandas as pd

import project_name.data.data_loader as dl
import project_name.globals as globals
from project_name.ml.c.ml_algo_c import MlAlgoC

# Are we using the resnet algo_c variant?
USE_RESNET = False


def prediction_test():
    data_type = 'prop'
    fma_set = 'medium'

    # Used to gather up all the files and associate them with splits and
    # labels.
    loader = dl.DataLoader(data_type=data_type, fma_set=fma_set)
    # Get filenames, labels, and splits
    data = loader.gather_data('.wav')
    # Do bagging (with replacement) on just the training set here
    test = data[data['split'] == 'test']
    test = test.sample(n=10)

    print(test)

    # Set up the algo
    if USE_RESNET:
        ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_RESNET_PATH, use_resnet=True)
    else:
        ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)

    # for filepath in test['filename']:
    for index, row in test.iterrows():
        filepath = row['filename']
        label = row['label']
        features = ml_algo.prep_data_from_file(filepath)
        prediction = ml_algo.predict(features)
        print('prediction values:\n', prediction, sep='')
        print('file:\t', filepath)
        print('genre num:\t', label.index(1), sep='')
        print('genre name:\t', loader.get_genre_name(label.index(1)), sep='')


def prediction_test_custom_music(filepaths):

    test = pd.DataFrame({
        'filename': filepaths,
    })
    test['filename'].map(Path)
    test['filename'].map(str)

    print(test)

    # Set up the algo
    if USE_RESNET:
        ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_RESNET_PATH, use_resnet=True)
    else:
        ml_algo = MlAlgoC(load_path=MlAlgoC.MODEL_PATH)

    # for filepath in test['filename']:
    for index, row in test.iterrows():
        filepath = row['filename']
        features = ml_algo.prep_data_from_file(filepath)
        prediction = ml_algo.predict(features)
        print('prediction values:\n', prediction, sep='')
        print('file:\t', filepath)


if __name__ == '__main__':

    custom_paths_source = globals.RESOURCES_DIR/'text'/'test_file_paths.txt'

    # Run tests on manually picked local files whose paths are written
    # in the custom_paths_source file.
    if custom_paths_source.exists():
        with open(custom_paths_source, 'r') as f:
            custom_paths = f.readlines()
            custom_paths = list(map(lambda x: x.rstrip(), custom_paths))

        prediction_test_custom_music(custom_paths)

    # Run tests on a few random training files
    prediction_test()
