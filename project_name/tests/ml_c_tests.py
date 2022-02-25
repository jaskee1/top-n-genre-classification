from pathlib import Path

import pandas as pd

import project_name.data.data_loader as dl
from project_name.ml.c.ml_algo_c import MlAlgoC

# Are we using the resnet algo_c variant?
USE_RESNET = False
HARD_PATHS = [
    r'C:\Users\Jason\Music\Music\Alice Phoebe Lou\Live\Berlin Blues.mp3',
    r'C:\Users\Jason\Music\Music\Kings of Leon\[2010] Come Around Sundown '
    r'(Deluxe Edition)\CD 1\03 - Pyro.mp3',
    r'C:\Users\Jason\Music\Music\Purity Ring\Shrines\02 Fineshrine.mp3',
    r'C:\Users\Jason\Music\Music\Sylvan Esso\Sylvan Esso - Sylvan Esso '
    r'[2014] 320 CD\01 Hey Mami.mp3',
    r'C:\Users\Jason\Music\Music\Warpaint\The Fool\01 - Set Your Arms Down.mp3'
]


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


def prediction_test_custom_music(filepaths=HARD_PATHS):

    test = pd.DataFrame({
        'filename': HARD_PATHS,
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

    # prediction_test()
    prediction_test_custom_music()
