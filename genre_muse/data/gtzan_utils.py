# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import random
import shutil
import librosa
import librosa.display
import time
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class gtzan_utils:
    FILE_EXTENSION = ".au"
    GENRES = ["blues", "classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"]

    def __init__(self,
                 mel_spectrogram_dir: str = os.getcwd() + "\\mels_gtzan",
                 train_dir: str = os.getcwd() + "\\mels_gtzan_train",
                 test_dir: str = os.getcwd() + "\\mels_gtzan_test"):
        self.mel_spectrogram_dir = mel_spectrogram_dir
        self.train_dir = train_dir
        self.test_dir = test_dir

    def buildMelSpectrograms(self):
        """Builds spectrograms from any nested .au files"""
        # Gathering list of files to bring in
        list_of_files = []
        path = os.curdir
        for root, dirs, files in os.walk(path):
            for file in files:
                if (file.endswith(self.FILE_EXTENSION)):
                    list_of_files.append(os.path.join(root, file))

        # If no files found, raise exception
        if not list_of_files:
            raise ValueError("No files found. Please ensure the GTZAN audio files\
                         are located in this directory or child directory.")

        # Creating genres for each file
        for genre in self.GENRES:
            path = f'{self.mel_spectrogram_dir}\\{genre}'
            if not os.path.exists(path):
                os.makedirs(path)

        # Creating mel spectrograms and saving into appropriate folder
        curr = time.time()
        cnt = 0
        num_files = len(list_of_files)
        for name in list_of_files:

            # Status update
            cnt += 1
            print(f'Working on {cnt} out of {num_files}...')

            # Grabbing genre and track_id
            genre_filtered = (name.split(sep="\\")[-1]).split(sep=".")[0]
            track_id = (name.split(sep="\\")[-1]).split(sep=".")[1]

            # Load into librosa
            y, sr = librosa.load(name)
            mels = librosa.feature.melspectrogram(y=y, sr=sr)

            # Create Plot
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            p = plt.imshow(librosa.power_to_db(mels, ref=np.max))

            # Save plot with genre in name and clearing plot
            plt.savefig(
                f'{self.mel_spectrogram_dir}\\{genre_filtered}\\{genre_filtered}_{track_id}.png')
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

        # Time to create mel plots
        print(f"Generating mels took {time.time() - curr} seconds")

    def createTrainTestSamples(self, random_selection: bool = True):
        """ Splits created mel spectrograms into training/testing

        Keyword arguments:
        random_selection -- Boolean indicating whether to shuffle spectrograms
        """

        # Copying mel spectrograms into train directory
        shutil.copytree(self.mel_spectrogram_dir, self.train_dir)

        # For each genre, select 1/4 of dataset for testing
        for genre in self.GENRES:

            # Grabbing random 25 of the 100 files of genre for testing
            filenames = os.listdir(self.train_dir + '\\' + genre)
            if random_selection:
                random.shuffle(filenames)
            test_files = filenames[75:100]

            # If the directory doesn't exist in the training dir, create
            if not os.path.exists(self.test_dir + '\\' + genre):
                os.makedirs(self.test_dir + '\\' + genre)

            # Move testing files from training dir to testing dir
            for f in test_files:
                shutil.move(self.train_dir + '\\' + genre + '\\' + f,
                            self.test_dir + '\\' + genre)