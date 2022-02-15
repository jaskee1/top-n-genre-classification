from pathlib import Path
import numpy as np
import pandas as pd


class DataLoader:
    """
    Provides methods for loading gtzan and fma data, including
    finding appropriate files (direct audio or processed feature files),
    getting associated labels, and assigning splits.

    Attributes
    ----------
    data_type : str
        The domain of data the loader should interact with,
        'fma' or 'gtzan'
    fma_set : str
        Which fma set (small, medium, or large) to use.
    split_vals : list[int]
        Values representing the training, validation, testing splits,
        in order. For example, [80, 10, 10] would give an 80/10/10 split.
        ONLY AFFECTS GTZAN. Fma has it's own predefined splits.

    Methods
    -------
    gather_data(file_extension, include_labels)
        Gather files with the given extension and associate them
        with labels and a test/validate/train split.
    get_genre_name(genre_id)
        Get the genre name corresponding to a genre_id.
    """

    DATA_DIR = __file__ + '/../../resources'
    DATA_TYPE = 'gtzan'
    FMA_SUBSETS = ('small', 'medium', 'large')
    # Training / validation / test split.
    # Out of 100 so we can work with integers.
    # If you don't need both validation and test, just add the two sets
    # together later.
    SPLIT_VALS = [75, 15, 10]

    def __init__(self,
                 data_type=DATA_TYPE,
                 data_dir=DATA_DIR,
                 gtzan_dir=None,
                 fma_dir=None,
                 fma_set=FMA_SUBSETS[0],
                 split_vals=SPLIT_VALS):
        """
        Parameters
        ----------
        data_type : str, optional
            The domain of data the loader should interact with,
            'fma' or 'gtzan'
        data_dir : str, optional
            The overarching directory in which the fma and gtzan
            directories are located. Use if gtzan and fma directories
            are together in a non-default location, but otherwise
            have default naming and directory structure.
        gtzan_dir : str, optional
            Override for default gtzan directory. This should be a full
            system path string of the folder containing the 10 gtzan genre
            folders.
        fma_dir : str, optional
            Override for default fma directory. This should be a full
            system path string of the folder containing the 155 numbered
            fma audio-containing folders.
        fma_set : str, optional
            Which fma set (small, medium, or large) to use.
        split_vals : list[int], optional
            Values representing the training, validation, testing splits,
            in order. For example, [80, 10, 10] would give an 80/10/10 split.
            ONLY AFFECTS GTZAN. Fma has it's own predefined splits.
        """

        self.data_type = data_type
        self.fma_set = fma_set
        self.split_vals = split_vals

        # Set the data directory as a Path object and resolve
        # data-containing folders dependent upon input parameters
        data_dir = Path(data_dir)
        if gtzan_dir is None:
            self._gtzan_dir = data_dir / 'gtzan' / 'genres'
        else:
            self._gtzan_dir = Path(gtzan_dir)

        if fma_dir is None:
            self._fma_dir = data_dir / 'fma' / f'fma_{self.fma_set}'
        else:
            self._fma_dir = Path(fma_dir)

        self._fma_meta_dir = self._fma_dir / '..' / 'fma_metadata'

        # Set up data type so the Data Processor can be customized
        # for compatibility with different datasets while maintaining a
        # consistent interface.
        if data_type == 'gtzan':
            self.gather_data = self._gather_data_gtzan
            self._genre_dict = {
                v.name: i for i, v in enumerate(self._gtzan_dir.iterdir())}

        elif data_type == 'fma':
            self.gather_data = self._gather_data_fma
            genres = pd.read_csv(self._fma_meta_dir / 'genres.csv',
                                 index_col=0,
                                 usecols=['genre_id', 'title'])

            # These can be used later if we want to clean up fma genre
            # lists to only include the small or medium genre sets.
            # Clean up at a later date.
   
            # genres = [
            #     'Hip-Hop',
            #     'Pop',
            #     'Folk',
            #     'Experimental',
            #     'Rock',
            #     'International',
            #     'Electronic',
            #     'Instrumental'
            # ]
            # genres = [
            #     'Hip-Hop',
            #     'Pop',w
            #     'Folk',
            #     'Experimental',
            #     'Rock',
            #     'International',
            #     'Electronic',
            #     'Instrumental'
            #     'Classical',
            #     'Old-Time / Historic',
            #     'Jazz',
            #     'Country',
            #     'Soul-RnB',
            #     'Spoken',
            #     'Blues',
            #     'Easy Listening'
            # ]
            # genres = pd.DataFrame({'title': genres})

            self._genre_dict = {v: i for i, v in enumerate(genres['title'])}

        else:
            self._genre_dict = {}

        self._genre_inverse_dict = {v: k for k, v in self._genre_dict.items()}

    def gather_data(self, file_extension, include_labels=True):
        """
        Gather files with the given extension and associate them
        with labels and a test/validate/train split.

        Resolves to the corresponding function, depending on the data
        type (gtzan or fma) set in the Data Loader constructor. In either
        case, it will collect files with the given file_extension
        from the appropriate folder.

        Parameters
        ----------
        file_extension : str
            The file extension of files we want to gather. Should include
            the leading dot. Examples: '.mp3', '.au', '.c.tfrecord'.
        include_labels : bool, optional
            Whether to include the labels column in the returned results

        Returns
        ------
        pandas.DataFrame
            A dataframe where each row represents a file. Columns are
            'filename', 'label', and 'split'.
            Filename is the full path string to the file.
            Label is an array of 0s with 1s for any genre(s) the file
            is labeled with.
            Split represents the split the file belongs to as 'training',
            'validation', or 'test'. If 'validation' and 'test' are not
            both needed, they can be added together to get a single set.
        """
        pass

    def _gather_data_gtzan(self, file_extension, include_labels=True):
        """
        Gather files with the given extension and associate them
        with labels and a test/validate/train split.

        For use with the gtzan dataset. Labels are based on the genre
        folders in the gtzan_dir. Splits are based on the split_vals
        set in the Data Loader constructor, and are simply based on
        the order of the audio files in the folders.

        Parameters
        ----------
        file_extension : str
            The file extension of files we want to gather. Should include
            the leading dot. Examples: '.mp3', '.au', '.c.tfrecord'.
        include_labels : bool, optional
            Whether to include the labels column in the returned results

        Returns
        ------
        pandas.DataFrame
            A dataframe where each row represents a file. Columns are
            'filename', 'label', and 'split'.
            Filename is the full path string to the file.
            Label is an array of 0s with 1s for any genre(s) the file
            is labeled with.
            Split represents the split the file belongs to as 'training',
            'validation', or 'test'. If 'validation' and 'test' are not
            both needed, they can be added together to get a single set.
        """
        file_paths = np.array(
            list(self._gtzan_dir.rglob(f'*{file_extension}')))

        if include_labels:
            genre_labels = [
                self._get_label(self._genre_dict[x.parent.name])
                for x in file_paths
            ]
            data = pd.DataFrame({'filename': file_paths,
                                 'label': genre_labels})
        else:
            data = pd.DataFrame({'filename': file_paths})

        # Assign splits based on each item's index and add
        # them to the dataframe as a new column
        data['split'] = data.index.map(self._assign_split)

        # Change the Path objects to strings
        data['filename'] = data['filename'].map(str)

        return data

    def _gather_data_fma(self, file_extension, include_labels=True):
        """
        Gather files with the given extension and associate them
        with labels and a test/validate/train split.

        For use with the fma dataset. Labels are based on the full genre
        list from the fma genres.csv file. Splits are the default fma splits
        set up by the dataset authors. This includes splitting genres
        representatively and ensuring individual artists do not appear
        across splits. This pulls from the small, medium, or large fma
        subsets. Tracks with no genre_top in the fma tracks.csv
        are ignored, as are any tracks for which the corresponding file
        cannot be found.

        Parameters
        ----------
        file_extension : str
            The file extension of files we want to gather. Should include
            the leading dot. Examples: '.mp3', '.au', '.c.tfrecord'.
        include_labels : bool, optional
            Whether to include the labels column in the returned results

        Returns
        ------
        pandas.DataFrame
            A dataframe where each row represents a file. Columns are
            'filename', 'label', and 'split'.
            Filename is the full path string to the file.
            Label is an array of 0s with 1s for any genre(s) the file
            is labeled with.
            Split represents the split the file belongs to as 'training',
            'validation', or 'test'. If 'validation' and 'test' are not
            both needed, they can be added together to get a single set.
        """
        # Load track data from csv and select only the requested subset
        tracks = pd.read_csv(self._fma_meta_dir / 'tracks.csv',
                             index_col=0,
                             header=1,
                             usecols=[0, 31, 32, 40, 41, 42],
                             skiprows=[2])
        # Setting categorical type here allows us to use the next line.
        tracks['subset'] = tracks['subset'].astype(
            pd.CategoricalDtype(categories=self.FMA_SUBSETS, ordered=True)
        )
        # Now we can select the specified subset and all lower sets.
        # For example, medium subset gives medium and small entries.
        tracks = tracks[tracks['subset'] <= self.fma_set]
        # Prune tracks with invalid genre entries
        tracks = tracks.dropna(subset='genre_top')

        # Treat the index column as a real data column
        tracks = tracks.reset_index()
        # Select only the columns we care about
        tracks = tracks[['index', 'genre_top', 'split']]

        # Map the track ids to their associated filenames as Path objects,
        # including the file_extension being sought.
        tracks['index'] = tracks['index'].map(
            lambda x:
            self._get_filename_fma(x, file_extension)
        )
        # Set more appropriate column names
        tracks = tracks.rename({'index': 'filename',
                                'genre_top': 'label'}, axis=1)

        if include_labels:
            # Map the genres to their associated labels, ready for ML
            tracks['label'] = tracks['label'].map(self._genre_dict)
            tracks['label'] = tracks['label'].map(self._get_label)
        else:
            tracks = tracks.loc[:, tracks.columns != 'label']

        # Remove any entries that don't represent real, present files
        tracks = tracks.loc[tracks['filename'].apply(lambda x: x.is_file())]

        # Change the Path objects to strings
        tracks['filename'] = tracks['filename'].map(str)

        return tracks

    def _assign_split(self, id):
        """
        Assign a split to a gtzan track based on self.split_vals.

        Parameters
        ----------
        id : int
            Used to determine which split to assign the track to. Just
            used for ordering and does not have to be meaningful or
            permanently associated with the track.

        Returns
        ------
        str
            the split assignment of the track as 'training', 'validation',
            or 'test'
        """
        split = ''

        # Assign split based on define split ratio
        if id % 100 < self.split_vals[0]:
            split = 'training'
        elif id % 100 < sum(self.split_vals[:2]):
            split = 'validation'
        elif id % 100 < sum(self.split_vals[:3]):
            split = 'test'

        return split

    def _get_filename_fma(self, file_id, file_extension):
        """
        Generate the associated filename of an fma track based on its
        unique track_id from the tracks.csv file. Actual items gathered
        will be based on the provided file_extension.

        Parameters
        ----------
        file_id : int
            Should be pulled from the tracks.csv file. Uniquely identifies
            the track.
        file_extension : str
            The file extension to add to the file. Should include
            the leading dot. Examples: '.mp3', '.png', '.c.tfrecord'.

        Returns
        ------
        Path
            A Python Path object with the full file path of the file
            corresponding to the file_id and file_extension.
        """
        # Get the folder and filename based on the song ID
        file_id_str = '{:06d}'.format(file_id)
        # Get the full file path based on where the data is stored, appending
        # the folder and filename
        path = self._fma_dir / file_id_str[:3] / file_id_str
        # Add the file extension
        path = path.with_suffix(file_extension)
        return path

    def _get_label(self, genre_id):
        """
        Get the label for a track based on its genre_id.

        Parameters
        ----------
        genre_id : int
            The index representing the genre to associate with this track.

        Returns
        ------
        list[int]
            A list with length corresponding to the number of possible
            genres for the track. Filled with 0s, except for 1s in any
            indexes matching the genre of this track.
        """
        label = [0] * len(self._genre_dict)
        label[genre_id] = 1
        return label

    def get_genre_name(self, genre_id):
        """
        Get the genre name corresponding to a genre_id.

        Parameters
        ----------
        genre_id : int
            The index representing the genre you want to look up the name
            for.

        Returns
        ------
        str
            the genre name
        """
        return self._genre_inverse_dict[genre_id]
