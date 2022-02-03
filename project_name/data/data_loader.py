from pathlib import Path
import numpy as np
import pandas as pd


class DataLoader:
    """
    """

    DATA_DIR = __file__ + '/../../resources'
    DATA_TYPE = 'gtzan'
    FMA_SET = 'small'

    def __init__(self,
                 data_type=DATA_TYPE,
                 data_dir=DATA_DIR,
                 gtzan_dir=None,
                 fma_dir=None,
                 fma_set=FMA_SET):
        """
        """

        self.data_type = data_type
        self.fma_set = fma_set

        # Set the data directory as a Path object and resolve
        # data-containing folders dependent upon input parameters
        data_dir = Path(data_dir)
        if gtzan_dir is None:
            self.gtzan_dir = data_dir / 'gtzan' / 'genres'
        else:
            self.gtzan_dir = Path(gtzan_dir)

        if fma_dir is None:
            self.fma_dir = data_dir / 'fma' / f'fma_{self.fma_set}'
        else:
            self.fma_dir = Path(fma_dir)

        self.fma_meta_dir = self.fma_dir / '..' / 'fma_metadata'

        # Set up data type so the Data Processor can be customized
        # for compatibility with different datasets while maintaining a
        # consistent interface.
        if data_type == 'gtzan':
            self.gather_data = self._gather_data_gtzan
            self.genre_dict = {
                v.name: i for i, v in enumerate(self.gtzan_dir.iterdir())}

        elif data_type == 'fma':
            self.gather_data = self._gather_data_fma
            genres = pd.read_csv(self.fma_meta_dir / 'genres.csv',
                                 index_col=0,
                                 usecols=['genre_id', 'title'])
            self.genre_dict = {v: i for i, v in enumerate(genres['title'])}

        else:
            self.gather_data = None
            self.genre_dict = {}

        self.genre_inverse_dict = {v: k for k, v in self.genre_dict.items()}

    def _gather_data_gtzan(self, file_extension, include_labels=True):
        """
        """
        file_paths = np.array(
            list(self.gtzan_dir.rglob(f'*.{file_extension}')))

        if include_labels:
            genre_labels = [self.get_label(
                self.genre_dict[x.parent.name]) for x in file_paths]
            data = pd.DataFrame({'data': file_paths, 'label': genre_labels})
        else:
            data = pd.DataFrame({'data': file_paths})

        # Change the Path objects to strings
        data['data'] = data['data'].map(str)

        return data

    def _gather_data_fma(self, file_extension, include_labels=True):
        """
        """
        # Load track data from csv and select only the requested subset
        tracks = pd.read_csv(self.fma_meta_dir / 'tracks.csv',
                             index_col=0,
                             header=1,
                             usecols=[0, 32, 40, 41, 42],
                             skiprows=[2])
        tracks = tracks[tracks['subset'] == self.fma_set]
        # Prune tracks with invalid genre entries
        tracks = tracks.dropna(subset='genre_top')

        # Treat the index column as a real data column
        tracks = tracks.reset_index()
        # Select only the columns we care about
        tracks = tracks[['index', 'genre_top']]

        # Map the track ids to their associated filenames
        tracks['index'] = tracks['index'].map(self._get_filename_fma)
        file_extender_str = '{{}}.{}'.format(file_extension)
        tracks['index'] = tracks['index'].map(file_extender_str.format)
        # More appropriate column names
        tracks = tracks.rename({'index': 'data', 'genre_top': 'label'}, axis=1)

        if include_labels:
            # Map the genres to their associated labels, ready for ML
            tracks['label'] = tracks['label'].map(self.genre_dict)
            tracks['label'] = tracks['label'].map(self.get_label)
        else:
            tracks = tracks[['data']]

        # Change the Path objects to strings
        tracks['data'] = tracks['data'].map(str)

        return tracks

    def _get_filename_fma(self, file_id):
        """
        """
        file_id_str = '{:06d}'.format(file_id)
        return self.fma_dir / file_id_str[:3] / file_id_str

    def get_label(self, genre_id):
        """
        """
        label = [0] * len(self.genre_dict)
        label[genre_id] = 1
        return label

    def get_genre_name(self, genre_id):
        """
        """
        return self.genre_inverse_dict[genre_id]
