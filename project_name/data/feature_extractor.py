import librosa
import numpy as np


class AudioLengthError(Exception):

    def __init__(self, filename, duration, duration_limit):
        self.filename = filename
        self.duration = duration
        self.duration_limit = duration_limit
        super().__init__()

    def __str__(self):
        return (f'\n\tfile {self.filename}'
                f'\n\taudio length = {self.duration}'
                f'\n\trequired minimum audio length = {self.duration_limit}')


class FeatureExtractor:
    """
    Provides a feature extraction interface for audio files.

    Attributes
    ----------
    sample_rate : int
        the sample rate at which to load/resample audio
    audio_length : float
        the length, in seconds, of audio to read
    mono : bool
        whether to load only a single channel from audio

    Methods
    -------
    load_normalized(filename)
        Gets the normalized time series and sample rate from the audio
    extract(filename)
        Performs feature extraction on the audio file
    """

    # Sampling rate in Hz
    SR_LOW = 22050
    SR_STANDARD = 44100
    SR_HIGH = 48000
    # Audio length in seconds
    AU_LEN_STANDARD = 29.9
    # Load audio in single channel
    MONO = True

    def __init__(self,
                 variant,
                 sample_rate=SR_LOW,
                 audio_length=AU_LEN_STANDARD,
                 mono=MONO):
        """
        Parameters
        ----------
        variant : str
            The extraction variant (A, B, or C)
        sample_rate : int, optional
            Rhe sample rate at which to load/resample audio
        audio_length : float, optional
            The length, in seconds, of audio to read
        mono : bool, optional
            Whether to load only a single channel from audio
        """
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.mono = mono

        # Set up extraction variant so the Feature Extractor can be customized
        # for compatibility with each ml algo while maintaining a consistent
        # .extract() interface.
        if variant == "C":
            self.extract = self._extract_C
        else:
            self.extract = None

    def load_normalized(self, filename):
        """
        Wraps librosa.load with normalization to ensure data consistency.

        Ensures consistent sample rate and audio length.

        Parameters
        ----------
        filename : str
            The audio file to load

        Returns
        ------
        numpy.array
            array of floats representing the audio time series
        int
            sample rate of the audio after resampling
        """

        duration = librosa.get_duration(filename=filename)
        if (duration < self.audio_length):
            # raise AudioLengthError(filename, duration, self.audio_length)
            padding = 0.0
        else:
            # Use padding to ensure the audio is taken from the middle of the
            # clip when the audio clip is longer than the sought load length.
            padding = (duration - self.audio_length) / 2

        ts, sr = librosa.load(filename,
                              sr=self.sample_rate,
                              mono=self.mono,
                              duration=self.audio_length,
                              offset=padding)

        # Pad with 0s if the audio is too short
        if (duration < self.audio_length):
            ts = librosa.util.fix_length(ts, size=int(sr * self.audio_length))

        return (ts, sr)

    def _extract_C(self, filename):
        """
        Feature extraction for ML algo C.

        Extracts the log mel spectrogram as a numpy.array of float16.

        Parameters
        ----------
        filename : str
            The audio file to load

        Returns
        ------
        numpy.array
            array of float16 representing the log mel spectrogram
        """

        HOP_LENGTH = 1024
        HOP_LENGTH = HOP_LENGTH if self.sample_rate < 40000 else HOP_LENGTH * 2
        N_FFT = HOP_LENGTH * 2
        N_MELS = 64
        LOG_MEL_REF = np.max
        OUTPUT_VAR_TYPE = np.float16

        ts, sr = self.load_normalized(filename)
        mel = librosa.feature.melspectrogram(y=ts,
                                             sr=sr,
                                             n_fft=N_FFT,
                                             hop_length=HOP_LENGTH,
                                             n_mels=N_MELS)

        log_mel = librosa.power_to_db(mel, ref=LOG_MEL_REF)
        return log_mel.astype(OUTPUT_VAR_TYPE)
