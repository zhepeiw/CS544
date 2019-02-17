"""!
@brief Dataloader for 2 dummy wavs for source separation.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import scipy.io.wavfile as wavfile
import glob2
import numpy as np


def get_source_wavs(n_samples=16000,
                    wavs_dir=None,
                    normalize_by_mean=True,
                    normalize_by_std=False,
                    mixing_matrix=None):
    """! Simple two sources mixing and return

    :param mixing_matrix: 2D numpy array sum of amplitudes across
    rows should add up to 1. If None is given then a random matrix
    would be generated.
    :param n_samples: Number of audio samples that you want to be
    returned for both of the sources. Always the audio clips would be
    cropped at the edges in order to match this parameter
    :param normalize_by_mean: Remove DC coefficient
    :param normalize_by_std: Divide by the std of each signal
    :param wavs_dir: The absolute path where the wavs can be loaded

    :return:
    mixtures as a numpy array of shape (2, n_samples) using the
    mixing matrix.
    sources as a numpy matrix of shape (2, n_samples) without the
    amplitudes of the mixing matrix in front.
    mixing_matrix as a 2x2 matrix that was used for the
    """

    if wavs_dir is None:
        wavs_dir = os.path.join(
                   os.path.dirname(os.path.realpath(__file__)),
                                   '../wavs_for_separation/')
    if mixing_matrix is None:
        mixing_matrix = np.array([[0.7, 0.3],
                                  [0.6, 0.4]])

    wavs_paths = glob2.glob(os.path.join(wavs_dir, '*.wav'))
    wavs_paths = wavs_paths[:2]

    speaker_wavs = [wavfile.read(wav_p) for wav_p in wavs_paths]

    if normalize_by_mean:
        speaker_wavs = [(sr, wav - np.mean(wav))
                        for (sr, wav) in speaker_wavs]

    if normalize_by_std:
        speaker_wavs = [(sr, wav / np.std(wav))
                        for (sr, wav) in speaker_wavs]

    cropped_sources = []
    for sr, wav in speaker_wavs:
        n_sample_diff = len(wav) - n_samples
        if n_sample_diff < 0:
            raise ValueError("All wavs should have at least the "
                             "number of samples specified: {}"
                             "".format(n_samples))

        st = int(n_sample_diff/2.)
        end = st + n_samples
        cropped_sources.append(wav[st:end])

    sources = np.array(cropped_sources)
    mixtures = np.matmul(mixing_matrix, sources)

    return mixtures, mixing_matrix, sources


def test_data_gen():
    mixing_matrix = np.array([[0.8, 0.2],
                              [0.6, 0.4]])
    n_samples = 16000
    normalize_by_std = False
    normalize_by_mean = True
    wavs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../wavs_for_separation/')

    mixtures, mixing_matrix, sources = get_source_wavs(
        mixing_matrix=mixing_matrix,
        n_samples=n_samples,
        normalize_by_mean=normalize_by_mean,
        normalize_by_std=normalize_by_std,
        wavs_dir=wavs_dir)

    for i in range(mixing_matrix.shape[0]):
        mixture_pred = (mixing_matrix[i][0] * sources[0] +
                        mixing_matrix[i][1] * sources[1])

        assert np.allclose(mixture_pred, mixtures[i])

    return mixtures, mixing_matrix, sources


if __name__ == "__main__":
    mixtures, mixing_matrix, sources = test_data_gen()
    print(mixtures.shape)
    print(mixing_matrix)
    print(sources.shape)
