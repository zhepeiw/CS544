"""!
@brief SI SDR computation given a source mixture
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
import librosa
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../')
sys.path.insert(0, root_dir)
import data_loader.data_loader as data_loader


def bss_eval(sep, i, sources, eps=10e-9):
    # Current target
    min_len = min([len(sep), len(sources[i])])
    sources = sources[:, :min_len]
    sep = sep[:min_len]
    target = sources[i]

    # Target contribution
    s_target = target * (np.dot(target, sep.T) /
                         (np.dot(target, target.T) + eps))

    # Interference contribution
    pse = np.dot(np.dot( sources, sep.T),
    np.linalg.inv(np.dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif = sep - pse

    # Interference + artifacts contribution
    e_total = e_interf + e_artif

    # Computation of the log energy ratios
    sdr = 10*np.log10(sum(s_target**2) / (sum(e_total**2)+eps))
    sir = 10*np.log10(sum(s_target**2) / (sum(e_interf**2)+eps))
    sar = 10*np.log10(sum((s_target + e_interf)**2) /
                      (sum(e_artif**2)+eps))

    # Done!
    return sdr, sir, sar


def naive_cpu_bss_eval(embedding_labels,
                       mix_real_tf,
                       mix_imag_tf,
                       sources_raw,
                       n_sources):

    mix_stft = mix_real_tf + 1j*mix_imag_tf

    if mix_stft.shape == embedding_labels.shape:
        embedding_clustered = embedding_labels
    else:
        embedding_clustered = embedding_labels.reshape(
                              mix_stft.shape[::-1]).T

    sdr_t, sir_t, sar_t = 0., 0., 0.
    for i in np.arange(n_sources):
        embed_mask = mix_stft*(embedding_clustered == i)
        reconstructed = librosa.core.istft(embed_mask,
                                           hop_length=128,
                                           win_length=512)
        bss_results = [bss_eval(reconstructed, j, sources_raw)
                       for j in np.arange(n_sources)]

        sdr, sir, sar = sorted(bss_results, key=lambda x: x[0])[-1]
        sdr_t += sdr
        sir_t += sir
        sar_t += sar

        print(sdr, sir, sar)

        # save_p = '/home/thymios/wavs/'
        # wav_p = os.path.join(save_p,
        #                      'batch_{}_source_{}'.format(
        #                          batch_index + 1, i + 1))
        # librosa.output.write_wav(wav_p, reconstructed, 16000)

    return sdr_t/n_sources, sir_t/n_sources, sar_t/n_sources


def mixture_bss_eval(mix_real_tf,
                     mix_imag_tf,
                     sources_raw,
                     n_sources):

    mix_stft = mix_real_tf + 1j*mix_imag_tf

    reconstructed = librosa.core.istft(mix_stft,
                                       hop_length=128,
                                       win_length=512)
    bss_results = [bss_eval(reconstructed, j, sources_raw)
                   for j in np.arange(n_sources)]

    (sdrs, sirs, sars) = (np.array([x[0] for x in bss_results]),
                          np.array([x[1] for x in bss_results]),
                          np.array([x[2] for x in bss_results]))

    return np.mean(sdrs), np.mean(sirs), np.mean(sars)


def evaluate_masks(mix_stft,
                   clean_sources,
                   mask_1,
                   mask_2,
                   n_sources=2):

    sdr1, sir1, sar1 = naive_cpu_bss_eval(mask_1,
                                          np.real(mix_stft),
                                          np.imag(mix_stft),
                                          clean_sources,
                                          n_sources)

    sdr2, sir2, sar2 = naive_cpu_bss_eval(mask_2,
                                          np.real(mix_stft),
                                          np.imag(mix_stft),
                                          clean_sources,
                                          n_sources)

    return (sdr1 + sdr2)/2, (sir1 + sir2)/2, (sar1 + sar2)/2

def example_of_usage(dataset_path='/mnt/data/CS544_data/'
                     'timit_5400_1800_512_2_fm_random_taus_delays/val',
                     return_items=['mixture_wav',
                                   'clean_sources_wavs',
                                   'ds',
                                   'rpd']):
    """!
    Simple example of how to use this pytorch data loader"""
    default_parameters = data_loader.get_args()
    # lets change the list of the return items:
    default_parameters.return_items = return_items
    default_parameters.input_dataset_p = dataset_path
    data_gen = data_loader.get_numpy_data_generator(default_parameters)

    flag = False
    for batch_data_list in data_gen:
        # the returned elements are tensors
        # Always the first dimension is the selected batch size

        numpy_data_list = data_loader.convert_to_numpy(batch_data_list)
        mix_wav, clean_wavs, mask, phase_diff = numpy_data_list
        mix_stft = librosa.core.stft(mix_wav,
                                     n_fft=512,
                                     win_length=512,
                                     hop_length=128)
        numpy_data_list.append(mix_stft)

        if not flag:
            for el, name in zip(numpy_data_list,
                                return_items+['mix_stft']):
                print(name, type(el), el.shape)
            flag = True

        ibm_1 = (mask == 0)
        ibm_2 = (mask == 1)

        sdr, sir, sar = evaluate_masks(mix_stft,
                                       clean_wavs,
                                       ibm_1,
                                       ibm_2,
                                       n_sources=2)

        print(sdr, sir, sar)


if __name__ == "__main__":
    example_of_usage()
