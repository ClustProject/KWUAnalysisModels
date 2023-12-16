import numpy as np


def load_subject_data(args, isdeap=False):
    de = np.load(args.tensor_save_path + 'subject_de.npy')
    psd = np.load(args.tensor_save_path + 'subject_psd.npy')
    label = np.load(args.tensor_save_path + 'subject_label.npy')
    sample_counts = None

    if isdeap == False:
        sample_counts = np.load(args.tensor_save_path + 'subject_sample_counts.npy')

    return de, psd, label, sample_counts


def load_mm_subject_data(args, isdeap=False):
    de = np.load(args.tensor_save_path + 'multimodal/subject_de.npy')
    psd = np.load(args.tensor_save_path + 'multimodal/subject_psd.npy')
    label = np.load(args.tensor_save_path + 'multimodal/subject_label.npy')
    sample_counts = None

    if isdeap == False:
        sample_counts = np.load(args.tensor_save_path + 'multimodal/subject_sample_counts.npy')

    return de, psd, label, sample_counts