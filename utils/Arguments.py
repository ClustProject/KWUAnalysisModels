""" KWUAnalysisModels Utility Package. """
import easydict
class __args:
    def __init__(self):
        self._args = self._args

        try:
            _ = self._args.bus_id
        except:
            print("Initialization Error: Set the args first.")

    def init_args(self):
        self._args = {}

    def get_args(self):
        return self._args

    def set_args(self, key: str, value):  # Corrected type hint here
        self._args[key] = value
        return


class seed_args(__args):
    def __init__(self):
        super().__init__()
        self._args = None
        self.init_args()

    def init_args(self):
        self._args = easydict.EasyDict({
            # args for setting device
            'bus_id': 'PCI_BUS_ID',
            'cuda_id': ['0', '1', '2'],

            # args for path
            'os_path': '/home/neuroai/users/dhkim/eer/SSLGCN',
            'seed_data_dir_path': 'dataset/seed/SEED_EEG/ExtractedFeatures/data/',
            'feature_name1': 'de_LDS',
            'feature_name2': 'psd_LDS',
            'seed_label_dir_path': 'dataset/seed/SEED_EEG/ExtractedFeatures/label/',
            'figure_save_path': '/home/neuroai/users/dhkim/eer/SSLGCN/store/figure/',
            'tensor_save_path': '/home/neuroai/users/dhkim/eer/SSLGCN/store/tensor/',
            'model_save_path': '/home/neuroai/users/dhkim/eer/SSLGCN/store/model/',

            # args for counts
            'n_subjects': 15,
            'n_sessions': 3,
            'n_trials': 15,
            'n_nodes': 62,
            'n_features': 5,
            'n_samples': 10182,
            'n_labels_by_trials1': 4,
            'n_labels_by_trials2': 6,
            'n_labels_by_trials3': 8,

            # args for running algorithm
            'seed': 2023,
            'EEG_band': None,
            'pca_components1': 9,
            'pca_components2': 6,
            'essm_lambda': 0.9,
            'de_k': 3394,  # 721
            'psd_k': 3394,  # 1861
            'k1': 30,
            'k2': 130,
            't1': 1,
            't2': 1,
            'feature_dimension': 620,
            'gcn_hid_channels': 64,
            'gcn_out_channels': 128,
            'out_channels': 3,
            'learning_rate': 0.005,
            'l2_lambda': 0.001,
            'epochs': 3000,
            'proj_hid_channels': 32,
            'ptau': 32,
            'pf1': 0.1,
            'pf2': 0.1,
            'pe1': 0.1,
            'pe2': 0.1,
            'tpf1': 0.7,
            'tpf2': 0.7,
            'tpe1': 0.7,
            'tpe2': 0.7,
            'loss_lambda': 0.01,
            'patience': 10,
            'val_split': 0.2
        })

class seedIV_args(__args):
    def __init__(self):
        super().__init__()
        self._args = None

        self.init_args()

    def init_args(self):
        self_args = easydict.EasyDict({
            # args for setting device
            'bus_id': 'PCI_BUS_ID',
            'cuda_id': ['0', '1', '2'],

            # args for path
            'os_path': '/home/neuroai/users/dhkim/eer/SSLGCN',
            'feature_name1': 'de_LDS',
            'feature_name2': 'psd_LDS',
            'seedIV_data_dir_path': 'dataset/seed_IV/eeg_feature_smooth/',
            'figure_save_path': 'store_seedIV/figure/',
            'tensor_save_path': 'store_seedIV/tensor/',
            'model_save_path': 'store_seedIV/model/',

            # args for counts
            'n_subjects': 15,
            'n_sessions': 3,
            'n_trials': 24,
            'n_nodes': 62,
            'n_features': 5,
            'n_samples': 2505,
            'n_labels_by_class1': 15,
            'n_labels_by_class2': 20,
            'n_labels_by_class3': 25,

            # args for running algorithm
            'seed': 2023,
            'EEG_band': None,
            'pca_components1': 9,
            'pca_components2': 6,
            'essm_lambda': 0.9,
            'de_k': 626,
            'psd_k': 626,
            'k1': 30,
            'k2': 130,
            't1': 1,
            't2': 1,
            'feature_dimension': 620,
            'gcn_hid_channels': 256,
            'gcn_out_channels': 64,
            'out_channels': 4,
            'learning_rate': 0.005,
            'l2_lambda': 0.001,
            'epochs': 3000,
            'proj_hid_channels': 16,
            'ptau': 0.7,
            'pf1': 0.1,
            'pf2': 0.2,
            'pe1': 0.1,
            'pe2': 0.2,
            'tpf1': 0.7,
            'tpf2': 0.7,
            'tpe1': 0.7,
            'tpe2': 0.7,
            'loss_lambda': 0.01,
            'patience': 10,
            'val_split': 0.2
        })

class deap_args(__args):
    def __init__(self):
        super().__init__()
        self._args = None

        self.init_args()

    def init_args(self):
        self._args = easydict.EasyDict({
            # args for setting device
            'bus_id': 'PCI_BUS_ID',
            'cuda_id': ['0', '1', '2'],

            # args for path
            'os_path': '/home/neuroai/users/dhkim/eer/SSLGCN',
            'feature_name1': 'DE_LDS_data',
            'feature_name2': 'PSD_LDS_data',
            'deap_label_dir_path': 'dataset/deap/data_preprocessed_matlab/',
            'deap_data_dir_path': 'dataset/deap/extractedfeatures/de_psd_lds/',
            'figure_save_path': 'store_deap/figure/',
            'tensor_save_path': 'store_deap/tensor/',
            'model_save_path': 'store_deap/model/',
            'valence': 'Valence',
            'arousal': 'Arousal',

            # args for counts
            'n_subjects': 32,
            'n_trials': 40,
            'n_nodes': 32,
            'n_features': 4,
            'n_samples': 2520,
            'n_labels_by_class1': 60,
            'n_labels_by_class2': 90,
            'n_labels_by_class3': 120,
            'n_labels': 2,

            # args for running algorithm
            'seed': 2023,
            'EEG_band': None,
            'pca_components1': 9,
            'pca_components2': 6,
            'essm_lambda': 0.9,
            'de_k': 1200,
            'psd_k': 1200,
            'k1': 30,
            'k2': 130,
            't1': 1,
            't2': 1,
            'feature_dimension': 256,
            'gcn_hid_channels': 128,
            'gcn_out_channels': 64,
            'out_channels': 2,
            'learning_rate': 0.005,
            'l2_lambda': 0.001,
            'epochs': 3000,
            'proj_hid_channels': 16,
            'ptau': 0.7,
            'pf1': 0.1,
            'pf2': 0.2,
            'pe1': 0.1,
            'pe2': 0.2,
            'tpf1': 0.7,
            'tpf2': 0.7,
            'tpe1': 0.7,
            'tpe2': 0.7,
            'loss_lambda': 0.01,
            'patience': 10,
            'val_split': 0.2
        })

class mdeap_args(__args):
    def __init__(self):
        super().__init__()
        self._args = None

        self.init_args()

    def init_args(self):
        self.args = easydict.EasyDict({
            # args for setting device
            'bus_id': 'PCI_BUS_ID',
            'cuda_id': ['0', '1', '2'],

            # args for path
            'os_path': '/home/neuroai/users/dhkim/eer/SSLGCN',
            'feature_name1': 'DE_LDS_data',
            'feature_name2': 'PSD_LDS_data',
            'deap_label_dir_path': 'dataset/deap/data_preprocessed_matlab/',
            'deap_data_dir_path': 'dataset/deap/mmextractedfeatures/de_psd_lds/',
            'figure_save_path': 'store_deap/figure/',
            'tensor_save_path': 'store_deap/tensor/',
            'model_save_path': 'store_deap/model/',
            'valence': 'Valence',
            'arousal': 'Arousal',

            # args for counts
            'n_subjects': 32,
            'n_trials': 40,
            'n_nodes': 34,
            'idx_nodes_eeg': 32,
            'idx_nodes_gsr': 32,
            'idx_nodes_ppg': 33,
            'n_features': 4,
            'n_samples': 2520,
            'n_labels_by_class1': 60,
            'n_labels_by_class2': 90,
            'n_labels_by_class3': 120,
            'n_labels': 2,

            # args for running algorithm
            'seed': 2023,
            'EEG_band': None,
            'pca_components1': 9,
            'pca_components2': 6,
            'essm_lambda': 0.9,
            'de_k': 1200,
            'psd_k': 1200,
            'k1': 30,
            'k2': 130,
            't1': 1,
            't2': 1,
            'feature_dimension': 272,
            'gcn_hid_channels': 128,
            'gcn_out_channels': 64,
            'out_channels': 2,
            'learning_rate': 0.005,
            'l2_lambda': 0.001,
            'epochs': 3000,
            'proj_hid_channels': 16,
            'ptau': 0.7,
            'pf1': 0.1,
            'pf2': 0.2,
            'pe1': 0.1,
            'pe2': 0.2,
            'tpf1': 0.7,
            'tpf2': 0.7,
            'tpe1': 0.7,
            'tpe2': 0.7,
            'loss_lambda': 0.01,
            'patience': 10,
            'val_split': 0.2
        })
