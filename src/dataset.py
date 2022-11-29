import numpy as np
import h5py
from neurokit2.signal import signal_filter
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

def load_ecg_sample_hchs(path):
    EKG_LEADS_MASK = ['ECG_I', 'ECG_II', 'ECG_III',
                 'ECG_aVR', 'ECG_aVL', 'ECG_aVF',
                 'ECG_V1', 'ECG_V2', 'ECG_V3', 'ECG_V4', 'ECG_V5', 'ECG_V6',]
    file = h5py.File(path, 'r')
    channel_12_ecg = []
    for lead in EKG_LEADS_MASK:
        arr = np.array(file[lead])
        channel_12_ecg.append(arr)
    file.close()
    return np.array(channel_12_ecg)


def preprocess(ecg, samples=5000, mode="train"):
    # ecg crop <- hchs specific
    if (samples == 5000):
        ecg = ecg[:, ::2]   #todo: das hier mit in hchs specific function (oben)
    else:
        ecg = ecg[:, ::4]
    ecg = ecg.astype(float)

    # apply powerline & butterworth filter per channel
    for i in range(12):
        pl = 50
        clean = signal_filter(signal=ecg[i], sampling_rate=250, lowcut=0.5, method="butterworth", order=5)
        ecg[i] = signal_filter(signal=clean, sampling_rate=250, method="powerline", powerline=pl)

    # random crop
    if (mode == "train"):
        a = np.random.randint(0, ecg.shape[1] - 2048)
    else:
        a = 0
    ecg = ecg[:, a: a + 2048]

    # StandarScaler
    scaler = StandardScaler()

    # Augmentations
    for i in range(12):
        ecg[i] = scaler.fit_transform(ecg[i].reshape(-1, 1)).flatten()
        ecg[i] -= np.median(ecg[i])

        if (mode == "train"):
            one = np.random.uniform(-0.25, 0.25)
            two = np.random.uniform(0.9, 1.2)
            if (np.random.randint(0, 100) < 50):
                ecg[i] += one

            if (np.random.randint(0, 100) < 50):
                ecg[i] *= two

            if (np.random.randint(0, 100) < 5):
                ecg[i] = 0

    ecg = torch.tensor(ecg).float()

    return ecg


class RegressionDataset1dPTB(Dataset):
    def __init__(self, df, indices, target, mode):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.ecg_paths = self.df["ecg_path"].values
        self.targets = self.df[target].values
        self.samples = self.df["num_samples"].values
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.targets[idx]

        target = torch.tensor([target]).float()

        # file = load_ecg_sample_ship(self.ecg_paths[idx])
        file = load_ecg_sample_hchs(self.ecg_paths[idx])

        ecg = file.astype(float)

        ecg = preprocess(ecg, self.samples[idx], mode=self.mode)

        if(self.mode == "test"):
            return ecg
        else:
            return ecg, target