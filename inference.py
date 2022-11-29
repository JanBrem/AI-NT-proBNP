import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import DataLoader

from src.config import TrainGlobalConfig
from src.model import load_model
from src.dataset import RegressionDataset1dPTB

def inference():

    config = TrainGlobalConfig()

    df = pd.read_csv(config.df_path)
    print(df.shape)
    df = df[(df["split"] == "test") & (df["num_samples"] == 5000)].reset_index(drop = True)
    test_ds = RegressionDataset1dPTB(df, np.arange(len(df)), config.TARGET, "test")

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    weight_arr = [sorted(glob(f"./weights/proBNP/best-auc-fold{i}*"))[-1] for i in range(5)]


    for idx, weight in enumerate(weight_arr):
        model = load_model()

        model.load_state_dict(torch.load(weight)["model_state_dict"])

        model = model.to(config.device)
        model.eval()

        predictions = []

        for ecg in tqdm(test_loader):
            with torch.no_grad():
                ecg = ecg.to(config.device)
                preds = model(ecg)[0].cpu().detach().numpy()
                predictions.append(preds)

        df["pred" + str(idx)] = np.concatenate(np.array(predictions))



#### if not os.path.exists(self.base_dir):
 ###           os.makedirs(self.base_dir)
    df.to_csv("./results/results.csv", index=False)