import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from src.config import TrainGlobalConfig
from src.model import load_model
from src.dataset import RegressionDataset1dPTB
from src.engine import Fitter

def train():
    config = TrainGlobalConfig()

    for fold in range(5):
        model = load_model()

        model = model.to(config.device)

        df = pd.read_csv(config.df_path)

        train_df = df[(df["fold"] != fold) & (df["split"] == "train")]
        val_df = df[(df["fold"] == fold) & (df["split"] == "train")]

        train_ds = RegressionDataset1dPTB(train_df, np.arange(len(train_df)), config.TARGET, "train")
        val_ds = RegressionDataset1dPTB(val_df, np.arange(len(val_df)), config.TARGET, "val")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, persistent_workers=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, persistent_workers=True, num_workers=config.num_workers)

        fitter = Fitter(model, config.device, config, fold=fold)
        fitter.fit(train_loader, val_loader)
