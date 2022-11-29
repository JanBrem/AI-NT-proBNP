import torch

class TrainGlobalConfig:
    device = torch.device("cuda")

    df_path = "./sample_data/sample_csv.csv"

    TARGET = "proBNP"

    num_workers = 1
    batch_size = 16
    n_epochs = 10
    lr = 0.001

    folder = 'weights/proBNP/'
    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.7,
        patience=5,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0.001,
        eps=1e-08
    )
