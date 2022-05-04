import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.set_style("dark")
_ = sns.despine()

def get_config():
    cfg = {
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "batch_size": torch.Size([64]),
        "batch_size_1D": torch.Size([64, 1]),
        "val_batch_size": torch.Size([100]),
        "num_epochs": 500,
        "w_gp": 10,
        "c_Clip": 0.01,
        "lr_Clip": 5e-4,
        "lr_GP": 1e-3,
        "betas_GP": (0, 0.9),
        "lr_c": 1e-4,
        "lr_c_eps": 1e-4,
        "(c,eps)_eps": 1,
        "Sinkhorn_eps": 0.01,
        "size": 3,
        "train_epoches": 500,
        "val_epoches": 100,
    }
    return cfg

def get_MLP(input_size=1, hidden_size=128, device='cpu'):
    MLP = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)).to(device)
    return MLP

def print_loss(losses, nb_train):
    plt.figure(figsize=(15, 10))
    for key, loss in losses.items():
        linestyle = ":" if key in ["Sinkhorn", "Wasserstain 1"] else None
        linewidth = 2 if key in ["Sinkhorn", "Wasserstain 1"] else 3
        alpha = 1 if key in ["Sinkhorn", "Wasserstain 1"] else 0.5
        plt.plot(loss, 
                 label=key, 
                 linestyle=linestyle,
                 linewidth=linewidth,
                 alpha=alpha
                )

    plt.axvline(nb_train, c="black", linestyle=linestyle)
    plt.legend()
    plt.show()