import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def random_initialization(seed: int = 2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)


def get_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def feature_extract(model: nn.Module, data_loader: DataLoader, device: torch.device):
    embedding_list, label_list = [], []
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(data_loader)):
            data, label = data.to(device), label.to(device)
            embedding = model(data)
            embedding_list.append(embedding)
            label_list.append(label)
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    return embedding_list, label_list

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot