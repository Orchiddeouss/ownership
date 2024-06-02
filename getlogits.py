import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler


def get_logits(model, train_loader, device):
    # get logits vector from model
    logits_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            logits_list.append(output.cpu().numpy())

    return logits_list