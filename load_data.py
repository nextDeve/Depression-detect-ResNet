import pandas as pd
from preprocess import get_files, get_dirs
import torch
import numpy as np


def load(img_path, label_path):
    label = pd.read_csv(label_path)
    types = ['Freeform', 'Northwind']
    paths, labels = [], []
    for t in types:
        dirs = get_dirs(img_path + t)
        for d in dirs:
            no = d[-5:]
            l = label[label['file'] == no]['label'].to_numpy()[0]
            files = get_files(d)
            for file in files:
                paths.append(d + '/' + file)
                labels.append(l)
    return paths, torch.from_numpy(np.array(labels)).view((len(labels), 1))
