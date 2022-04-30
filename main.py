from writer import MyWriter
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from train import train
import torch

batch_size = 128
lr = 0.001
epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = './model_dict'
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
writer = MyWriter(log_dir)

dataset_train = MyDataset('./processed/train/', './processed/label.csv')
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                          drop_last=True)

dataset_test = MyDataset('./processed/validate/', './processed/label.csv')
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

train(train_loader, test_loader, writer, epochs, lr, device, model_dir)
