from model import ResNet18
import torch
from dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

batch_size = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18()
model_dict = torch.load('./model_dict/ResNet.pth', map_location=device)
model.load_state_dict(model_dict['ResNet'])

dataset_test = MyDataset('./processed/test/', './processed/label.csv')
num_test = len(dataset_test)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True,
                         drop_last=True)

rmse, mae = 0., 0.
step = 0
paths, labels, predicts = [], [], []
with torch.no_grad():
    loader = tqdm(test_loader)
    for img, label, path in loader:
        paths += list(path)
        labels += torch.flatten(label).tolist()
        img, label = img.to(device), label.to(device).to(torch.float32)
        predict = model(img)
        predicts += torch.flatten(predict).tolist()
        rmse += torch.sqrt(torch.pow(torch.abs(predict - label), 2).mean()).item()
        mae += torch.abs(predict - label).mean().item()
        step += 1
        loader.set_description('step:{} {}/{}'.format(step, step * batch_size, num_test))
    rmse /= step
    mae /= step
print('Test\tMAE:{}\t RMSE:{}'.format(mae, rmse))
pd.DataFrame({'file': paths, 'label': labels, 'predict': predicts}).to_csv('testInfo.csv', index=False)
