from model import ResNet18
import torch
import torch.nn as nn
from validate import validate
from tqdm import tqdm


def train(train_loader, test_loader, writer, epochs, lr, device, model_dict):
    best_l = 1000
    model = ResNet18().to(device)
    optimizer_e = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_rmes, train_mae, train_loss = 0., 0., 0.
        step = 0
        loader = tqdm(train_loader)
        for img, label,_ in loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            optimizer_e.zero_grad()
            score = model(img)
            loss = criterion(score, label)
            train_loss += loss.item()
            rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            train_rmes += rmse
            mae = torch.abs(score - label).mean().item()
            train_mae += mae
            loss.backward()
            optimizer_e.step()
            step += 1
            loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
        train_rmes /= step
        train_mae /= step
        train_loss /= step
        model.eval()
        val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
        writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
        if val_loss < best_l:
            torch.save({'ResNet': model.state_dict()}, '{}/ResNet.pth'.format(model_dict))
            print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
            best_l = val_loss
        print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes,
                                                                                   val_mae))
