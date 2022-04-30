import torch


def validate(model, test_loader, device, criterion):
    with torch.no_grad():
        rmse, mae, loss_all = 0., 0., 0
        step = 0
        for img, label, _ in test_loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            score = model(img)
            loss = criterion(score, label)
            loss_all += loss.item()
            rmse += torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            mae += torch.abs(score - label).mean().item()
            step += 1
        rmse /= step
        mae /= step
        loss_all /= step
    return rmse, mae, loss_all
