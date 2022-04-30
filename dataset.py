from torch.utils.data import Dataset
from load_data import load
import torchvision
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.path, self.label = load(img_path, label_path)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = Image.open(self.path[idx])
        img = self.transform(img)
        return img, self.label[idx], self.path[idx]

