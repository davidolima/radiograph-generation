import torch, os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class _RadiographAgeSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super(RadiographAgeSet, self).__init__()
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        
        labels = self.labels.iloc[idx, 1:]
        labels = np.array([labels])

        if self.transform:
            image = np.array(image)
            image = self.transform(image)

        return image, labels


class RadiographAgeSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = np.array(image)
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def show_labels(image, labels):
    """Show image with labels"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    root = "/media/david/bb5b899e-a7e9-49dd-b267-5014035bf700/datasets/pan-radiographs/1st-set"
    radiographs = pd.read_csv(root+'/pan-radiographs.csv')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([.5, .5, .5], [.5, .5, .5]),
        T.Resize((128, 128)),
        # Pad(1),
        ]
    )

    dataset = RadiographAgeSet(annotations_file=f"{root}/pan-radiographs.csv",
                               img_dir=f"{root}/images/",
                               transform=transform)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(train_features)
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img[0])
    plt.show()
    print(f"Label: {label}")
