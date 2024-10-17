import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01
import torchvision.transforms as transforms
target_size = (64, 64)

                                        

transform_train = transforms.Compose([ transforms.ToTensor(), To01(),
                                        AssertChannelFirst(),transforms.Resize((128, 128)),])








class s2_CustomDataset(Dataset):
    def __init__(self, csv_file, transform=transform_train):
        self.data = pd.read_csv(csv_file)
        
        
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  
        image = Image.open(img_name).convert("RGB")

        

        if self.transform:
            image = self.transform(image)

        return image















