import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import PIL
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel

print(PIL.__version__)

torch.manual_seed(42)
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
])


class EncoderDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        return image
    

ds = load_dataset("nlphuji/flickr30k")
test_dataset = ds['test']

def apply_transform(example):
    example['image'] = transform(example['image'])
    return example


# splitting the dataset as it only has one split
split_dataset = test_dataset.train_test_split(test_size=0.2, seed=42)


train_dataset = EncoderDataset(split_dataset['train']['image'])
test_dataset = EncoderDataset(split_dataset['test']['image'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_embeddings = []

with torch.no_grad():
    for images in train_loader:
        inputs = processor(images=list(images), return_tensors="pt", padding=True)

        outputs = model(**inputs)
        all_embeddings.append(outputs)

all_embeddings = torch.cat(all_embeddings, dim=0)
torch.save(all_embeddings, "clip_image_embeddings.pt")



